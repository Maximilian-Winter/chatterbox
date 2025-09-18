# Copyright (c) 2025 Resemble AI
# MIT License
import logging
from typing import Union, Optional, List

logger = logging.getLogger(__name__)

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import LlamaModel, LlamaConfig
from transformers.generation.logits_process import TopPLogitsWarper, RepetitionPenaltyLogitsProcessor, MinPLogitsWarper

from .modules.learned_pos_emb import LearnedPositionEmbeddings

from .modules.cond_enc import T3CondEnc, T3Cond
from .modules.t3_config import T3Config
from .llama_configs import LLAMA_CONFIGS
from .inference.t3_hf_backend import T3HuggingfaceBackend
from .inference.alignment_stream_analyzer import AlignmentStreamAnalyzer
from .batch_state import BatchGenerationState, BatchKVCache
from ..utils import AttrDict


logger = logging.getLogger(__name__)


def _ensure_BOT_EOT(text_tokens: Tensor, hp):
    B = text_tokens.size(0)
    assert (text_tokens == hp.start_text_token).int().sum() >= B, "missing start_text_token"
    assert (text_tokens == hp.stop_text_token).int().sum() >= B, "missing stop_text_token"


class T3(nn.Module):
    """
    Token-To-Token (T3) TTS model using huggingface transformer models as backbones,
        * tokenization, including start / stop tokens are always added externally to this class
        * conditioning data like CLAP, emotion, etc are all in a separate file for more modularity
        * careful! this class assumes relative positional encoding -- with absolute PE, we would at
            least want to reset the position to 0 when speech tokens begin, and optionally use a
            different PE embedding space for speech.
    """

    def __init__(self, hp=None):
        if hp is None:
            hp = T3Config.english_only()  # Default to English-only config for backward compatibility
        super().__init__()
        self.hp = hp
        self.cfg = LlamaConfig(**LLAMA_CONFIGS[hp.llama_config_name])
        self.tfmr = LlamaModel(self.cfg)
        self.dim = self.cfg.hidden_size
        self.deepspeed_patch_applied = False

        # conditioning / embedding
        self.cond_enc = T3CondEnc(hp)
        self.text_emb = nn.Embedding(hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(hp.speech_tokens_dict_size, self.dim)

        # custom position embedding
        if hp.input_pos_emb == "learned":
            max_text_seq_len = hp.max_text_tokens + 2
            self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, self.dim)

            max_mel_seq_len = hp.max_speech_tokens + 2 + 2
            self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

        # logit projection
        self.text_head = nn.Linear(self.cfg.hidden_size, hp.text_tokens_dict_size, bias=False)
        self.speech_head = nn.Linear(self.cfg.hidden_size, hp.speech_tokens_dict_size, bias=False)
        self.compiled = False

    @property
    def device(self):
        return self.speech_head.weight.device

    def prepare_conditioning(self, t3_cond: T3Cond):
        """
        Token cond data needs to be embedded, so that needs to be here instead of in `T3CondEnc`.
        """
        if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
            t3_cond.cond_prompt_speech_emb = self.speech_emb(t3_cond.cond_prompt_speech_tokens) + \
                self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
        return self.cond_enc(t3_cond)  # (B, len_cond, dim)

    def prepare_input_embeds(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        cfg_weight: float = 0.0,
    ):
        # prepare input embeddings (skip backbone tranformer embeddings)
        cond_emb = self.prepare_conditioning(t3_cond)  # (B, len_cond, dim)
        text_emb = self.text_emb(text_tokens)  # (B, len_text, dim)
        if cfg_weight > 0.0:
            text_emb[1].zero_()  # CFG uncond

        speech_emb = self.speech_emb(speech_tokens)  # (B, len_speech, dim)
        if self.hp.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)
            speech_emb = speech_emb + self.speech_pos_emb(speech_tokens)
        len_cond = cond_emb.size(1)

        if cond_emb.size(0) != text_emb.size(0):
             cond_emb = cond_emb.expand(text_emb.size(0), -1, -1)

        # concat
        embeds = torch.stack([
            torch.cat((ce, te, se))
            for ce, te, se in zip(cond_emb, text_emb, speech_emb)
        ])  # (B, length, dim)
        return embeds, len_cond

    def forward(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        speech_token_lens: torch.LongTensor,
        training=False,
    ):
        _ensure_BOT_EOT(text_tokens, self.hp)

        # prepare custom input embeds
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
        )

        # backbone tranformer forward
        tfmr_out = self.tfmr.forward(
            input_ids=None,
            # position_ids=position_ids, # TODO? ROPE should be fine?
            inputs_embeds=embeds,
            output_hidden_states=True,
            return_dict=True,
            use_cache=(not training),
        )
        hidden_states = tfmr_out.hidden_states[-1]  # final tfmr layer output, (B, seq, dim)

        # post-processing: splice out text and speech parts of hidden states
        len_text = text_tokens.size(1)
        len_speech = speech_tokens.size(1)
        B, _, dim = hidden_states.shape
        device, dtype = hidden_states.device, hidden_states.dtype
        text_latents = torch.zeros(B, len_text, dim, dtype=dtype, device=device)
        speech_latents = torch.zeros(B, len_speech, dim, dtype=dtype, device=device)
        ttl, stl = text_token_lens, speech_token_lens
        for i in range(B):
            text_end = len_cond + ttl[i].item()
            speech_start = len_cond + text_tokens.size(1)
            speech_end = speech_start + stl[i].item()
            text_latents[i, :ttl[i]] = hidden_states[i, len_cond:text_end]
            speech_latents[i, :stl[i]] = hidden_states[i, speech_start:speech_end]

        # logit projection
        text_logits = self.text_head(text_latents)
        speech_logits = self.speech_head(speech_latents)

        return AttrDict(
            text_logits=text_logits,
            text_latents=text_latents,
            speech_logits=speech_logits,
            speech_latents=speech_latents,
            hidden_states=hidden_states,
        )

    def loss(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        speech_token_lens: torch.LongTensor,
    ):
        "training method"
        len_text = text_tokens.size(1)
        len_speech = speech_tokens.size(1)
        assert len_text == text_token_lens.max()
        assert len_speech == speech_token_lens.max()

        out = self.forward(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            training=True,
        )  # (B, seq, vocab_size)

        # Calc CCE losses
        IGNORE_ID = -100
        device = out.text_logits.device
        mask_text = torch.arange(len_text, device=device)[None] >= text_token_lens[:, None]  # (B, len_text)
        mask_speech = torch.arange(len_speech, device=device)[None] >= speech_token_lens[:, None]  # (B, len_speech)
        masked_text = text_tokens.masked_fill(mask_text, IGNORE_ID)
        masked_speech = speech_tokens.masked_fill(mask_speech, IGNORE_ID)
        loss_text = F.cross_entropy(out.text_logits, masked_text, ignore_index=IGNORE_ID)
        loss_speech = F.cross_entropy(out.speech_logits, masked_speech, ignore_index=IGNORE_ID)

        return loss_text, loss_speech

    @torch.inference_mode()
    def inference(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: Tensor,
        initial_speech_tokens: Optional[Tensor]=None,

        # misc conditioning
        prepend_prompt_speech_tokens: Optional[Tensor]=None,

        # HF generate args
        num_return_sequences=1,
        max_new_tokens=None,
        stop_on_eos=True,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        min_p=0.05,
        length_penalty=1.0,
        repetition_penalty=1.2,
        cfg_weight=0.5,
    ):
        """
        Args:
            text_tokens: a 1D (unbatched) or 2D (batched) tensor.
        """
        # Validate / sanitize inputs
        assert prepend_prompt_speech_tokens is None, "not implemented"
        _ensure_BOT_EOT(text_tokens, self.hp)
        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)

        # Default initial speech to a single start-of-speech token
        if initial_speech_tokens is None:
            initial_speech_tokens = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])

        # Prepare custom input embeds
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech_tokens,
            cfg_weight=cfg_weight,
        )

        # In order to use the standard HF generate method, we need to extend some methods to inject our custom logic
        # Note the llama-specific logic. Other tfmr types can be added later.

        self.compiled = False

        # TODO? synchronize the expensive compile function
        # with self.compile_lock:
        if not self.compiled:
            # Default to None for English models, only create for multilingual
            alignment_stream_analyzer = None
            if self.hp.is_multilingual:
                alignment_stream_analyzer = AlignmentStreamAnalyzer(
                    self.tfmr,
                    None,
                    text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
                    alignment_layer_idx=9, # TODO: hparam or something?
                    eos_idx=self.hp.stop_speech_token,
                )
                assert alignment_stream_analyzer.eos_idx == self.hp.stop_speech_token

            patched_model = T3HuggingfaceBackend(
                config=self.cfg,
                llama=self.tfmr,
                speech_enc=self.speech_emb,
                speech_head=self.speech_head,
                alignment_stream_analyzer=alignment_stream_analyzer,
            )
            self.patched_model = patched_model
            self.compiled = True

        # # Run normal generate method, which calls our custom extended methods
        # return self.patched_model.generate(
        #     inputs=initial_speech_tokens,
        #     decoder_cond=embeds,
        #     bos_token_id=self.hp.start_speech_token,
        #     eos_token_id=(self.hp.stop_speech_token if stop_on_eos else -1),
        #     pad_token_id=self.hp.stop_speech_token,
        #     max_new_tokens=max_new_tokens or self.hp.max_speech_tokens,
        #     num_return_sequences=num_return_sequences,
        #     temperature=temperature,
        #     min_p=min_p,
        #     length_penalty=length_penalty,
        #     repetition_penalty=repetition_penalty,
        #     do_sample=do_sample,
        #     # cache_implementation=None if not self.compiled else "static",
        # )

        device = embeds.device

        bos_token = torch.tensor([[self.hp.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = self.speech_emb(bos_token)  # shape: (B, 1, embed_dim)
        bos_embed = bos_embed + self.speech_pos_emb.get_fixed_embedding(0)

        # batch_size=2 for CFG
        bos_embed = torch.cat([bos_embed, bos_embed])

        # Combine condition and BOS token for the initial input
        inputs_embeds = torch.cat([embeds, bos_embed], dim=1)

        # Track generated token ids; start with the BOS token.
        generated_ids = bos_token.clone()
        predicted = []  # To store the predicted tokens

        # Instantiate the logits processors.
        top_p_warper = TopPLogitsWarper(top_p=top_p)
        min_p_warper = MinPLogitsWarper(min_p=min_p)
        top_p_warper = TopPLogitsWarper(top_p=top_p)
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))

        # ---- Initial Forward Pass (no kv_cache yet) ----
        output = self.patched_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        # Initialize kv_cache with the full context.
        past = output.past_key_values

        # ---- Generation Loop using kv_cache ----
        for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True):
            logits_step = output.logits[:, -1, :]                
            # CFG combine  → (1, V)
            cond   = logits_step[0:1, :]
            uncond = logits_step[1:2, :]
            cfg = torch.as_tensor(cfg_weight, device=cond.device, dtype=cond.dtype)
            logits = cond + cfg * (cond - uncond)
            
            # Apply alignment stream analyzer integrity checks
            if self.patched_model.alignment_stream_analyzer is not None:
                if logits.dim() == 1:            # guard in case something upstream squeezed
                    logits = logits.unsqueeze(0) # (1, V)
                # Pass the last generated token for repetition tracking
                last_token = generated_ids[0, -1].item() if len(generated_ids[0]) > 0 else None
                logits = self.patched_model.alignment_stream_analyzer.step(logits, next_token=last_token)  # (1, V)

            # Apply repetition penalty
            ids_for_proc = generated_ids[:1, ...]   # batch = 1
            logits = repetition_penalty_processor(ids_for_proc, logits)  # expects (B,V)
            
            # Apply temperature scaling.
            if temperature != 1.0:
                logits = logits / temperature
                
            # Apply min_p and top_p filtering
            logits = min_p_warper(ids_for_proc, logits)
            logits = top_p_warper(ids_for_proc, logits)

            # Convert logits to probabilities and sample the next token.
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # shape: (B, 1)

            predicted.append(next_token)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Check for EOS token.
            if next_token.view(-1) == self.hp.stop_speech_token:
                logger.info(f"✅ EOS token detected! Stopping generation at step {i+1}")
                break

            # Get embedding for the new token.
            next_token_embed = self.speech_emb(next_token)
            next_token_embed = next_token_embed + self.speech_pos_emb.get_fixed_embedding(i + 1)

            #  For CFG
            next_token_embed = torch.cat([next_token_embed, next_token_embed])

            # Forward pass with only the new token and the cached past.
            output = self.patched_model(
                inputs_embeds=next_token_embed,
                past_key_values=past,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            # Update the kv_cache.
            past = output.past_key_values

        # Concatenate all predicted tokens along the sequence dimension.
        predicted_tokens = torch.cat(predicted, dim=1)  # shape: (B, num_tokens)
        return predicted_tokens

    @torch.inference_mode()
    def batch_inference(
        self,
        batch_text_tokens: List[Tensor],
        batch_t3_conds: List[T3Cond],
        max_new_tokens=1000,
        temperature=0.8,
        top_p=0.95,
        min_p=0.05,
        repetition_penalty=1.2,
        cfg_weight=0.5,
        stop_on_eos=True,
    ):
        """
        Perform parallel inference for multiple text sequences.

        Args:
            batch_text_tokens: List of text token tensors, each shape (1, seq_len) or (2, seq_len) for CFG
            batch_t3_conds: List of T3Cond objects for each sequence
            max_new_tokens: Maximum tokens to generate per sequence
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            min_p: Minimum probability threshold
            repetition_penalty: Repetition penalty factor
            cfg_weight: Classifier-free guidance weight
            stop_on_eos: Whether to stop on EOS token

        Returns:
            List of generated speech token tensors
        """
        batch_size = len(batch_text_tokens)
        device = self.device

        if batch_size == 0:
            return []

        # Validate inputs
        for text_tokens in batch_text_tokens:
            _ensure_BOT_EOT(text_tokens, self.hp)

        # Initialize batch generation state
        batch_state = BatchGenerationState(
            batch_size=batch_size,
            max_tokens=max_new_tokens,
            device=device,
            start_token=self.hp.start_speech_token,
            stop_token=self.hp.stop_speech_token,
            model_config={
                'num_hidden_layers': self.cfg.num_hidden_layers,
                'num_attention_heads': self.cfg.num_attention_heads,
                'hidden_size': self.cfg.hidden_size,
            }
        )

        # Prepare conditioning for all sequences
        batch_cond_embeds = []
        for i in range(batch_size):
            text_tokens = batch_text_tokens[i].to(device)
            t3_cond = batch_t3_conds[i]

            # Prepare initial speech tokens (start token)
            initial_speech_tokens = self.hp.start_speech_token * torch.ones((text_tokens.size(0), 1),
                                                                           device=device, dtype=torch.long)

            # Prepare input embeddings for this sequence
            embeds, len_cond = self.prepare_input_embeds(
                t3_cond=t3_cond,
                text_tokens=text_tokens,
                speech_tokens=initial_speech_tokens,
                cfg_weight=cfg_weight,
            )
            batch_cond_embeds.append(embeds)

        # Set up model for batch generation
        if not self.compiled:
            # Initialize alignment analyzers for multilingual if needed
            alignment_analyzers = []
            if self.hp.is_multilingual:
                for i, text_tokens in enumerate(batch_text_tokens):
                    len_cond = batch_cond_embeds[i].size(1) - text_tokens.size(1) - 1  # Approximate conditioning length
                    analyzer = AlignmentStreamAnalyzer(
                        self.tfmr,
                        None,
                        text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
                        alignment_layer_idx=9,
                        eos_idx=self.hp.stop_speech_token,
                    )
                    alignment_analyzers.append(analyzer)

            # Create patched model for first sequence (reuse for others)
            patched_model = T3HuggingfaceBackend(
                config=self.cfg,
                llama=self.tfmr,
                speech_enc=self.speech_emb,
                speech_head=self.speech_head,
                alignment_stream_analyzer=alignment_analyzers[0] if alignment_analyzers else None,
            )
            self.patched_model = patched_model
            self.compiled = True

        # Initialize logits processors
        top_p_warper = TopPLogitsWarper(top_p=top_p)
        min_p_warper = MinPLogitsWarper(min_p=min_p)
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))

        # Prepare initial BOS embeddings for all sequences
        bos_token = torch.tensor([[self.hp.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = self.speech_emb(bos_token)
        bos_embed = bos_embed + self.speech_pos_emb.get_fixed_embedding(0)

        # Initial forward pass for all sequences
        all_past_key_values = []
        for i in range(batch_size):
            # For CFG, duplicate the BOS embedding
            if cfg_weight > 0.0:
                seq_bos_embed = torch.cat([bos_embed, bos_embed])
            else:
                seq_bos_embed = bos_embed

            # Combine conditioning and BOS for this sequence
            inputs_embeds = torch.cat([batch_cond_embeds[i], seq_bos_embed], dim=1)

            # Forward pass
            output = self.patched_model(
                inputs_embeds=inputs_embeds,
                past_key_values=None,
                use_cache=True,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            all_past_key_values.append(output.past_key_values)

        # Generation loop
        for step in tqdm(range(max_new_tokens), desc="Batch Sampling", dynamic_ncols=True):
            if not batch_state.has_active_sequences():
                break

            active_sequences = batch_state.active_sequences
            batch_logits_list = []

            # Forward pass for all active sequences
            for seq_idx, seq_id in enumerate(active_sequences):
                past_kv = all_past_key_values[seq_id]

                # Get current token for this sequence
                current_tokens = batch_state.sequences[seq_id].generated_tokens[:, -1:]

                # Get embedding for current token
                token_embed = self.speech_emb(current_tokens)
                if self.hp.input_pos_emb == "learned":
                    token_embed = token_embed + self.speech_pos_emb.get_fixed_embedding(step + 1)

                # For CFG, duplicate the token embedding
                if cfg_weight > 0.0:
                    token_embed = torch.cat([token_embed, token_embed])

                # Forward pass
                output = self.patched_model(
                    inputs_embeds=token_embed,
                    past_key_values=past_kv,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                )

                # Update past key values
                all_past_key_values[seq_id] = output.past_key_values

                # Get logits and apply CFG
                logits_step = output.logits[:, -1, :]

                if cfg_weight > 0.0:
                    cond = logits_step[0:1, :]
                    uncond = logits_step[1:2, :]
                    cfg_tensor = torch.as_tensor(cfg_weight, device=cond.device, dtype=cond.dtype)
                    logits = cond + cfg_tensor * (cond - uncond)
                else:
                    logits = logits_step

                # Apply alignment stream analyzer if available
                if hasattr(self.patched_model, 'alignment_stream_analyzer') and self.patched_model.alignment_stream_analyzer is not None:
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)

                    # Get last generated token for repetition tracking
                    generated_tokens = batch_state.sequences[seq_id].generated_tokens
                    last_token = generated_tokens[0, -1].item() if generated_tokens.size(1) > 0 else None
                    logits = self.patched_model.alignment_stream_analyzer.step(logits, next_token=last_token)

                batch_logits_list.append(logits)

            # Process logits for all active sequences
            if batch_logits_list:
                # Stack logits from all active sequences
                batch_logits = torch.cat(batch_logits_list, dim=0)

                # Apply repetition penalty for all sequences
                current_tokens_batch = batch_state.get_all_generated_tokens()
                if current_tokens_batch.size(0) > 0:
                    batch_logits = repetition_penalty_processor(current_tokens_batch, batch_logits)

                # Apply temperature scaling
                if temperature != 1.0:
                    batch_logits = batch_logits / temperature

                # Apply min_p and top_p filtering
                batch_logits = min_p_warper(current_tokens_batch, batch_logits)
                batch_logits = top_p_warper(current_tokens_batch, batch_logits)

                # Sample next tokens
                probs = torch.softmax(batch_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)  # (active_batch_size, 1)

                # Update batch state
                batch_state.update_with_new_tokens(next_tokens)

                # Check for early stopping
                if stop_on_eos and all(
                    self.hp.stop_speech_token in batch_state.sequences[seq_id].generated_tokens[0]
                    for seq_id in range(batch_size)
                    if batch_state.sequences[seq_id].is_completed
                ):
                    logger.info(f"✅ All sequences completed at step {step+1}")
                    break

        # Extract results
        results = batch_state.get_results()

        # Remove start tokens and return only generated speech tokens
        final_results = []
        for result in results:
            if result.size(1) > 0:
                # Remove start token if present
                if result.size(1) > 0 and result[0, 0].item() == self.hp.start_speech_token:
                    result = result[:, 1:]
                final_results.append(result.squeeze(0))  # Remove batch dimension
            else:
                # Empty result
                final_results.append(torch.empty(0, device=device, dtype=torch.long))

        return final_results
