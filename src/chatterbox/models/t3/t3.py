# Copyright (c) 2025 Resemble AI
# MIT License
import logging
from typing import Union, Optional, List, Dict, Any
import warnings

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
from ..utils import AttrDict


def _ensure_BOT_EOT(text_tokens: Tensor, hp):
    B = text_tokens.size(0)
    assert (text_tokens == hp.start_text_token).int().sum() >= B, "missing start_text_token"
    assert (text_tokens == hp.stop_text_token).int().sum() >= B, "missing stop_text_token"


class T3(nn.Module):
    """
    Token-To-Token (T3) TTS model using huggingface transformer models as backbones
    """

    def __init__(self, hp=None):
        if hp is None:
            hp = T3Config.english_only()
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
        """Token cond data needs to be embedded"""
        if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
            t3_cond.cond_prompt_speech_emb = self.speech_emb(t3_cond.cond_prompt_speech_tokens) + \
                                             self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
        return self.cond_enc(t3_cond)

    def prepare_input_embeds(
            self,
            *,
            t3_cond: T3Cond,
            text_tokens: torch.LongTensor,
            speech_tokens: torch.LongTensor,
            cfg_weight: float = 0.0,
    ):
        cond_emb = self.prepare_conditioning(t3_cond)
        text_emb = self.text_emb(text_tokens)
        if cfg_weight > 0.0 and text_emb.size(0) > 1:
            text_emb[1].zero_()  # CFG uncond

        speech_emb = self.speech_emb(speech_tokens)
        if self.hp.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)
            speech_emb = speech_emb + self.speech_pos_emb(speech_tokens)
        len_cond = cond_emb.size(1)

        if cond_emb.size(0) != text_emb.size(0):
            batch_size_needed = text_emb.size(0)
            if cond_emb.size(0) == 1:
                cond_emb = cond_emb.expand(batch_size_needed, -1, -1)
            else:
                cond_emb = cond_emb.repeat(batch_size_needed // cond_emb.size(0) + 1, 1, 1)[:batch_size_needed]

        embeds = torch.stack([
            torch.cat((ce, te, se))
            for ce, te, se in zip(cond_emb, text_emb, speech_emb)
        ])
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

        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
        )

        tfmr_out = self.tfmr.forward(
            input_ids=None,
            inputs_embeds=embeds,
            output_hidden_states=True,
            return_dict=True,
            use_cache=(not training),
        )
        hidden_states = tfmr_out.hidden_states[-1]

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
        )

        IGNORE_ID = -100
        device = out.text_logits.device
        mask_text = torch.arange(len_text, device=device)[None] >= text_token_lens[:, None]
        mask_speech = torch.arange(len_speech, device=device)[None] >= speech_token_lens[:, None]
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
            initial_speech_tokens: Optional[Tensor] = None,
            prepend_prompt_speech_tokens: Optional[Tensor] = None,
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
        """Single sequence inference (original implementation)"""
        assert prepend_prompt_speech_tokens is None, "not implemented"
        _ensure_BOT_EOT(text_tokens, self.hp)
        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)

        if initial_speech_tokens is None:
            initial_speech_tokens = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])

        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech_tokens,
            cfg_weight=cfg_weight,
        )

        self.compiled = False
        if not self.compiled:
            alignment_stream_analyzer = None
            if self.hp.is_multilingual:
                alignment_stream_analyzer = AlignmentStreamAnalyzer(
                    self.tfmr,
                    None,
                    text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
                    alignment_layer_idx=9,
                    eos_idx=self.hp.stop_speech_token,
                )

            patched_model = T3HuggingfaceBackend(
                config=self.cfg,
                llama=self.tfmr,
                speech_enc=self.speech_emb,
                speech_head=self.speech_head,
                alignment_stream_analyzer=alignment_stream_analyzer,
            )
            self.patched_model = patched_model
            self.compiled = True

        device = embeds.device
        bos_token = torch.tensor([[self.hp.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = self.speech_emb(bos_token)
        bos_embed = bos_embed + self.speech_pos_emb.get_fixed_embedding(0)
        bos_embed = torch.cat([bos_embed, bos_embed])  # For CFG

        inputs_embeds = torch.cat([embeds, bos_embed], dim=1)
        generated_ids = bos_token.clone()
        predicted = []

        top_p_warper = TopPLogitsWarper(top_p=top_p)
        min_p_warper = MinPLogitsWarper(min_p=min_p)
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))

        output = self.patched_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = output.past_key_values

        for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True):
            logits_step = output.logits[:, -1, :]
            cond = logits_step[0:1, :]
            uncond = logits_step[1:2, :]
            cfg = torch.as_tensor(cfg_weight, device=cond.device, dtype=cond.dtype)
            logits = cond + cfg * (cond - uncond)

            if self.patched_model.alignment_stream_analyzer is not None:
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                last_token = generated_ids[0, -1].item() if len(generated_ids[0]) > 0 else None
                logits = self.patched_model.alignment_stream_analyzer.step(logits, next_token=last_token)

            ids_for_proc = generated_ids[:1, ...]
            logits = repetition_penalty_processor(ids_for_proc, logits)

            if temperature != 1.0:
                logits = logits / temperature

            logits = min_p_warper(ids_for_proc, logits)
            logits = top_p_warper(ids_for_proc, logits)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            predicted.append(next_token)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.view(-1) == self.hp.stop_speech_token:
                logger.info(f"✅ EOS token detected! Stopping generation at step {i + 1}")
                break

            next_token_embed = self.speech_emb(next_token)
            next_token_embed = next_token_embed + self.speech_pos_emb.get_fixed_embedding(i + 1)
            next_token_embed = torch.cat([next_token_embed, next_token_embed])

            output = self.patched_model(
                inputs_embeds=next_token_embed,
                past_key_values=past,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = output.past_key_values

        predicted_tokens = torch.cat(predicted, dim=1)
        return predicted_tokens

    @torch.inference_mode()
    def inference_batch(
            self,
            t3_cond_batch: List[T3Cond],
            text_tokens_batch: List[Tensor],
            initial_speech_tokens_batch: Optional[List[Tensor]] = None,
            max_new_tokens: Optional[int] = None,
            temperatures: Union[float, List[float]] = 0.8,
            top_ps: Union[float, List[float]] = 0.95,
            min_ps: Union[float, List[float]] = 0.05,
            repetition_penalties: Union[float, List[float]] = 1.2,
            cfg_weights: Union[float, List[float]] = 0.5,
            max_batch_size: int = 4,
    ) -> List[Tensor]:
        """Truly batched inference with proper vectorization"""

        batch_size = len(t3_cond_batch)

        # Ensure parameters are lists
        def ensure_list(param, name):
            if isinstance(param, (int, float)):
                return [float(param)] * batch_size
            elif len(param) != batch_size:
                raise ValueError(f"{name} must be scalar or list with same length as batch")
            return param

        temperatures = ensure_list(temperatures, "temperatures")
        top_ps = ensure_list(top_ps, "top_ps")
        min_ps = ensure_list(min_ps, "min_ps")
        repetition_penalties = ensure_list(repetition_penalties, "repetition_penalties")
        cfg_weights = ensure_list(cfg_weights, "cfg_weights")

        # Process in sub-batches
        all_results = []
        for i in range(0, batch_size, max_batch_size):
            end_idx = min(i + max_batch_size, batch_size)
            sub_results = self._inference_batch_vectorized(
                t3_cond_batch[i:end_idx],
                text_tokens_batch[i:end_idx],
                initial_speech_tokens_batch[i:end_idx] if initial_speech_tokens_batch else None,
                max_new_tokens,
                temperatures[i:end_idx],
                top_ps[i:end_idx],
                min_ps[i:end_idx],
                repetition_penalties[i:end_idx],
                cfg_weights[i:end_idx],
            )
            all_results.extend(sub_results)

        return all_results

    def _inference_batch_vectorized(
            self,
            t3_cond_batch: List[T3Cond],
            text_tokens_batch: List[Tensor],
            initial_speech_tokens_batch: Optional[List[Tensor]],
            max_new_tokens: Optional[int],
            temperatures: List[float],
            top_ps: List[float],
            min_ps: List[float],
            repetition_penalties: List[float],
            cfg_weights: List[float],
    ) -> List[Tensor]:
        """Vectorized batch inference implementation"""

        batch_size = len(t3_cond_batch)
        device = self.device

        # 1. Prepare inputs (Padding and Batching)
        padded_text_tokens = torch.cat(text_tokens_batch, dim=0).to(dtype=torch.long, device=device)

        batched_t3_cond = T3Cond(
            speaker_emb=torch.cat([c.speaker_emb for c in t3_cond_batch], dim=0),
            cond_prompt_speech_tokens=torch.cat([c.cond_prompt_speech_tokens for c in t3_cond_batch if c.cond_prompt_speech_tokens is not None], dim=0) if any(c.cond_prompt_speech_tokens is not None for c in t3_cond_batch) else None,
            emotion_adv=torch.cat([c.emotion_adv for c in t3_cond_batch], dim=0)
        ).to(device)

        # Prepare for CFG
        padded_text_tokens_cfg = padded_text_tokens.repeat(2, 1)
        batched_t3_cond_cfg = T3Cond(
            speaker_emb=batched_t3_cond.speaker_emb.repeat(2, 1, 1),
            cond_prompt_speech_tokens=batched_t3_cond.cond_prompt_speech_tokens.repeat(2, 1) if batched_t3_cond.cond_prompt_speech_tokens is not None else None,
            emotion_adv=batched_t3_cond.emotion_adv.repeat(2, 1, 1)
        ).to(device)

        if initial_speech_tokens_batch is None:
            initial_speech_tokens = torch.full((batch_size, 1), self.hp.start_speech_token, device=device, dtype=torch.long)
        else:
            initial_speech_tokens = torch.stack(initial_speech_tokens_batch).to(device)
        initial_speech_tokens_cfg = initial_speech_tokens.repeat(2, 1)

        # 2. Get input embeddings
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=batched_t3_cond_cfg,
            text_tokens=padded_text_tokens_cfg,
            speech_tokens=initial_speech_tokens_cfg,
            cfg_weight=1.0  # We do CFG manually later
        )

        # 3. Initial forward pass
        output = self.tfmr(
            inputs_embeds=embeds,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past_key_values = output.past_key_values

        # 4. Generation loop (vectorized)
        generated_tokens = torch.full((batch_size, 0), -1, dtype=torch.long, device=device)
        active_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)
        max_tokens = max_new_tokens or self.hp.max_speech_tokens

        temperatures_t = torch.tensor(temperatures, device=device, dtype=torch.float32).unsqueeze(1)
        cfg_weights_t = torch.tensor(cfg_weights, device=device, dtype=torch.float32).unsqueeze(1)
        repetition_penalties_t = torch.tensor(repetition_penalties, device=device, dtype=torch.float32)

        next_tokens = initial_speech_tokens

        for step in range(max_tokens):
            next_tokens_embeds = self.speech_emb(next_tokens.repeat(2, 1))
            if self.hp.input_pos_emb == "learned":
                next_tokens_embeds += self.speech_pos_emb.get_fixed_embedding(step)

            output = self.tfmr(
                inputs_embeds=next_tokens_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past_key_values = output.past_key_values
            logits = self.speech_head(output.hidden_states[-1][:, -1, :])

            cond_logits, uncond_logits = logits.chunk(2, dim=0)
            combined_logits = cond_logits + cfg_weights_t * (cond_logits - uncond_logits)

            if torch.any(repetition_penalties_t != 1.0):
                if generated_tokens.shape[1] > 0:
                    combined_logits.scatter_add_(1, generated_tokens, -torch.log(repetition_penalties_t).unsqueeze(1) * (combined_logits > 0))

            combined_logits = combined_logits / temperatures_t

            top_p_warper = TopPLogitsWarper(top_p=top_ps[0])
            min_p_warper = MinPLogitsWarper(min_p=min_ps[0])
            combined_logits = min_p_warper(generated_tokens, combined_logits)
            combined_logits = top_p_warper(generated_tokens, combined_logits)

            probs = torch.softmax(combined_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            generated_tokens = torch.cat([generated_tokens, next_tokens], dim=1)

            active_sequences = active_sequences & (next_tokens.squeeze(-1) != self.hp.stop_speech_token)
            if not active_sequences.any():
                break

        results = [row[row != self.hp.stop_speech_token] for row in generated_tokens]
        return results