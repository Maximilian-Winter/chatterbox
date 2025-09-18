from typing import Optional

import torch
from torch import nn as nn
from transformers import LlamaConfig, LlamaModel, LlamaPreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class T3HuggingfaceBackend(LlamaPreTrainedModel, GenerationMixin):
    """
    Override some HuggingFace interface methods so we can use the standard `generate` method with our
    custom embedding / logit layers.

    NOTE: need to extend "*PreTrainedModel" to avoid re-initializing weights!
    """

    def __init__(
        self,
        config: LlamaConfig,
        llama: LlamaModel,
        *,
        speech_enc,
        speech_head,
        latents_queue=None,
        logits_queue=None,
        alignment_stream_analyzer: 'AlignmentStreamAnalyzer'=None,
    ):
        super().__init__(config)
        self.model = llama
        self.speech_enc = speech_enc
        self.speech_head = speech_head
        # Replace global flag with per-sequence state tracking for batch processing
        self._sequence_cond_states = {}  # sequence_id -> bool
        self._current_batch_size = 0
        self.alignment_stream_analyzer = alignment_stream_analyzer

    @torch.inference_mode()
    def prepare_inputs_for_generation(
        self, input_ids: torch.Tensor, decoder_cond: torch.Tensor, use_cache: bool, past_key_values=None,
        # This argument was introduced in some recent version of transformers (>=4.29.1)
        cache_position=None
    ):
        """
        This is a method used by huggingface's generate() method.
        Overridden here to apply our custom speech token embedding layer.

        :param input_ids: (B, S) int64 tensors of input tokens.
        :param decoder_cond: (B, T, C) float32 tensor of conditioning (prefixed to <input_embeds>)
        """

        # Make use of the kv cache: only the last input ID is new, we trim away all the ones before
        if not use_cache:
            past_key_values = None
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # custom speech token embedding layer
        inputs_embeds = self.speech_enc(input_ids)

        # Batch-aware conditioning management
        batch_size = inputs_embeds.size(0)
        if batch_size != self._current_batch_size:
            # Reset state tracking for new batch
            self._sequence_cond_states = {i: False for i in range(batch_size)}
            self._current_batch_size = batch_size

        # Check per-sequence conditioning state
        batch_needs_cond = []
        for seq_idx in range(batch_size):
            needs_cond = not self._sequence_cond_states.get(seq_idx, False)
            batch_needs_cond.append(needs_cond)

        # Apply conditioning per sequence as needed
        if any(batch_needs_cond):
            # For initial conditioning step, past_key_values can be None
            # This is expected for the first forward pass with conditioning
            if decoder_cond.size(0) != inputs_embeds.size(0):
                decoder_cond = decoder_cond.expand(inputs_embeds.size(0), -1, -1)

            # Concatenate conditioning for sequences that need it
            processed_embeds = []
            for seq_idx in range(batch_size):
                if batch_needs_cond[seq_idx]:
                    seq_embeds = torch.cat([decoder_cond[seq_idx:seq_idx+1], inputs_embeds[seq_idx:seq_idx+1]], dim=1)
                    self._sequence_cond_states[seq_idx] = True
                else:
                    seq_embeds = inputs_embeds[seq_idx:seq_idx+1]
                processed_embeds.append(seq_embeds)

            inputs_embeds = torch.cat(processed_embeds, dim=0)

        return {
            "inputs_embeds": inputs_embeds,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @torch.inference_mode()
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[torch.Tensor]=None,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
    ):
        """
        This is a method used by huggingface's generate() method.
        Overridden here to apply our custom layer norm and speech logit projection layers.

        :param inputs_embeds: (B, S, C) float32 tensor of conditioning inputs. If past key values are given,
        S should be 1.
        """
        is_large_input = inputs_embeds.size(1) != 1
        has_cache = past_key_values is not None and len(past_key_values) > 0
        assert not (is_large_input and has_cache)
        assert return_dict
        assert output_hidden_states

        tfmr_out = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden_states = tfmr_out.hidden_states[-1]  # (B, seq, dim)

        logits = self.speech_head(hidden_states)
        # assert inputs_embeds.size(0) == 1 # (disabled for CFG)

        # NOTE: hallucination handler may modify logits to force emit an EOS token
        # logits = self.alignment_stream_analyzer.step(logits)

        return CausalLMOutputWithCrossAttentions(
            logits=logits,
            past_key_values=tfmr_out.past_key_values,
            hidden_states=tfmr_out.hidden_states,
            attentions=tfmr_out.attentions,
        )

    def reset_batch_state(self):
        """Reset per-sequence conditioning states for new batch processing."""
        self._sequence_cond_states = {}
        self._current_batch_size = 0

    def set_batch_size(self, batch_size: int):
        """Initialize state tracking for a specific batch size."""
        self._sequence_cond_states = {i: False for i in range(batch_size)}
        self._current_batch_size = batch_size
