"""
Batch generation state management for T3 model.
Handles tracking of generation progress, completion flags, and KV-cache for multiple sequences.
"""

import torch
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class SequenceState:
    """State for a single sequence in batch generation."""
    sequence_id: int
    generated_tokens: torch.Tensor
    is_completed: bool = False
    completion_step: Optional[int] = None
    position: int = 0


class BatchKVCache:
    """
    Efficient KV-cache management for batch generation.
    Handles variable-length sequences and dynamic completion.
    """

    def __init__(self, batch_size: int, num_layers: int, num_heads: int,
                 head_dim: int, max_seq_len: int, device: torch.device, dtype: torch.dtype = torch.float16):
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        # Initialize cache storage - tuple of (key, value) for each layer
        self.cache = {}
        self.current_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)

    def initialize_cache(self):
        """Initialize empty KV cache for all layers."""
        for layer_idx in range(self.num_layers):
            # Each cache entry is (keys, values) where:
            # keys/values shape: (batch_size, num_heads, max_seq_len, head_dim)
            keys = torch.zeros(
                self.batch_size, self.num_heads, self.max_seq_len, self.head_dim,
                device=self.device, dtype=self.dtype
            )
            values = torch.zeros(
                self.batch_size, self.num_heads, self.max_seq_len, self.head_dim,
                device=self.device, dtype=self.dtype
            )
            self.cache[layer_idx] = (keys, values)

    def get_cache_for_layer(self, layer_idx: int, active_batch_indices: Optional[List[int]] = None):
        """
        Retrieve KV cache for specific layer and batch elements.

        Args:
            layer_idx: Transformer layer index
            active_batch_indices: Indices of sequences still generating (None = all)

        Returns:
            Tuple of (keys, values) tensors
        """
        if layer_idx not in self.cache:
            return None

        keys, values = self.cache[layer_idx]

        if active_batch_indices is None:
            return (keys, values)

        # Return only active sequences
        active_keys = keys[active_batch_indices]
        active_values = values[active_batch_indices]
        return (active_keys, active_values)

    def update_cache(self, layer_idx: int, new_keys: torch.Tensor, new_values: torch.Tensor,
                     active_batch_indices: Optional[List[int]] = None):
        """
        Update KV cache with new key-value pairs.

        Args:
            layer_idx: Transformer layer index
            new_keys: New keys tensor (batch_size, num_heads, seq_len, head_dim)
            new_values: New values tensor (batch_size, num_heads, seq_len, head_dim)
            active_batch_indices: Indices of sequences being updated
        """
        if layer_idx not in self.cache:
            self.initialize_cache()

        keys, values = self.cache[layer_idx]

        if active_batch_indices is None:
            active_batch_indices = list(range(self.batch_size))

        # Update cache for active sequences
        for i, batch_idx in enumerate(active_batch_indices):
            current_len = self.current_lengths[batch_idx].item()
            seq_len = new_keys.size(2)

            # Update cache slices
            keys[batch_idx, :, current_len:current_len + seq_len] = new_keys[i]
            values[batch_idx, :, current_len:current_len + seq_len] = new_values[i]

    def increment_lengths(self, active_batch_indices: Optional[List[int]] = None, increment: int = 1):
        """Increment sequence lengths for active sequences."""
        if active_batch_indices is None:
            active_batch_indices = list(range(self.batch_size))

        for batch_idx in active_batch_indices:
            self.current_lengths[batch_idx] += increment

    def get_past_key_values(self, active_batch_indices: Optional[List[int]] = None):
        """
        Get past key values in HuggingFace format.

        Returns:
            List of tuples (key, value) for each layer
        """
        past_key_values = []

        for layer_idx in range(self.num_layers):
            keys, values = self.get_cache_for_layer(layer_idx, active_batch_indices)
            if keys is not None:
                # Trim to actual sequence lengths
                if active_batch_indices is None:
                    max_len = self.current_lengths.max().item()
                    keys = keys[:, :, :max_len]
                    values = values[:, :, :max_len]
                else:
                    max_len = self.current_lengths[active_batch_indices].max().item()
                    keys = keys[:, :, :max_len]
                    values = values[:, :, :max_len]

                past_key_values.append((keys, values))
            else:
                past_key_values.append(None)

        return past_key_values


class BatchGenerationState:
    """
    Manages generation state for multiple sequences in parallel.
    Tracks completion, progress, and coordinates KV-cache management.
    """

    def __init__(self, batch_size: int, max_tokens: int, device: torch.device,
                 start_token: int, stop_token: int, model_config: Dict[str, Any]):
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.device = device
        self.start_token = start_token
        self.stop_token = stop_token

        # Initialize sequence states (don't include start token in generated_tokens yet)
        self.sequences = [
            SequenceState(
                sequence_id=i,
                generated_tokens=torch.empty(1, 0, device=device, dtype=torch.long),  # Start empty
                position=0
            ) for i in range(batch_size)
        ]

        # Initialize KV cache
        self.kv_cache = BatchKVCache(
            batch_size=batch_size,
            num_layers=model_config.get('num_hidden_layers', 30),
            num_heads=model_config.get('num_attention_heads', 16),
            head_dim=model_config.get('hidden_size', 2048) // model_config.get('num_attention_heads', 16),
            max_seq_len=max_tokens + 100,  # Extra buffer for conditioning
            device=device
        )
        self.kv_cache.initialize_cache()

        # Generation tracking
        self.current_step = 0
        self.active_sequences = list(range(batch_size))
        self.completed_sequences = []

    def get_current_tokens(self) -> torch.Tensor:
        """Get current tokens for all active sequences."""
        active_tokens = []
        for seq_id in self.active_sequences:
            seq = self.sequences[seq_id]
            if seq.generated_tokens.size(1) > 0:
                active_tokens.append(seq.generated_tokens[:, -1:])  # Last token
            else:
                # First iteration - use start token
                active_tokens.append(torch.tensor([[self.start_token]], device=self.device, dtype=torch.long))

        if not active_tokens:
            return torch.empty(0, 1, device=self.device, dtype=torch.long)

        return torch.cat(active_tokens, dim=0)

    def get_all_generated_tokens(self) -> torch.Tensor:
        """Get all generated tokens for active sequences."""
        active_tokens = []
        max_len = max(seq.generated_tokens.size(1) for seq in self.sequences if not seq.is_completed)

        for seq_id in self.active_sequences:
            seq = self.sequences[seq_id]
            tokens = seq.generated_tokens

            # Pad to max length if needed
            if tokens.size(1) < max_len:
                pad_size = max_len - tokens.size(1)
                tokens = torch.cat([
                    tokens,
                    torch.full((1, pad_size), self.stop_token, device=self.device, dtype=torch.long)
                ], dim=1)

            active_tokens.append(tokens)

        if not active_tokens:
            return torch.empty(0, 0, device=self.device, dtype=torch.long)

        return torch.cat(active_tokens, dim=0)

    def update_with_new_tokens(self, new_tokens: torch.Tensor):
        """
        Update state with newly generated tokens.

        Args:
            new_tokens: Tensor of shape (active_batch_size, 1) with new tokens
        """
        if new_tokens.size(0) != len(self.active_sequences):
            raise ValueError(f"Expected {len(self.active_sequences)} tokens, got {new_tokens.size(0)}")

        # Update each active sequence
        for i, seq_id in enumerate(self.active_sequences):
            seq = self.sequences[seq_id]
            new_token = new_tokens[i:i+1]

            # Append new token
            seq.generated_tokens = torch.cat([seq.generated_tokens, new_token], dim=1)
            seq.position += 1

            # Check for completion
            if (new_token.item() == self.stop_token or
                seq.generated_tokens.size(1) >= self.max_tokens):
                seq.is_completed = True
                seq.completion_step = self.current_step

        # Update active sequences list
        newly_completed = [seq_id for seq_id in self.active_sequences
                          if self.sequences[seq_id].is_completed]

        for seq_id in newly_completed:
            self.active_sequences.remove(seq_id)
            self.completed_sequences.append(seq_id)

        # Update cache lengths for active sequences
        self.kv_cache.increment_lengths(self.active_sequences)
        self.current_step += 1

    def all_completed(self) -> bool:
        """Check if all sequences have completed generation."""
        return len(self.active_sequences) == 0

    def has_active_sequences(self) -> bool:
        """Check if there are still sequences generating."""
        return len(self.active_sequences) > 0

    def get_active_batch_size(self) -> int:
        """Get current number of active sequences."""
        return len(self.active_sequences)

    def get_results(self) -> List[torch.Tensor]:
        """
        Get final generated token sequences for all original sequences.

        Returns:
            List of tensors, one per original sequence
        """
        results = []
        for seq in self.sequences:
            # Return generated sequence as-is (no start token to skip)
            if seq.generated_tokens.size(1) > 0:
                result = seq.generated_tokens
            else:
                result = torch.empty(1, 0, device=self.device, dtype=torch.long)
            results.append(result)

        return results

    def get_generation_info(self) -> Dict[str, Any]:
        """Get generation statistics and info."""
        return {
            'total_sequences': self.batch_size,
            'completed_sequences': len(self.completed_sequences),
            'active_sequences': len(self.active_sequences),
            'current_step': self.current_step,
            'completion_steps': [self.sequences[i].completion_step for i in range(self.batch_size)],
            'sequence_lengths': [seq.generated_tokens.size(1) for seq in self.sequences]
        }