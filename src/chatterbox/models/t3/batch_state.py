"""
Batch generation state management for T3 model.
Handles tracking of generation progress, completion flags, and KV-cache for multiple sequences.
"""

import torch
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from transformers.cache_utils import DynamicCache


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
    Optimized KV-cache management for true parallel batch generation.
    Handles variable-length sequences, dynamic completion, and unified batched operations.
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

        # Unified cache storage for parallel operations
        self.keys_cache = {}  # layer_idx -> (batch_size, num_heads, max_seq_len, head_dim)
        self.values_cache = {}  # layer_idx -> (batch_size, num_heads, max_seq_len, head_dim)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.attention_masks = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=device)

        # Track active sequences for dynamic batch management
        self.active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        self.active_indices = torch.arange(batch_size, device=device)

        # Performance optimization flags
        self.is_initialized = False

    def initialize_cache(self):
        """Initialize unified KV cache for all layers with optimal memory layout."""
        if self.is_initialized:
            return

        for layer_idx in range(self.num_layers):
            # Unified cache tensors for parallel operations
            self.keys_cache[layer_idx] = torch.zeros(
                self.batch_size, self.num_heads, self.max_seq_len, self.head_dim,
                device=self.device, dtype=self.dtype, requires_grad=False
            )
            self.values_cache[layer_idx] = torch.zeros(
                self.batch_size, self.num_heads, self.max_seq_len, self.head_dim,
                device=self.device, dtype=self.dtype, requires_grad=False
            )

        self.is_initialized = True

    def get_unified_past_key_values(self) -> Optional[DynamicCache]:
        """
        Get past key values in HuggingFace DynamicCache format for unified batch processing.
        Only returns data for currently active sequences.
        Ensures consistent batch size handling for CFG and other scenarios.

        Returns:
            DynamicCache object containing keys and values for each layer, filtered to active sequences
        """
        if not self.is_initialized:
            return None

        active_batch_size = self.active_mask.sum().item()
        if active_batch_size == 0:
            return None

        # Get currently active indices
        active_indices = self.active_indices[self.active_mask]

        # Check if we actually have any cached data
        has_cached_data = False
        for layer_idx in range(self.num_layers):
            if layer_idx in self.keys_cache:
                max_len = self.current_lengths[active_indices].max().item()
                if max_len > 0:
                    has_cached_data = True
                    break

        if not has_cached_data:
            return None

        # Create DynamicCache instance
        cache = DynamicCache()

        # Debug logging for cache extraction
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Creating DynamicCache for {active_batch_size} active sequences")

        for layer_idx in range(self.num_layers):
            if layer_idx in self.keys_cache:
                # Extract active sequences and trim to actual lengths
                keys = self.keys_cache[layer_idx][active_indices]  # (active_batch, heads, seq, dim)
                values = self.values_cache[layer_idx][active_indices]

                # Trim to maximum current length among active sequences
                max_len = self.current_lengths[active_indices].max().item()
                if max_len > 0:
                    keys = keys[:, :, :max_len, :]
                    values = values[:, :, :max_len, :]

                    # Verify tensor shapes before adding to cache
                    expected_shape = (active_batch_size, self.num_heads, max_len, self.head_dim)
                    if keys.shape != expected_shape:
                        logger.warning(f"Layer {layer_idx}: keys shape mismatch. "
                                     f"Expected {expected_shape}, got {keys.shape}")

                    # Update cache with properly shaped tensors
                    cache.update(keys, values, layer_idx)
                    logger.debug(f"Layer {layer_idx}: Added keys/values with shape {keys.shape}")
                else:
                    # For empty sequences, add empty tensors with correct shape
                    empty_keys = torch.zeros(active_batch_size, self.num_heads, 0, self.head_dim,
                                           device=self.device, dtype=self.dtype)
                    empty_values = torch.zeros(active_batch_size, self.num_heads, 0, self.head_dim,
                                             device=self.device, dtype=self.dtype)
                    cache.update(empty_keys, empty_values, layer_idx)
                    logger.debug(f"Layer {layer_idx}: Added empty tensors for inactive sequences")
            else:
                # Add empty tensors for missing layers
                empty_keys = torch.zeros(active_batch_size, self.num_heads, 0, self.head_dim,
                                       device=self.device, dtype=self.dtype)
                empty_values = torch.zeros(active_batch_size, self.num_heads, 0, self.head_dim,
                                         device=self.device, dtype=self.dtype)
                cache.update(empty_keys, empty_values, layer_idx)
                logger.debug(f"Layer {layer_idx}: Added empty tensors for missing layer")

        return cache

    def update_unified_cache(self, past_key_values):
        """
        Update cache with new key-value pairs from transformer output.
        Supports both DynamicCache objects and legacy tuple format.

        Args:
            past_key_values: Either DynamicCache object or List of (keys, values) tuples from transformer output
        """
        if not self.is_initialized:
            self.initialize_cache()

        active_indices = self.active_indices[self.active_mask]
        active_batch_size = active_indices.size(0)

        if active_batch_size == 0:
            return

        # Handle DynamicCache format
        if isinstance(past_key_values, DynamicCache):
            # Extract key-value pairs from DynamicCache
            for layer_idx in range(len(past_key_values.key_cache)):
                new_keys = past_key_values.key_cache[layer_idx]
                new_values = past_key_values.value_cache[layer_idx]
                self._update_layer_cache(layer_idx, new_keys, new_values, active_indices, active_batch_size)

        # Handle legacy tuple format
        elif isinstance(past_key_values, (list, tuple)):
            for layer_idx, (new_keys, new_values) in enumerate(past_key_values):
                if new_keys is not None and new_values is not None:
                    self._update_layer_cache(layer_idx, new_keys, new_values, active_indices, active_batch_size)

    def _update_layer_cache(self, layer_idx: int, new_keys: torch.Tensor, new_values: torch.Tensor,
                          active_indices: torch.Tensor, active_batch_size: int):
        """
        Update cache for a specific layer with proper batch size handling.
        Handles CFG batch size doubling and other batch size mismatches.
        """
        if new_keys is None or new_values is None:
            return

        # Debug logging for tensor shape analysis
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Layer {layer_idx}: new_keys.shape={new_keys.shape}, "
                    f"expected active_batch_size={active_batch_size}")

        # Handle batch size mismatch (e.g., CFG scenarios)
        original_batch_size = new_keys.size(0)
        if original_batch_size != active_batch_size:
            if original_batch_size == active_batch_size * 2:
                # CFG mode: conditional + unconditional sequences
                # Take only the conditional sequences (first half)
                logger.debug(f"CFG detected: reducing batch from {original_batch_size} to {active_batch_size}")
                new_keys = new_keys[:active_batch_size]  # Take first half (conditional)
                new_values = new_values[:active_batch_size]
            elif original_batch_size > active_batch_size:
                # General case: truncate to active batch size
                logger.debug(f"Truncating batch from {original_batch_size} to {active_batch_size}")
                new_keys = new_keys[:active_batch_size]
                new_values = new_values[:active_batch_size]
            else:
                # Handle case where we have fewer keys than expected
                logger.warning(f"Fewer keys than expected: {original_batch_size} < {active_batch_size}")
                # Pad active_indices to match available keys
                active_indices = active_indices[:original_batch_size]
                active_batch_size = original_batch_size

        current_seq_len = new_keys.size(2)
        final_batch_size = new_keys.size(0)

        # Ensure we don't exceed available indices
        available_indices = active_indices[:final_batch_size]

        # Store in unified cache
        for i, batch_idx in enumerate(available_indices):
            current_len = self.current_lengths[batch_idx].item()
            end_pos = current_len + current_seq_len

            if end_pos <= self.max_seq_len:
                self.keys_cache[layer_idx][batch_idx, :, current_len:end_pos, :] = new_keys[i]
                self.values_cache[layer_idx][batch_idx, :, current_len:end_pos, :] = new_values[i]
            else:
                logger.warning(f"Sequence {batch_idx} exceeds max_seq_len: {end_pos} > {self.max_seq_len}")

        # Update current lengths after storing new data
        for i, batch_idx in enumerate(available_indices):
            self.current_lengths[batch_idx] += current_seq_len

    def update_sequence_states(self, completed_indices: torch.Tensor):
        """
        Update active sequence tracking when sequences complete.

        Args:
            completed_indices: Tensor of global batch indices that completed this step
        """
        if completed_indices.numel() > 0:
            # Mark completed sequences as inactive
            self.active_mask[completed_indices] = False

    def increment_lengths(self, active_indices: Optional[torch.Tensor] = None, increment: int = 1):
        """Increment sequence lengths for active sequences."""
        if active_indices is None:
            active_indices = self.active_indices[self.active_mask]

        self.current_lengths[active_indices] += increment

    def get_active_batch_size(self) -> int:
        """Get current number of active sequences."""
        return self.active_mask.sum().item()

    def prepare_for_next_step(self):
        """Prepare cache state for the next generation step."""
        # Update attention masks for new tokens
        active_indices = self.active_indices[self.active_mask]
        for idx in active_indices:
            current_len = self.current_lengths[idx].item()
            if current_len < self.max_seq_len:
                self.attention_masks[idx, current_len] = True


class BatchGenerationState:
    """
    Optimized parallel batch generation state manager.
    Efficiently handles variable-length sequences, dynamic completion, and unified operations.
    """

    def __init__(self, batch_size: int, max_tokens: int, device: torch.device,
                 start_token: int, stop_token: int, model_config: Dict[str, Any]):
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.device = device
        self.start_token = start_token
        self.stop_token = stop_token

        # Unified tensor-based state management for efficiency
        self.generated_tokens = torch.full(
            (batch_size, max_tokens), self.stop_token,
            device=device, dtype=torch.long
        )
        self.sequence_lengths = torch.zeros(batch_size, device=device, dtype=torch.long)
        self.completion_flags = torch.zeros(batch_size, device=device, dtype=torch.bool)
        self.completion_steps = torch.full((batch_size,), -1, device=device, dtype=torch.long)

        # Active sequence tracking for dynamic batch processing
        self.active_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
        self.active_indices = torch.arange(batch_size, device=device)

        # Initialize optimized KV cache
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

        # Legacy compatibility
        self.sequences = [
            SequenceState(
                sequence_id=i,
                generated_tokens=torch.empty(1, 0, device=device, dtype=torch.long),
                position=0
            ) for i in range(batch_size)
        ]
        self.active_sequences = list(range(batch_size))
        self.completed_sequences = []

    def get_current_tokens(self) -> torch.Tensor:
        """Get current tokens for all active sequences using optimized tensor operations."""
        active_indices = self.active_indices[self.active_mask]

        if active_indices.numel() == 0:
            return torch.empty(0, 1, device=self.device, dtype=torch.long)

        # Get last generated token for each active sequence
        current_tokens = []
        for idx in active_indices:
            seq_len = self.sequence_lengths[idx].item()
            if seq_len > 0:
                current_tokens.append(self.generated_tokens[idx, seq_len-1:seq_len].unsqueeze(0))
            else:
                # First iteration - use start token
                current_tokens.append(torch.tensor([[self.start_token]], device=self.device, dtype=torch.long))

        return torch.cat(current_tokens, dim=0)

    def get_unified_active_tokens(self) -> torch.Tensor:
        """
        Get all generated tokens for active sequences in unified tensor format.
        Optimized for parallel processing.
        """
        active_indices = self.active_indices[self.active_mask]
        active_batch_size = active_indices.numel()

        if active_batch_size == 0:
            return torch.empty(0, 0, device=self.device, dtype=torch.long)

        # Get maximum sequence length among active sequences
        active_lengths = self.sequence_lengths[active_indices]
        max_len = active_lengths.max().item()

        if max_len == 0:
            return torch.empty(active_batch_size, 0, device=self.device, dtype=torch.long)

        # Extract active sequences and pad to max length
        active_tokens = torch.full(
            (active_batch_size, max_len), self.stop_token,
            device=self.device, dtype=torch.long
        )

        for i, idx in enumerate(active_indices):
            seq_len = active_lengths[i].item()
            if seq_len > 0:
                active_tokens[i, :seq_len] = self.generated_tokens[idx, :seq_len]

        return active_tokens

    def get_all_generated_tokens(self) -> torch.Tensor:
        """Get all generated tokens for active sequences - legacy compatibility method."""
        return self.get_unified_active_tokens()

    def update_with_new_tokens(self, new_tokens: torch.Tensor):
        """
        Optimized batch update with newly generated tokens using vectorized operations.

        Args:
            new_tokens: Tensor of shape (active_batch_size, 1) with new tokens
        """
        active_indices = self.active_indices[self.active_mask]
        active_batch_size = active_indices.numel()

        if new_tokens.size(0) != active_batch_size:
            raise ValueError(f"Expected {active_batch_size} tokens, got {new_tokens.size(0)}")

        if active_batch_size == 0:
            return

        # Vectorized token updates
        current_lengths = self.sequence_lengths[active_indices]
        new_token_values = new_tokens.squeeze(1)  # (active_batch_size,)

        # Store new tokens at current positions
        for i, idx in enumerate(active_indices):
            pos = current_lengths[i].item()
            if pos < self.max_tokens:
                self.generated_tokens[idx, pos] = new_token_values[i]

        # Update sequence lengths
        self.sequence_lengths[active_indices] += 1

        # Check for completion using vectorized operations
        stop_condition = (new_token_values == self.stop_token)
        length_condition = (self.sequence_lengths[active_indices] >= self.max_tokens)
        completion_mask = stop_condition | length_condition

        # Mark completed sequences
        completed_indices = active_indices[completion_mask]
        if completed_indices.numel() > 0:
            self.completion_flags[completed_indices] = True
            self.completion_steps[completed_indices] = self.current_step
            self.active_mask[completed_indices] = False

            # Update KV cache state
            self.kv_cache.update_sequence_states(completed_indices)

            # Update legacy compatibility structures
            for idx in completed_indices:
                idx_item = idx.item()
                if idx_item in self.active_sequences:
                    self.active_sequences.remove(idx_item)
                    self.completed_sequences.append(idx_item)
                    self.sequences[idx_item].is_completed = True
                    self.sequences[idx_item].completion_step = self.current_step

        # Update cache lengths for remaining active sequences
        remaining_active = self.active_indices[self.active_mask]
        self.kv_cache.increment_lengths(remaining_active)

        # Update legacy sequence states for compatibility
        for i, idx in enumerate(active_indices):
            seq = self.sequences[idx.item()]
            new_token = new_tokens[i:i+1]
            seq.generated_tokens = torch.cat([seq.generated_tokens, new_token], dim=1)
            seq.position += 1

        self.current_step += 1
        self.kv_cache.prepare_for_next_step()

    def all_completed(self) -> bool:
        """Check if all sequences have completed generation using optimized tensor operations."""
        return not self.active_mask.any().item()

    def has_active_sequences(self) -> bool:
        """Check if there are still sequences generating using optimized tensor operations."""
        return self.active_mask.any().item()

    def get_active_batch_size(self) -> int:
        """Get current number of active sequences using optimized tensor operations."""
        return self.active_mask.sum().item()

    def get_completion_status(self) -> Dict[str, Any]:
        """
        Get detailed completion status for debugging EOS detection.

        Returns:
            Dictionary with completion status details for each sequence
        """
        completion_status = {
            'total_sequences': self.batch_size,
            'completed_count': self.completion_flags.sum().item(),
            'active_count': self.active_mask.sum().item(),
            'completion_flags': self.completion_flags.tolist(),
            'sequence_lengths': self.sequence_lengths.tolist(),
            'completion_steps': self.completion_steps.tolist(),
            'current_step': self.current_step,
            'stop_token': self.stop_token,
        }

        # Add per-sequence details
        sequence_details = []
        for i in range(self.batch_size):
            seq_detail = {
                'seq_id': i,
                'is_completed': self.completion_flags[i].item(),
                'is_active': self.active_mask[i].item(),
                'length': self.sequence_lengths[i].item(),
                'completion_step': self.completion_steps[i].item() if self.completion_steps[i].item() >= 0 else None,
                'last_token': self.generated_tokens[i, self.sequence_lengths[i].item()-1].item() if self.sequence_lengths[i].item() > 0 else None,
            }
            sequence_details.append(seq_detail)

        completion_status['sequence_details'] = sequence_details
        return completion_status

    def validate_eos_detection(self) -> bool:
        """
        Validate that EOS detection is working correctly.
        Check for sequences that should have stopped but didn't.

        Returns:
            True if EOS detection is working correctly, False otherwise
        """
        issues_found = []

        for i in range(self.batch_size):
            seq_len = self.sequence_lengths[i].item()
            if seq_len > 0:
                # Check if sequence contains stop token but is still active
                sequence_tokens = self.generated_tokens[i, :seq_len]
                has_stop_token = (sequence_tokens == self.stop_token).any().item()
                is_active = self.active_mask[i].item()

                if has_stop_token and is_active:
                    issues_found.append(f"Sequence {i} contains stop token but is still active")

                # Check if sequence is at max length but still active
                if seq_len >= self.max_tokens and is_active:
                    issues_found.append(f"Sequence {i} at max length {seq_len} but still active")

        if issues_found:
            import logging
            logger = logging.getLogger(__name__)
            for issue in issues_found:
                logger.warning(f"⚠️ EOS Detection Issue: {issue}")
            return False

        return True

    def get_results(self) -> List[torch.Tensor]:
        """
        Get final generated token sequences for all original sequences.
        Uses optimized tensor operations for better performance.

        Returns:
            List of tensors, one per original sequence
        """
        results = []
        for i in range(self.batch_size):
            seq_len = self.sequence_lengths[i].item()
            if seq_len > 0:
                # Extract actual sequence without padding
                result = self.generated_tokens[i, :seq_len].unsqueeze(0)
            else:
                result = torch.empty(1, 0, device=self.device, dtype=torch.long)
            results.append(result)

        return results

    def get_generation_info(self) -> Dict[str, Any]:
        """Get generation statistics and info using optimized tensor operations."""
        return {
            'total_sequences': self.batch_size,
            'completed_sequences': self.completion_flags.sum().item(),
            'active_sequences': self.active_mask.sum().item(),
            'current_step': self.current_step,
            'completion_steps': self.completion_steps.tolist(),
            'sequence_lengths': self.sequence_lengths.tolist(),
            'average_length': self.sequence_lengths.float().mean().item(),
            'max_length': self.sequence_lengths.max().item(),
            'min_length': self.sequence_lengths.min().item(),
        }

    def get_attention_mask(self) -> torch.Tensor:
        """
        Generate attention mask for current batch state.
        Returns tensor of shape (active_batch_size, max_seq_len) for efficient attention computation.
        """
        active_indices = self.active_indices[self.active_mask]
        active_batch_size = active_indices.numel()

        if active_batch_size == 0:
            return torch.empty(0, 0, device=self.device, dtype=torch.bool)

        active_lengths = self.sequence_lengths[active_indices]
        max_len = active_lengths.max().item()

        if max_len == 0:
            return torch.empty(active_batch_size, 0, device=self.device, dtype=torch.bool)

        # Create attention mask: True for valid positions
        attention_mask = torch.arange(max_len, device=self.device).unsqueeze(0) < active_lengths.unsqueeze(1)
        return attention_mask

    def optimize_for_parallel_processing(self):
        """
        Optimize internal state for maximum parallel processing efficiency.
        Call this before starting generation loop.
        """
        # Pre-allocate commonly used tensors
        self._temp_active_indices = torch.empty(self.batch_size, device=self.device, dtype=torch.long)
        self._temp_active_lengths = torch.empty(self.batch_size, device=self.device, dtype=torch.long)

        # Ensure KV cache is properly initialized
        self.kv_cache.initialize_cache()

        # Pre-compute static masks for better memory access patterns
        self.kv_cache.prepare_for_next_step()