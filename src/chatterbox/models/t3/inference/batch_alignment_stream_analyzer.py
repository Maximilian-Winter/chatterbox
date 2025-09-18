"""
Batch Alignment Stream Analyzer for Multilingual TTS Models

This module provides efficient batch processing for alignment analysis in multilingual TTS models,
maintaining quality control while enabling parallel generation across multiple sequences.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


@dataclass
class LanguageConfig:
    """Configuration for language-specific alignment analysis."""
    alignment_head: int
    quality_threshold: float
    repetition_threshold: float
    max_intervention_rate: float
    attention_buffer_size: int = 50


class MultilinguaAlignmentConfig:
    """Multilingual alignment configuration manager."""

    DEFAULT_CONFIGS = {
        'en': LanguageConfig(alignment_head=9, quality_threshold=0.7, repetition_threshold=0.8, max_intervention_rate=0.15),
        'es': LanguageConfig(alignment_head=8, quality_threshold=0.65, repetition_threshold=0.75, max_intervention_rate=0.18),
        'fr': LanguageConfig(alignment_head=8, quality_threshold=0.65, repetition_threshold=0.75, max_intervention_rate=0.18),
        'de': LanguageConfig(alignment_head=9, quality_threshold=0.68, repetition_threshold=0.77, max_intervention_rate=0.16),
        'it': LanguageConfig(alignment_head=8, quality_threshold=0.66, repetition_threshold=0.76, max_intervention_rate=0.17),
        'pt': LanguageConfig(alignment_head=8, quality_threshold=0.65, repetition_threshold=0.75, max_intervention_rate=0.18),
        'zh': LanguageConfig(alignment_head=10, quality_threshold=0.72, repetition_threshold=0.82, max_intervention_rate=0.12),
        'ja': LanguageConfig(alignment_head=10, quality_threshold=0.71, repetition_threshold=0.81, max_intervention_rate=0.13),
        'ko': LanguageConfig(alignment_head=10, quality_threshold=0.70, repetition_threshold=0.80, max_intervention_rate=0.14),
    }

    @classmethod
    def get_config(cls, language: str) -> LanguageConfig:
        """Get configuration for a specific language."""
        return cls.DEFAULT_CONFIGS.get(language, cls.DEFAULT_CONFIGS['en'])

    @classmethod
    def detect_language(cls, t3_cond) -> str:
        """Detect language from T3Cond object."""
        # Try to extract language from conditioning data
        if hasattr(t3_cond, 'language') and t3_cond.language:
            return t3_cond.language.lower()

        # Fallback to English
        return 'en'


class BatchSequenceState:
    """State tracking for a single sequence in batch alignment analysis."""

    def __init__(self, sequence_id: int, language: str, text_tokens_slice: Tuple[int, int],
                 eos_idx: int, config: LanguageConfig):
        self.sequence_id = sequence_id
        self.language = language
        self.text_tokens_slice = text_tokens_slice
        self.eos_idx = eos_idx
        self.config = config

        # Attention tracking
        self.attention_buffer = deque(maxlen=config.attention_buffer_size)
        self.last_attention_weights = None

        # Quality monitoring
        self.intervention_count = 0
        self.total_tokens = 0
        self.repetition_history = deque(maxlen=10)
        self.quality_scores = deque(maxlen=20)

        # State flags
        self.is_completed = False
        self.needs_intervention = False
        self.last_token = None

    def update_attention(self, attention_weights: torch.Tensor):
        """Update attention weights for this sequence."""
        self.attention_buffer.append(attention_weights.detach().cpu())
        self.last_attention_weights = attention_weights

    def analyze_quality(self) -> float:
        """Analyze current alignment quality."""
        if self.last_attention_weights is None:
            return 1.0

        # Calculate attention alignment score
        attention = self.last_attention_weights
        text_start, text_end = self.text_tokens_slice

        # Focus on text region attention
        text_attention = attention[:, text_start:text_end]

        # Calculate alignment quality metrics
        max_attention = text_attention.max(dim=-1)[0]
        attention_entropy = -torch.sum(text_attention * torch.log(text_attention + 1e-8), dim=-1)

        # Normalize entropy by sequence length
        seq_len = text_end - text_start
        normalized_entropy = attention_entropy / torch.log(torch.tensor(seq_len, dtype=torch.float))

        # Quality score (higher is better)
        quality_score = max_attention.mean().item() * (1 - normalized_entropy.mean().item())

        self.quality_scores.append(quality_score)
        return quality_score

    def check_repetition(self, next_token: int) -> bool:
        """Check for repetition patterns."""
        if next_token == self.last_token:
            self.repetition_history.append(1)
        else:
            self.repetition_history.append(0)

        self.last_token = next_token

        # Calculate repetition rate
        if len(self.repetition_history) >= 5:
            recent_repetition_rate = sum(list(self.repetition_history)[-5:]) / 5
            return recent_repetition_rate > self.config.repetition_threshold

        return False

    def should_intervene(self, next_token: int) -> bool:
        """Determine if intervention is needed for this sequence."""
        if self.is_completed:
            return False

        # Check intervention rate limit
        intervention_rate = self.intervention_count / max(1, self.total_tokens)
        if intervention_rate > self.config.max_intervention_rate:
            return False

        # Analyze current quality
        quality_score = self.analyze_quality()

        # Check for repetition
        has_repetition = self.check_repetition(next_token)

        # Determine intervention need
        quality_below_threshold = quality_score < self.config.quality_threshold

        self.needs_intervention = quality_below_threshold or has_repetition
        return self.needs_intervention

    def apply_intervention(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply intervention to logits for this sequence."""
        if not self.needs_intervention:
            return logits

        self.intervention_count += 1

        # Apply penalties based on the type of issue
        if len(self.repetition_history) > 0 and self.repetition_history[-1] == 1:
            # Penalize repeating the same token
            if self.last_token is not None:
                logits[self.last_token] -= 2.0

        # Boost EOS token if quality is consistently low
        if len(self.quality_scores) >= 5:
            avg_quality = sum(list(self.quality_scores)[-5:]) / 5
            if avg_quality < self.config.quality_threshold * 0.8:
                logits[self.eos_idx] += 1.5

        return logits

    def update_stats(self):
        """Update sequence statistics."""
        self.total_tokens += 1
        self.needs_intervention = False


class BatchAlignmentStreamAnalyzer:
    """
    Efficient batch alignment stream analyzer for multilingual TTS models.

    Supports parallel quality analysis across multiple sequences while maintaining
    language-specific optimization and quality control.
    """

    def __init__(self, model: nn.Module, attention_layer_idx: int = 9,
                 enable_quality_monitoring: bool = True):
        self.model = model
        self.attention_layer_idx = attention_layer_idx
        self.enable_quality_monitoring = enable_quality_monitoring

        # Batch state management
        self.sequence_states: Dict[int, BatchSequenceState] = {}
        self.active_sequences: List[int] = []
        self.completed_sequences: List[int] = []

        # Performance optimization
        self.attention_cache = {}
        self.attention_hooks = {}

        # Statistics
        self.total_interventions = 0
        self.total_processed_tokens = 0

        logger.info(f"Initialized BatchAlignmentStreamAnalyzer with attention layer {attention_layer_idx}")

    def initialize_batch(self, batch_t3_conds: List[Any],
                        batch_text_tokens_slices: List[Tuple[int, int]],
                        eos_idx: int) -> None:
        """
        Initialize batch processing with multiple sequences.

        Args:
            batch_t3_conds: List of T3Cond objects for each sequence
            batch_text_tokens_slices: List of (start, end) tuples for text tokens
            eos_idx: End-of-sequence token index
        """
        self.sequence_states.clear()
        self.active_sequences.clear()
        self.completed_sequences.clear()

        for seq_id, (t3_cond, text_slice) in enumerate(zip(batch_t3_conds, batch_text_tokens_slices)):
            language = MultilinguaAlignmentConfig.detect_language(t3_cond)
            config = MultilinguaAlignmentConfig.get_config(language)

            state = BatchSequenceState(
                sequence_id=seq_id,
                language=language,
                text_tokens_slice=text_slice,
                eos_idx=eos_idx,
                config=config
            )

            self.sequence_states[seq_id] = state
            self.active_sequences.append(seq_id)

        logger.info(f"Initialized batch analysis for {len(self.active_sequences)} sequences")

    def register_attention_hooks(self):
        """Register attention hooks for quality monitoring."""
        if not self.enable_quality_monitoring:
            return

        def attention_hook(module, input, output):
            """Hook to capture attention weights."""
            if hasattr(output, 'attentions') and output.attentions is not None:
                # Store attention weights for analysis
                layer_attention = output.attentions[self.attention_layer_idx]
                self.attention_cache['current'] = layer_attention.detach()

        # Register hook on the model
        hook_handle = self.model.register_forward_hook(attention_hook)
        self.attention_hooks['main'] = hook_handle

    def remove_attention_hooks(self):
        """Remove attention hooks."""
        for hook_handle in self.attention_hooks.values():
            hook_handle.remove()
        self.attention_hooks.clear()

    def analyze_batch_attention(self, batch_size: int) -> None:
        """Analyze attention patterns for the current batch."""
        if not self.enable_quality_monitoring or 'current' not in self.attention_cache:
            return

        attention_weights = self.attention_cache['current']

        # Process attention for each active sequence
        for i, seq_id in enumerate(self.active_sequences[:batch_size]):
            if seq_id in self.sequence_states:
                # Extract attention for this sequence
                seq_attention = attention_weights[i] if i < attention_weights.size(0) else attention_weights[0]
                self.sequence_states[seq_id].update_attention(seq_attention)

    def batch_step(self, batch_logits: torch.Tensor,
                   next_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process a batch of logits with alignment-aware quality control.

        Args:
            batch_logits: Tensor of shape (batch_size, vocab_size)
            next_tokens: Previously generated tokens for repetition analysis

        Returns:
            Modified logits with quality interventions applied
        """
        batch_size = batch_logits.size(0)

        # Analyze attention patterns
        self.analyze_batch_attention(batch_size)

        # Apply interventions per sequence
        modified_logits = batch_logits.clone()

        for i, seq_id in enumerate(self.active_sequences[:batch_size]):
            if seq_id not in self.sequence_states or self.sequence_states[seq_id].is_completed:
                continue

            state = self.sequence_states[seq_id]

            # Determine next token for this sequence
            next_token = next_tokens[i].item() if next_tokens is not None else None

            # Check if intervention is needed
            if state.should_intervene(next_token or 0):
                # Apply sequence-specific intervention
                seq_logits = modified_logits[i:i+1].squeeze(0)
                modified_logits[i] = state.apply_intervention(seq_logits)
                self.total_interventions += 1

            # Update sequence statistics
            state.update_stats()
            self.total_processed_tokens += 1

        return modified_logits

    def mark_sequence_completed(self, sequence_ids: List[int]):
        """Mark sequences as completed."""
        for seq_id in sequence_ids:
            if seq_id in self.sequence_states:
                self.sequence_states[seq_id].is_completed = True
                if seq_id in self.active_sequences:
                    self.active_sequences.remove(seq_id)
                    self.completed_sequences.append(seq_id)

    def get_batch_statistics(self) -> Dict[str, Any]:
        """Get comprehensive batch processing statistics."""
        total_sequences = len(self.sequence_states)
        active_count = len(self.active_sequences)
        completed_count = len(self.completed_sequences)

        # Calculate per-language statistics
        language_stats = defaultdict(lambda: {'count': 0, 'interventions': 0, 'avg_quality': 0.0})

        for state in self.sequence_states.values():
            lang = state.language
            language_stats[lang]['count'] += 1
            language_stats[lang]['interventions'] += state.intervention_count

            if state.quality_scores:
                language_stats[lang]['avg_quality'] += sum(state.quality_scores) / len(state.quality_scores)

        # Average quality scores
        for lang_stat in language_stats.values():
            if lang_stat['count'] > 0:
                lang_stat['avg_quality'] /= lang_stat['count']

        return {
            'total_sequences': total_sequences,
            'active_sequences': active_count,
            'completed_sequences': completed_count,
            'total_interventions': self.total_interventions,
            'total_processed_tokens': self.total_processed_tokens,
            'intervention_rate': self.total_interventions / max(1, self.total_processed_tokens),
            'language_statistics': dict(language_stats),
            'average_quality': sum(
                sum(state.quality_scores) / len(state.quality_scores)
                for state in self.sequence_states.values()
                if state.quality_scores
            ) / max(1, len([s for s in self.sequence_states.values() if s.quality_scores]))
        }

    def cleanup(self):
        """Clean up resources and hooks."""
        self.remove_attention_hooks()
        self.attention_cache.clear()
        self.sequence_states.clear()
        self.active_sequences.clear()
        self.completed_sequences.clear()

        logger.info("BatchAlignmentStreamAnalyzer cleanup completed")

    def __enter__(self):
        """Context manager entry."""
        self.register_attention_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()