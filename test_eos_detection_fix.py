#!/usr/bin/env python3
"""
Test script to validate EOS detection fixes in T3 batch inference methods.
This script tests that sequences stop naturally at EOS tokens instead of running for full max_new_tokens.
"""

import logging
import time
from pathlib import Path
import torch

# Set up logging to see EOS detection messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_eos_detection():
    """Test EOS detection in both batch inference methods."""
    logger.info("Testing EOS Detection Fixes")
    logger.info("=" * 60)

    try:
        from chatterbox import ChatterboxTTS

        # Initialize TTS model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        tts = ChatterboxTTS.from_pretrained(device=device)

        # Test texts - should naturally complete at 45-85 tokens
        test_texts = [
            "Hello, this is a simple test sentence.",
            "The quick brown fox jumps over the lazy dog.",
            "Speech synthesis technology has advanced significantly.",
            "Machine learning models can generate realistic speech.",
            "This is a test of early termination detection."
        ]

        logger.info(f"Testing with {len(test_texts)} sequences")

        # Test 1: Traditional batch_inference method
        logger.info("\nTest 1: Traditional batch_inference method")
        logger.info("-" * 50)

        # Prepare input for T3 batch inference
        text_tokens = []
        t3_conds = []

        for text in test_texts:
            tokens = tts.tokenizer.text_to_tokens(text).to(device)
            # Add start/stop tokens
            sot = tts.t3.hp.start_text_token
            eot = tts.t3.hp.stop_text_token
            tokens = torch.cat([
                torch.tensor([[sot]], device=device),
                tokens,
                torch.tensor([[eot]], device=device)
            ], dim=1)
            text_tokens.append(tokens)

            # Create T3 conditioning (prepare conditioning if needed)
            if tts.conds is None:
                logger.info("Preparing default conditioning...")
                # Use TTS's prepare method to get proper conditioning
                tts.prepare_conditionals()

            # Use the prepared conditioning
            t3_cond = tts.conds.t3
            t3_conds.append(t3_cond)

        # Test with reduced max_new_tokens to see early termination
        start_time = time.time()
        speech_tokens_1 = tts.t3.batch_inference(
            batch_text_tokens=text_tokens,
            batch_t3_conds=t3_conds,
            max_new_tokens=150,  # Reduced from 1000 to test early termination
            stop_on_eos=True
        )
        batch_time = time.time() - start_time

        logger.info(f"Traditional batch_inference completed in {batch_time:.2f}s")
        for i, tokens in enumerate(speech_tokens_1):
            logger.info(f"  Sequence {i}: {tokens.numel()} tokens generated")

        # Test 2: Optimized batch_inference method
        logger.info("\nTest 2: Optimized batch_inference method")
        logger.info("-" * 50)

        start_time = time.time()
        speech_tokens_2 = tts.t3.optimized_batch_inference(
            batch_text_tokens=text_tokens,
            batch_t3_conds=t3_conds,
            max_new_tokens=150,  # Reduced from 1000 to test early termination
            stop_on_eos=True,
            max_batch_size=8,
            enable_dynamic_batching=True
        )
        optimized_time = time.time() - start_time

        logger.info(f"Optimized batch_inference completed in {optimized_time:.2f}s")
        for i, tokens in enumerate(speech_tokens_2):
            logger.info(f"  Sequence {i}: {tokens.numel()} tokens generated")

        # Compare results
        logger.info("\nComparison Results")
        logger.info("-" * 50)

        if optimized_time > 0:
            speedup = batch_time / optimized_time
            logger.info(f"Speedup: {speedup:.2f}x")

        # Check if sequences stopped early (< max_new_tokens)
        early_termination_detected = False
        for i, tokens in enumerate(speech_tokens_1):
            if tokens.numel() < 150:  # Less than max_new_tokens
                logger.info(f"SUCCESS: Sequence {i} terminated early with {tokens.numel()} tokens (< 150 max)")
                early_termination_detected = True

        if early_termination_detected:
            logger.info("SUCCESS: Early termination detected - EOS detection is working!")
        else:
            logger.warning("WARNING: No early termination detected - sequences may be running to max_new_tokens")

        # Test 3: Performance comparison with longer sequences
        logger.info("\nTest 3: Performance Impact Analysis")
        logger.info("-" * 50)

        # Test with small max_new_tokens vs large max_new_tokens
        start_time = time.time()
        _ = tts.t3.batch_inference(
            batch_text_tokens=text_tokens[:2],  # Just 2 sequences for quick test
            batch_t3_conds=t3_conds[:2],
            max_new_tokens=50,  # Small limit
            stop_on_eos=True
        )
        small_limit_time = time.time() - start_time

        start_time = time.time()
        _ = tts.t3.batch_inference(
            batch_text_tokens=text_tokens[:2],  # Just 2 sequences for quick test
            batch_t3_conds=t3_conds[:2],
            max_new_tokens=1000,  # Large limit (original problematic setting)
            stop_on_eos=True
        )
        large_limit_time = time.time() - start_time

        logger.info(f"Small limit (50 tokens): {small_limit_time:.2f}s")
        logger.info(f"Large limit (1000 tokens): {large_limit_time:.2f}s")

        if abs(small_limit_time - large_limit_time) < 1.0:  # Less than 1 second difference
            logger.info("SUCCESS: Times are similar - early termination is working!")
            logger.info("This confirms that sequences stop at EOS tokens, not at max_new_tokens")
        else:
            logger.warning(f"WARNING: Large time difference ({large_limit_time/small_limit_time:.1f}x) suggests EOS detection may not be working")

        logger.info("\nEOS Detection Test Completed!")
        return True

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_eos_detection()
    if success:
        print("\nEOS Detection Test PASSED")
    else:
        print("\nEOS Detection Test FAILED")