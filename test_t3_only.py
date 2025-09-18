#!/usr/bin/env python3
"""
Test T3 batch processing optimizations only.
"""

import torch
import logging
from chatterbox import ChatterboxTTS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_t3_batch_processing():
    """Test T3 optimized batch processing."""
    logger.info("Testing T3 batch processing optimizations...")

    # Initialize TTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = ChatterboxTTS.from_pretrained(device=device)

    # Prepare some test texts
    test_texts = [
        "Hello, this is a test.",
        "This is another test sentence.",
        "Testing batch processing."
    ]

    # Tokenize the texts
    text_tokens = []
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

    # Create T3 conditioning
    if tts.conds is None:
        # Create a dummy conditioning for testing
        logger.warning("No conditioning available, using default")
        return False

    t3_conds = [tts.conds.t3] * len(test_texts)

    try:
        # Test optimized batch inference
        logger.info("Testing optimized T3 batch inference...")
        speech_tokens = tts.t3.optimized_batch_inference(
            batch_text_tokens=text_tokens,
            batch_t3_conds=t3_conds,
            max_new_tokens=100,
            max_batch_size=4,
            enable_dynamic_batching=True,
            memory_efficient_attention=True,
            cfg_weight=0.0  # Disable CFG for this test
        )

        logger.info(f"T3 batch inference successful! Generated {len(speech_tokens)} sequences")
        for i, tokens in enumerate(speech_tokens):
            logger.info(f"  Sequence {i}: {tokens.shape}")

        return True

    except Exception as e:
        logger.error(f"T3 batch inference failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

def main():
    """Main test function."""
    logger.info("Starting T3 batch processing test...")

    success = test_t3_batch_processing()

    if success:
        logger.info("SUCCESS: T3 batch processing optimizations are working!")
    else:
        logger.error("FAILED: T3 batch processing has issues")

    return success

if __name__ == "__main__":
    main()