#!/usr/bin/env python3
"""
Quick test to verify the batch processing fixes work correctly.
"""

import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from chatterbox import ChatterboxTTS

def test_basic_generation():
    """Test basic generation to verify fixes."""
    logger.info("Initializing TTS model...")

    # Initialize TTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = ChatterboxTTS.from_pretrained(device=device)

    # Test single generation first
    logger.info("Testing single generation...")
    try:
        result = tts.generate("Hello, this is a test.")
        logger.info(f"Single generation successful! Output shape: {result.shape}")
    except Exception as e:
        logger.error(f"Single generation failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

    # Test batch generation
    logger.info("Testing batch generation...")
    try:
        test_texts = [
            "Hello, this is a test.",
            "This is another test sentence.",
            "Testing batch processing."
        ]

        results = tts.generate_batch(test_texts)
        logger.info(f"Batch generation successful! Generated {len(results)} results")

        for i, result in enumerate(results):
            logger.info(f"  Result {i}: {result.shape}")

        return True

    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        return False

def main():
    """Main test function."""
    logger.info("Starting batch processing test...")

    success = test_basic_generation()

    if success:
        logger.info("SUCCESS: Batch processing fixes are working correctly!")
    else:
        logger.error("FAILED: Batch processing still has issues")

    return success

if __name__ == "__main__":
    main()