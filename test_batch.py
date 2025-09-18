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
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = ChatterboxTTS.from_pretrained(device=device)

    # Test single generation first
    logger.info("Testing single generation...")
    try:
        # Test just the core generation process that had the tensor dimension issue

        # Generate speech tokens using T3 (use tts.generate for simpler interface)
        # But first, we'll just test the flow model directly by calling S3Gen inference with dummy data

        # Create dummy speech tokens to test the flow model
        # This simulates the tensor shapes that were causing the dimension mismatch

        # Create test tensors similar to what T3 would generate
        # The error occurred when prompt_token had wrong dimensions
        dummy_speech_tokens = torch.randint(0, 100, (1, 50), device=device)  # [batch_size=1, seq_len=50]

        logger.info(f"Testing S3Gen flow inference with dummy tokens shape: {dummy_speech_tokens.shape}")

        # Test S3Gen flow inference (this is where the tensor dimension error occurred)
        wav, _ = tts.s3gen.inference(
            speech_tokens=dummy_speech_tokens,
            ref_dict=tts.conds.gen,
        )

        logger.info(f"S3Gen flow inference successful! Output shape: {wav.shape}")

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
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
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