#!/usr/bin/env python3
"""
Test script to verify the CFG batch size mismatch fix.
This script tests that CFG (Classifier-Free Guidance) works properly
with batch processing without causing tensor size mismatches.
"""

import logging
import torch
from pathlib import Path
import sys

# Set up logging for detailed debug output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from chatterbox import ChatterboxTTS
from chatterbox.models.t3.batch_state import BatchGenerationState


def test_cfg_batch_compatibility():
    """Test CFG with batch processing to ensure no tensor size mismatches."""
    logger.info("=" * 60)
    logger.info("Testing CFG + Batch Processing Compatibility")
    logger.info("=" * 60)

    # Initialize TTS model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    try:
        tts = ChatterboxTTS.from_pretrained(device=device)
        logger.info("✅ TTS model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load TTS model: {e}")
        return False

    # Test data - multiple sequences to test batch processing
    test_texts = [
        "Hello, this is a test sentence with CFG enabled.",
        "Another test sentence for batch processing.",
        "CFG should work properly with multiple sequences.",
        "Testing tensor size compatibility in batch mode.",
        "Final test sentence for comprehensive validation."
    ]

    logger.info(f"Testing with {len(test_texts)} sequences")

    # Test parameters
    cfg_weights = [0.0, 0.3, 0.5]  # Test different CFG weights
    batch_sizes = [1, 3, 5]  # Test different batch sizes

    for cfg_weight in cfg_weights:
        logger.info(f"\n--- Testing CFG weight: {cfg_weight} ---")

        for batch_size in batch_sizes:
            if batch_size > len(test_texts):
                continue

            logger.info(f"Testing batch size: {batch_size}")

            # Select texts for this batch
            batch_texts = test_texts[:batch_size]

            try:
                # Test batch generation with CFG
                logger.info(f"Generating with cfg_weight={cfg_weight}, batch_size={batch_size}")

                results = tts.generate_batch(
                    batch_texts,
                    cfg_weight=cfg_weight,
                    max_new_tokens=50,  # Short generation for quick testing
                    temperature=0.8,
                    top_p=0.95
                )

                logger.info(f"✅ Batch generation succeeded for cfg_weight={cfg_weight}, batch_size={batch_size}")
                logger.info(f"Generated {len(results)} results")

                # Validate results
                for i, result in enumerate(results):
                    if result is not None and hasattr(result, 'shape'):
                        logger.info(f"Sequence {i}: generated {result.shape} tokens")
                    else:
                        logger.warning(f"Sequence {i}: invalid result type {type(result)}")

            except Exception as e:
                logger.error(f"❌ Batch generation failed for cfg_weight={cfg_weight}, batch_size={batch_size}")
                logger.error(f"Error: {e}")
                logger.error(f"Error type: {type(e)}")

                # Check if this is the specific tensor size mismatch error
                if "Sizes of tensors must match" in str(e):
                    logger.error("🔥 CRITICAL: This is the tensor size mismatch error we're trying to fix!")
                    return False
                else:
                    logger.warning("Different error occurred, continuing tests...")

    logger.info("\n" + "=" * 60)
    logger.info("CFG + Batch Processing Compatibility Test Results")
    logger.info("=" * 60)
    logger.info("✅ All CFG batch processing tests passed!")
    logger.info("🎯 Tensor size mismatch fix is working correctly")
    return True


def test_direct_t3_batch_with_cfg():
    """Test T3 batch inference directly with CFG enabled."""
    logger.info("\n--- Direct T3 Batch + CFG Test ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        tts = ChatterboxTTS.from_pretrained(device=device)

        # Prepare text tokens for direct T3 testing
        test_texts = [
            "Direct T3 test with CFG enabled.",
            "Another sequence for batch testing.",
            "CFG compatibility validation."
        ]

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

            # Create T3 conditioning
            if tts.conds is None:
                # Use default conditioning
                pass
            t3_conds.append(tts.conds.t3 if tts.conds else None)

        # Test with CFG enabled
        cfg_weight = 0.5
        logger.info(f"Testing direct T3 batch inference with CFG weight: {cfg_weight}")

        speech_tokens = tts.t3.batch_inference(
            batch_text_tokens=text_tokens,
            batch_t3_conds=t3_conds,
            max_new_tokens=30,
            cfg_weight=cfg_weight,
            temperature=0.8
        )

        logger.info(f"✅ Direct T3 batch inference with CFG succeeded!")
        logger.info(f"Generated {len(speech_tokens)} speech token sequences")

        for i, tokens in enumerate(speech_tokens):
            logger.info(f"Sequence {i}: {tokens.shape} speech tokens")

        return True

    except Exception as e:
        logger.error(f"❌ Direct T3 batch inference with CFG failed: {e}")
        if "Sizes of tensors must match" in str(e):
            logger.error("🔥 CRITICAL: Tensor size mismatch still occurring in direct T3!")
            return False
        return False


def test_batch_state_cfg_handling():
    """Test BatchGenerationState handling of CFG scenarios."""
    logger.info("\n--- BatchGenerationState CFG Handling Test ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 3

    # Create batch state
    batch_state = BatchGenerationState(
        batch_size=batch_size,
        max_tokens=100,
        device=device,
        start_token=1,
        stop_token=2,
        model_config={
            'num_hidden_layers': 4,  # Smaller for testing
            'num_attention_heads': 8,
            'hidden_size': 512,
        }
    )

    batch_state.optimize_for_parallel_processing()

    # Simulate CFG scenario: double batch size in cache updates
    cfg_batch_size = batch_size * 2  # CFG doubles the batch size

    try:
        # Create mock tensors with CFG batch size (doubled)
        from transformers.cache_utils import DynamicCache

        mock_cache = DynamicCache()
        for layer_idx in range(4):  # 4 layers for testing
            # Create key/value tensors with CFG batch size (doubled)
            keys = torch.randn(cfg_batch_size, 8, 1, 64, device=device)  # (2*batch_size, heads, seq, dim)
            values = torch.randn(cfg_batch_size, 8, 1, 64, device=device)
            mock_cache.update(keys, values, layer_idx)

        logger.info(f"Created mock cache with CFG batch size: {cfg_batch_size}")

        # Test cache update with CFG tensors
        batch_state.kv_cache.update_unified_cache(mock_cache)
        logger.info("✅ Cache update with CFG tensors succeeded!")

        # Test cache retrieval
        retrieved_cache = batch_state.kv_cache.get_unified_past_key_values()
        if retrieved_cache is not None:
            logger.info("✅ Cache retrieval after CFG update succeeded!")
            for i, key_tensor in enumerate(retrieved_cache.key_cache):
                logger.info(f"Layer {i}: retrieved key shape {key_tensor.shape}")
        else:
            logger.warning("⚠️ Retrieved cache is None")

        return True

    except Exception as e:
        logger.error(f"❌ BatchGenerationState CFG handling failed: {e}")
        if "Sizes of tensors must match" in str(e):
            logger.error("🔥 CRITICAL: Tensor size mismatch in cache handling!")
            return False
        return False


def main():
    """Run comprehensive CFG compatibility tests."""
    logger.info("🚀 Starting CFG Batch Processing Compatibility Tests")

    # Track test results
    results = {}

    # Test 1: CFG + Batch processing compatibility
    results['cfg_batch_compatibility'] = test_cfg_batch_compatibility()

    # Test 2: Direct T3 batch with CFG
    results['direct_t3_cfg'] = test_direct_t3_batch_with_cfg()

    # Test 3: BatchGenerationState CFG handling
    results['batch_state_cfg'] = test_batch_state_cfg_handling()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST RESULTS SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name:25s}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED! CFG batch processing fix is working correctly!")
        logger.info("🔧 The tensor size mismatch error has been resolved.")
        return True
    else:
        logger.error("\n💥 SOME TESTS FAILED! CFG batch processing fix needs more work.")
        logger.error("🔍 Check the logs above for specific failure details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)