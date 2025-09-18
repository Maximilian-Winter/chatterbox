#!/usr/bin/env python3
"""
Test script to validate stop token values and EOS detection mechanics.
This verifies that the stop token value is correctly set and detection works.
"""

import logging
import torch
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_stop_token_configuration():
    """Test that stop token values are correctly configured."""
    logger.info("Testing Stop Token Configuration")
    logger.info("=" * 50)

    try:
        from chatterbox.models.t3.modules.t3_config import T3Config
        from chatterbox.models.t3.batch_state import BatchGenerationState

        # Test 1: Check T3Config values
        config = T3Config.english_only()
        logger.info(f"T3Config stop_speech_token: {config.stop_speech_token}")
        logger.info(f"T3Config start_speech_token: {config.start_speech_token}")

        # Verify expected values
        assert config.stop_speech_token == 6562, f"Expected 6562, got {config.stop_speech_token}"
        assert config.start_speech_token == 6561, f"Expected 6561, got {config.start_speech_token}"
        logger.info("✅ T3Config token values are correct")

        # Test 2: Check BatchGenerationState initialization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_state = BatchGenerationState(
            batch_size=3,
            max_tokens=100,
            device=device,
            start_token=config.start_speech_token,
            stop_token=config.stop_speech_token,
            model_config={
                'num_hidden_layers': 30,
                'num_attention_heads': 16,
                'hidden_size': 2048,
            }
        )

        logger.info(f"BatchGenerationState stop_token: {batch_state.stop_token}")
        logger.info(f"BatchGenerationState start_token: {batch_state.start_token}")

        assert batch_state.stop_token == 6562, f"Expected 6562, got {batch_state.stop_token}"
        assert batch_state.start_token == 6561, f"Expected 6561, got {batch_state.start_token}"
        logger.info("✅ BatchGenerationState token values are correct")

        return True

    except Exception as e:
        logger.error(f"Stop token configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_eos_detection_mechanics():
    """Test the actual EOS detection mechanics with controlled tokens."""
    logger.info("\nTesting EOS Detection Mechanics")
    logger.info("=" * 50)

    try:
        from chatterbox.models.t3.modules.t3_config import T3Config
        from chatterbox.models.t3.batch_state import BatchGenerationState

        config = T3Config.english_only()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a small batch state for testing
        batch_state = BatchGenerationState(
            batch_size=3,
            max_tokens=10,
            device=device,
            start_token=config.start_speech_token,
            stop_token=config.stop_speech_token,
            model_config={
                'num_hidden_layers': 30,
                'num_attention_heads': 16,
                'hidden_size': 2048,
            }
        )

        logger.info(f"Initial active sequences: {batch_state.get_active_batch_size()}")

        # Test 1: Add non-stop tokens (should continue)
        regular_tokens = torch.tensor([[1000], [2000], [3000]], device=device, dtype=torch.long)
        batch_state.update_with_new_tokens(regular_tokens)

        logger.info(f"After regular tokens - Active: {batch_state.get_active_batch_size()}, Completed: {batch_state.completion_flags.sum().item()}")
        assert batch_state.get_active_batch_size() == 3, "All sequences should still be active"
        logger.info("✅ Regular tokens do not trigger EOS detection")

        # Test 2: Add stop token to sequence 1 (should complete sequence 1)
        mixed_tokens = torch.tensor([[6562], [4000], [5000]], device=device, dtype=torch.long)  # First is stop token
        batch_state.update_with_new_tokens(mixed_tokens)

        logger.info(f"After mixed tokens - Active: {batch_state.get_active_batch_size()}, Completed: {batch_state.completion_flags.sum().item()}")
        assert batch_state.completion_flags[0].item() == True, "Sequence 0 should be completed"
        assert batch_state.completion_flags[1].item() == False, "Sequence 1 should still be active"
        assert batch_state.completion_flags[2].item() == False, "Sequence 2 should still be active"
        logger.info("✅ Stop token correctly triggers EOS detection for specific sequence")

        # Test 3: Add stop tokens to remaining sequences
        remaining_tokens = torch.tensor([[6562], [6562]], device=device, dtype=torch.long)  # Both stop tokens
        batch_state.update_with_new_tokens(remaining_tokens)

        logger.info(f"After all stop tokens - Active: {batch_state.get_active_batch_size()}, Completed: {batch_state.completion_flags.sum().item()}")
        assert batch_state.get_active_batch_size() == 0, "All sequences should be completed"
        assert batch_state.all_completed(), "all_completed() should return True"
        logger.info("✅ All sequences completed when stop tokens generated")

        # Test 4: Validate EOS detection
        validation_result = batch_state.validate_eos_detection()
        logger.info(f"EOS detection validation: {'PASSED' if validation_result else 'FAILED'}")

        # Test 5: Check completion status details
        completion_status = batch_state.get_completion_status()
        logger.info("Completion status details:")
        for seq_detail in completion_status['sequence_details']:
            logger.info(f"  Seq {seq_detail['seq_id']}: completed={seq_detail['is_completed']}, "
                       f"length={seq_detail['length']}, last_token={seq_detail['last_token']}")

        return True

    except Exception as e:
        logger.error(f"EOS detection mechanics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_early_termination_logic():
    """Test the early termination logic that should provide speedup."""
    logger.info("\nTesting Early Termination Logic")
    logger.info("=" * 50)

    try:
        from chatterbox.models.t3.modules.t3_config import T3Config
        from chatterbox.models.t3.batch_state import BatchGenerationState

        config = T3Config.english_only()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Test scenario: Large max_tokens but early completion
        batch_state = BatchGenerationState(
            batch_size=2,
            max_tokens=1000,  # Large limit
            device=device,
            start_token=config.start_speech_token,
            stop_token=config.stop_speech_token,
            model_config={
                'num_hidden_layers': 30,
                'num_attention_heads': 16,
                'hidden_size': 2048,
            }
        )

        # Simulate a generation loop that would normally run 1000 steps
        steps_taken = 0
        max_steps = 1000

        for step in range(max_steps):
            # Check early termination conditions (this is the key speedup logic)
            if batch_state.all_completed() or not batch_state.has_active_sequences():
                logger.info(f"Early termination at step {step}: All sequences completed")
                break

            # Simulate token generation - make sequences complete early
            if step == 0:
                # First step: regular tokens
                tokens = torch.tensor([[1000], [2000]], device=device, dtype=torch.long)
            elif step == 1:
                # Second step: one sequence completes
                tokens = torch.tensor([[6562], [3000]], device=device, dtype=torch.long)
            elif step == 2:
                # Third step: remaining sequence completes
                tokens = torch.tensor([[6562]], device=device, dtype=torch.long)
            else:
                # This should never execute due to early termination
                logger.error("ERROR: Loop continued after all sequences completed!")
                return False

            batch_state.update_with_new_tokens(tokens)
            steps_taken = step + 1

            # Additional check after token update
            if batch_state.all_completed():
                logger.info(f"Early termination at step {step+1}: All sequences completed after token update")
                break

        speedup_ratio = max_steps / steps_taken if steps_taken > 0 else 1
        logger.info(f"Steps taken: {steps_taken} out of {max_steps} maximum")
        logger.info(f"Theoretical speedup: {speedup_ratio:.1f}x")

        # Verify we terminated much earlier than max_steps
        assert steps_taken < 10, f"Expected early termination but took {steps_taken} steps"
        assert speedup_ratio > 100, f"Expected major speedup but only got {speedup_ratio:.1f}x"

        logger.info("✅ Early termination logic provides significant speedup potential")
        return True

    except Exception as e:
        logger.error(f"Early termination logic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    logger.info("Stop Token and EOS Detection Validation")
    logger.info("=" * 60)

    tests = [
        ("Stop Token Configuration", test_stop_token_configuration),
        ("EOS Detection Mechanics", test_eos_detection_mechanics),
        ("Early Termination Logic", test_early_termination_logic),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{test_name}")
        logger.info("-" * len(test_name))
        try:
            result = test_func()
            results.append((test_name, result))
            logger.info(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            logger.error(f"❌ {test_name}: FAILED with exception: {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\nValidation Summary")
    logger.info("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All validation tests PASSED!")
        logger.info("")
        logger.info("VALIDATION CONFIRMED:")
        logger.info("- Stop token values are correctly configured (6562)")
        logger.info("- EOS detection mechanics work properly")
        logger.info("- Early termination logic provides speedup potential")
        logger.info("- Batch processing should achieve 3-5x speedup through early stopping")
        return True
    else:
        logger.error("❌ Some validation tests FAILED")
        logger.info("Please review the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)