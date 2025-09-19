"""
Comprehensive test suite for HiFTGenerator edge cases and kernel size fixes.

This test validates that the fixes handle:
1. Zero-width inputs (empty tensors)
2. Very small inputs (1-2 frames)
3. STFT padding issues
4. Convolution kernel size mismatches
5. Tensor dimension mismatches during fusion
6. Backward compatibility with normal inputs
"""

import torch
import traceback
from chatterbox.models.s3gen.hifigan import HiFTGenerator
from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor


def test_comprehensive_edge_cases():
    """Comprehensive test for all edge cases addressed in the kernel size fix"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running comprehensive edge case tests on device: {device}")

    # Initialize the model
    f0_predictor = ConvRNNF0Predictor()
    model = HiFTGenerator(
        sampling_rate=24000,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        f0_predictor=f0_predictor,
    )
    model = model.to(device)
    model.eval()

    test_results = {
        "passed": 0,
        "failed": 0,
        "errors": []
    }

    def run_test(test_name, test_func):
        print(f"\n{test_name}...")
        try:
            test_func()
            print(f"PASS: {test_name}")
            test_results["passed"] += 1
        except Exception as e:
            print(f"FAIL: {test_name} - {e}")
            test_results["failed"] += 1
            test_results["errors"].append((test_name, str(e)))

    # Test 1: Zero-width speech features
    def test_zero_width_inference():
        speech_feat = torch.zeros(1, 80, 0, device=device)
        cache_source = torch.zeros(1, 1, 0, device=device)
        with torch.no_grad():
            result = model.inference(speech_feat, cache_source)
        assert len(result) == 2, "Should return (speech, source) tuple"
        assert result[0].shape[0] == 1, "Should maintain batch dimension"

    # Test 2: Single frame speech features
    def test_single_frame_inference():
        speech_feat = torch.randn(1, 80, 1, device=device)
        cache_source = torch.zeros(1, 1, 0, device=device)
        with torch.no_grad():
            result = model.inference(speech_feat, cache_source)
        assert len(result) == 2, "Should return (speech, source) tuple"
        assert result[0].numel() > 0, "Should produce non-empty output"

    # Test 3: Two frame speech features
    def test_two_frame_inference():
        speech_feat = torch.randn(1, 80, 2, device=device)
        cache_source = torch.zeros(1, 1, 0, device=device)
        with torch.no_grad():
            result = model.inference(speech_feat, cache_source)
        assert len(result) == 2, "Should return (speech, source) tuple"
        assert result[0].numel() > 0, "Should produce non-empty output"

    # Test 4: Small decode method direct test
    def test_decode_small_inputs():
        # Test 1-frame decode
        speech_feat = torch.randn(1, 80, 1, device=device)
        source = torch.zeros(1, 1, 1, device=device)
        with torch.no_grad():
            result = model.decode(x=speech_feat, s=source)
        assert result.numel() > 0, "Should produce non-empty output"

        # Test 2-frame decode
        speech_feat = torch.randn(1, 80, 2, device=device)
        source = torch.zeros(1, 1, 2, device=device)
        with torch.no_grad():
            result = model.decode(x=speech_feat, s=source)
        assert result.numel() > 0, "Should produce non-empty output"

    # Test 5: Empty source tensor handling
    def test_empty_source_decode():
        speech_feat = torch.randn(1, 80, 10, device=device)
        empty_source = torch.zeros(1, 1, 0, device=device)
        with torch.no_grad():
            result = model.decode(x=speech_feat, s=empty_source)
        assert result.numel() > 0, "Should handle empty source gracefully"

    # Test 6: Normal inputs (backward compatibility)
    def test_normal_inputs():
        speech_feat = torch.randn(1, 80, 100, device=device)
        cache_source = torch.zeros(1, 1, 0, device=device)
        with torch.no_grad():
            result = model.inference(speech_feat, cache_source)
        assert len(result) == 2, "Should return (speech, source) tuple"
        assert result[0].numel() > 0, "Should produce non-empty output"

    # Test 7: Batch processing with mixed sizes
    def test_batch_mixed_sizes():
        # This tests that our fixes don't break when processing different sizes
        for size in [1, 2, 5, 10, 50, 100]:
            speech_feat = torch.randn(1, 80, size, device=device)
            cache_source = torch.zeros(1, 1, 0, device=device)
            with torch.no_grad():
                result = model.inference(speech_feat, cache_source)
            assert len(result) == 2, f"Failed for size {size}"

    # Test 8: STFT method edge cases
    def test_stft_edge_cases():
        # Test very small inputs to STFT
        small_input = torch.randn(1, 1, device=device)
        with torch.no_grad():
            real, imag = model._stft(small_input)
        assert real.numel() > 0 and imag.numel() > 0, "STFT should handle small inputs"

        # Test inputs smaller than n_fft
        tiny_input = torch.randn(1, 5, device=device)
        with torch.no_grad():
            real, imag = model._stft(tiny_input)
        assert real.numel() > 0 and imag.numel() > 0, "STFT should pad small inputs"

    # Run all tests
    run_test("Zero-width inference", test_zero_width_inference)
    run_test("Single frame inference", test_single_frame_inference)
    run_test("Two frame inference", test_two_frame_inference)
    run_test("Small inputs decode", test_decode_small_inputs)
    run_test("Empty source decode", test_empty_source_decode)
    run_test("Normal inputs (backward compatibility)", test_normal_inputs)
    run_test("Batch mixed sizes", test_batch_mixed_sizes)
    run_test("STFT edge cases", test_stft_edge_cases)

    # Print results
    print(f"\n{'='*50}")
    print(f"COMPREHENSIVE TEST RESULTS")
    print(f"{'='*50}")
    print(f"Tests passed: {test_results['passed']}")
    print(f"Tests failed: {test_results['failed']}")
    print(f"Total tests: {test_results['passed'] + test_results['failed']}")

    if test_results['failed'] > 0:
        print(f"\nFailed tests:")
        for test_name, error in test_results['errors']:
            print(f"  - {test_name}: {error}")
    else:
        print(f"\nAll tests passed! The kernel size fixes are working correctly.")

    return test_results['failed'] == 0


if __name__ == "__main__":
    success = test_comprehensive_edge_cases()
    exit(0 if success else 1)