#!/usr/bin/env python3
"""
Test script to verify the f0_predictor fix for small inputs.
This script tests edge cases that previously caused kernel size errors.
"""

import torch
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor

def test_f0_predictor_edge_cases():
    """Test f0_predictor with various edge cases."""
    print("Testing f0_predictor with edge cases...")

    # Initialize the predictor
    predictor = ConvRNNF0Predictor(
        num_class=1,
        in_channels=80,
        cond_channels=512
    )
    predictor.eval()

    # Test cases
    test_cases = [
        ("Empty input (0 frames)", torch.randn(1, 80, 0)),
        ("Single frame input", torch.randn(1, 80, 1)),
        ("Two frames input (original error case)", torch.randn(1, 80, 2)),
        ("Three frames input (minimum for kernel_size=3)", torch.randn(1, 80, 3)),
        ("Normal input", torch.randn(1, 80, 10)),
    ]

    all_passed = True

    for name, input_tensor in test_cases:
        try:
            print(f"\n  Testing {name}: input shape {input_tensor.shape}")

            with torch.no_grad():
                output = predictor(input_tensor)

            print(f"    [OK] Success: output shape {output.shape}")

            # Verify output shape is correct
            expected_time_dim = input_tensor.shape[-1]
            if output.shape[-1] != expected_time_dim:
                print(f"    [ERROR] Expected time dimension {expected_time_dim}, got {output.shape[-1]}")
                all_passed = False
            else:
                print(f"    [OK] Output time dimension matches input: {expected_time_dim}")

        except Exception as e:
            print(f"    [ERROR] Failed with error: {str(e)}")
            all_passed = False

    return all_passed

if __name__ == "__main__":
    success = test_f0_predictor_edge_cases()

    if success:
        print("\n[SUCCESS] All tests passed! The f0_predictor fix is working correctly.")
        sys.exit(0)
    else:
        print("\n[FAILED] Some tests failed. Please check the implementation.")
        sys.exit(1)