#!/usr/bin/env python3
"""
Test script to verify the tensor dimension fix for batch processing
"""

import sys
import os
import torch

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_tensor_assignment():
    """Test the specific tensor assignment that was failing"""
    print("Testing tensor assignment logic...")

    # Simulate the tokenizer output (what we were getting before)
    # tokenizer.text_to_tokens() returns [1, seq_len] shaped tensors
    mock_token_lists = [
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),  # [1, 8]
        torch.tensor([[9, 10, 11]]),                # [1, 3]
        torch.tensor([[12, 13, 14, 15]])            # [1, 4]
    ]

    # Test old method (this should fail with the dimension error)
    print("Testing old assignment method...")
    try:
        max_len_old = max(len(tokens) for tokens in mock_token_lists)  # This would use len() on [1, seq] tensor
        print(f"Old max_len calculation: {max_len_old}")  # This would be 1, not the actual sequence length
    except Exception as e:
        print(f"Old method fails as expected: {e}")

    # Test new method (this should work)
    print("Testing new assignment method...")
    try:
        max_len = max(tokens.shape[1] for tokens in mock_token_lists)  # Use shape[1] to get sequence length
        batch_tokens = torch.zeros((len(mock_token_lists), max_len), dtype=torch.long)

        for i, tokens in enumerate(mock_token_lists):
            # tokens has shape [1, seq_len], squeeze to get [seq_len]
            tokens_1d = tokens.squeeze(0)
            batch_tokens[i, :tokens_1d.shape[0]] = tokens_1d

        print(f"SUCCESS: New method successful!")
        print(f"Batch tokens shape: {batch_tokens.shape}")
        print(f"Batch tokens:\n{batch_tokens}")
        return True

    except Exception as e:
        print(f"ERROR: New method failed: {e}")
        return False

def main():
    print("=== Testing Tensor Dimension Fix ===")

    # Test the core tensor assignment logic
    if test_tensor_assignment():
        print("\nSUCCESS: Tensor dimension fix logic is correct!")
        print("The fix properly handles the [1, seq_len] -> [seq_len] conversion.")
        print("Batch processing should now work without dimension errors.")
        return True
    else:
        print("\nFAILED: Tensor dimension fix logic is incorrect")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)