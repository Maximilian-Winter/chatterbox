import torch
import traceback
from chatterbox.models.s3gen.hifigan import HiFTGenerator
from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor

def test_kernel_size_error():
    """Test cases that might trigger the kernel size error in HiFTGenerator"""

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    print(f"Testing on device: {device}")

    # Test case 1: Zero-width input (this is the most likely to cause the error)
    print("\n1. Testing zero-width input...")
    try:
        speech_feat_zero = torch.zeros(1, 80, 0, device=device)  # [batch, 80, 0]
        cache_source_zero = torch.zeros(1, 1, 0, device=device)
        with torch.no_grad():
            result = model.inference(speech_feat_zero, cache_source_zero)
        print("PASS: Zero-width input test passed")
    except Exception as e:
        print(f"FAIL: Zero-width input test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()

    # Test case 2: Very small input (1 frame)
    print("\n2. Testing 1-frame input...")
    try:
        speech_feat_small = torch.randn(1, 80, 1, device=device)  # [batch, 80, 1]
        cache_source_small = torch.zeros(1, 1, 0, device=device)
        with torch.no_grad():
            result = model.inference(speech_feat_small, cache_source_small)
        print("PASS: 1-frame input test passed")
    except Exception as e:
        print(f"FAIL: 1-frame input test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()

    # Test case 3: 2-frame input
    print("\n3. Testing 2-frame input...")
    try:
        speech_feat_tiny = torch.randn(1, 80, 2, device=device)  # [batch, 80, 2]
        cache_source_tiny = torch.zeros(1, 1, 0, device=device)
        with torch.no_grad():
            result = model.inference(speech_feat_tiny, cache_source_tiny)
        print("PASS: 2-frame input test passed")
    except Exception as e:
        print(f"FAIL: 2-frame input test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()

    # Test case 4: Normal input (should always work)
    print("\n4. Testing normal input...")
    try:
        speech_feat_normal = torch.randn(1, 80, 100, device=device)  # [batch, 80, 100]
        cache_source_normal = torch.zeros(1, 1, 0, device=device)
        with torch.no_grad():
            result = model.inference(speech_feat_normal, cache_source_normal)
        print("PASS: Normal input test passed")
    except Exception as e:
        print(f"FAIL: Normal input test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()

if __name__ == "__main__":
    test_kernel_size_error()