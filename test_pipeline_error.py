import torch
import traceback
from chatterbox.models.s3gen.hifigan import HiFTGenerator
from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor

def test_decode_method():
    """Test the decode method specifically to find kernel size issues"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing decode method on device: {device}")

    # Initialize the model components
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

    print("\n=== Testing decode method directly ===")

    # Test case 1: Very small speech_feat (after conv_pre, this will be even smaller)
    print("\n1. Testing decode with 1 time frame...")
    try:
        speech_feat_small = torch.randn(1, 80, 1, device=device)  # [batch, 80, 1]
        source_small = torch.zeros(1, 1, 1, device=device)  # minimal source
        with torch.no_grad():
            result = model.decode(x=speech_feat_small, s=source_small)
        print("PASS: decode 1-frame test passed")
        print(f"Output shape: {result.shape}")
    except Exception as e:
        print(f"FAIL: decode 1-frame test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()

    # Test case 2: Slightly larger input
    print("\n2. Testing decode with 2 time frames...")
    try:
        speech_feat_small = torch.randn(1, 80, 2, device=device)  # [batch, 80, 2]
        source_small = torch.zeros(1, 1, 2, device=device)
        with torch.no_grad():
            result = model.decode(x=speech_feat_small, s=source_small)
        print("PASS: decode 2-frame test passed")
        print(f"Output shape: {result.shape}")
    except Exception as e:
        print(f"FAIL: decode 2-frame test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()

    print("\n=== Testing individual components ===")

    # Test conv_pre specifically
    print("\n3. Testing conv_pre with small input...")
    try:
        speech_feat_tiny = torch.randn(1, 80, 1, device=device)
        with torch.no_grad():
            x = model.conv_pre(speech_feat_tiny)
        print(f"PASS: conv_pre passed, output shape: {x.shape}")
    except Exception as e:
        print(f"FAIL: conv_pre failed: {e}")
        traceback.print_exc()

    # Test upsampling layers
    print("\n4. Testing upsampling layers...")
    try:
        # Start with a small feature
        x = torch.randn(1, 512, 1, device=device)  # After conv_pre
        for i, up_layer in enumerate(model.ups):
            with torch.no_grad():
                x = up_layer(x)
            print(f"  After upsample {i}: shape {x.shape}")
    except Exception as e:
        print(f"FAIL: upsampling failed at layer {i}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_decode_method()