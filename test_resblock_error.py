import torch
import traceback
from chatterbox.models.s3gen.hifigan import ResBlock

def test_resblock_kernel_error():
    """Test ResBlock with very small inputs to trigger kernel size error"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing ResBlock on device: {device}")

    # Create ResBlock with default settings (kernel_size=3, dilations=[1,3,5])
    resblock = ResBlock(channels=256, kernel_size=3, dilations=[1, 3, 5]).to(device)
    resblock.eval()

    print("\n1. Testing ResBlock with 1 time frame...")
    try:
        x_small = torch.randn(1, 256, 1, device=device)  # [batch, channels, time=1]
        with torch.no_grad():
            result = resblock(x_small)
        print("PASS: ResBlock 1-frame test passed")
    except Exception as e:
        print(f"FAIL: ResBlock 1-frame test failed: {e}")
        print(f"Error type: {type(e).__name__}")

    print("\n2. Testing ResBlock with 2 time frames...")
    try:
        x_small = torch.randn(1, 256, 2, device=device)  # [batch, channels, time=2]
        with torch.no_grad():
            result = resblock(x_small)
        print("PASS: ResBlock 2-frame test passed")
    except Exception as e:
        print(f"FAIL: ResBlock 2-frame test failed: {e}")
        print(f"Error type: {type(e).__name__}")

    print("\n3. Testing ResBlock with 5 time frames...")
    try:
        x_small = torch.randn(1, 256, 5, device=device)  # [batch, channels, time=5]
        with torch.no_grad():
            result = resblock(x_small)
        print("PASS: ResBlock 5-frame test passed")
    except Exception as e:
        print(f"FAIL: ResBlock 5-frame test failed: {e}")
        print(f"Error type: {type(e).__name__}")

    print("\n4. Testing ResBlock with normal input...")
    try:
        x_normal = torch.randn(1, 256, 100, device=device)  # [batch, channels, time=100]
        with torch.no_grad():
            result = resblock(x_normal)
        print("PASS: ResBlock normal test passed")
    except Exception as e:
        print(f"FAIL: ResBlock normal test failed: {e}")
        print(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    test_resblock_kernel_error()