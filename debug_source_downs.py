import torch
import numpy as np
from chatterbox.models.s3gen.hifigan import HiFTGenerator
from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor

def debug_source_downs():
    """Debug the source_downs layer configurations"""

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

    print("=== Source downs configuration ===")
    upsample_rates = [8, 5, 3]
    istft_params = {"n_fft": 16, "hop_len": 4}
    source_resblock_kernel_sizes = [7, 7, 11]

    downsample_rates = [1] + upsample_rates[::-1][:-1]
    downsample_cum_rates = np.cumprod(downsample_rates)

    print(f"upsample_rates: {upsample_rates}")
    print(f"downsample_rates: {downsample_rates}")
    print(f"downsample_cum_rates: {downsample_cum_rates}")
    print(f"source_resblock_kernel_sizes: {source_resblock_kernel_sizes}")

    for i, (u, k) in enumerate(zip(downsample_cum_rates[::-1], source_resblock_kernel_sizes)):
        if u == 1:
            kernel_size = 1
            stride = 1
            padding = 0
        else:
            kernel_size = u * 2
            stride = u
            padding = u // 2

        print(f"source_downs[{i}]: kernel_size={kernel_size}, stride={stride}, padding={padding}")

    print("\n=== Testing small STFT processing ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Test STFT output size for small inputs
    with torch.no_grad():
        test_input = torch.zeros(1, 16, device=device)  # minimum size after our fix
        stft_real, stft_imag = model._stft(test_input)
        stft_combined = torch.cat([stft_real, stft_imag], dim=1)
        print(f"STFT output shape: {stft_combined.shape}")
        print(f"STFT time dimension: {stft_combined.shape[-1]}")

if __name__ == "__main__":
    debug_source_downs()