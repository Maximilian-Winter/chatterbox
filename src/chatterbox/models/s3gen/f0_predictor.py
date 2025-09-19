# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Kai Hu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class ConvRNNF0Predictor(nn.Module):
    def __init__(self,
                 num_class: int = 1,
                 in_channels: int = 80,
                 cond_channels: int = 512
                 ):
        super().__init__()

        self.num_class = num_class
        self.condnet = nn.Sequential(
            weight_norm(
                nn.Conv1d(in_channels, cond_channels, kernel_size=3, padding=1)
            ),
            nn.ELU(),
            weight_norm(
                nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
            ),
            nn.ELU(),
            weight_norm(
                nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
            ),
            nn.ELU(),
            weight_norm(
                nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
            ),
            nn.ELU(),
            weight_norm(
                nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
            ),
            nn.ELU(),
        )
        self.classifier = nn.Linear(in_features=cond_channels, out_features=self.num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle edge cases with very small or empty inputs
        # This can happen when speech tokens are very short (e.g., only start/stop tokens) or empty
        original_length = x.shape[-1]
        min_length = 3  # kernel_size

        # Handle completely empty input
        if original_length == 0:
            # Return zero tensor with proper shape
            batch_size = x.shape[0]
            return torch.zeros(batch_size, 0, device=x.device, dtype=x.dtype)

        if original_length < min_length:
            # Pad the input to minimum required length
            pad_amount = min_length - original_length
            x = torch.nn.functional.pad(x, (0, pad_amount), mode='replicate')

        x = self.condnet(x)
        x = x.transpose(1, 2)

        # If we padded the input, trim the output back to original length
        if original_length < min_length:
            x = x[:, :original_length, :]

        return torch.abs(self.classifier(x).squeeze(-1))
