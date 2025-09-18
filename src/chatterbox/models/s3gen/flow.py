# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
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
import logging
import random
from typing import Dict, Optional

logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
from torch.nn import functional as F
from .utils.mask import make_pad_mask
from .configs import CFM_PARAMS


class MaskedDiffWithXvec(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 4096,
        input_frame_rate: int = 50,
        only_mask_loss: bool = True,
        encoder: torch.nn.Module = None,
        length_regulator: torch.nn.Module = None,
        decoder: torch.nn.Module = None,
        decoder_conf: Dict = {
            'in_channels': 240,
            'out_channel': 80,
            'spk_emb_dim': 80,
            'n_spks': 1,
            'cfm_params': CFM_PARAMS,
            'decoder_params': {
                'channels': [256, 256],
                'dropout': 0.0,
                'attention_head_dim': 64,
                'n_blocks': 4,
                'num_mid_blocks': 12,
                'num_heads': 8,
                'act_fn': 'gelu',
            }
        },
        mel_feat_conf: Dict = {
            'n_fft': 1024,
            'num_mels': 80,
            'sampling_rate': 22050,
            'hop_size': 256,
            'win_size': 1024,
            'fmin': 0,
            'fmax': 8000
        }
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.length_regulator = length_regulator
        self.only_mask_loss = only_mask_loss

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        token = batch['speech_token'].to(device)
        token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device)
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['embedding'].to(device)

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, min=0, max=self.input_embedding.num_embeddings-1)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        h, h_lengths = self.length_regulator(h, feat_len)

        # get conditions
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(feat_len)).to(h)
        feat = F.interpolate(feat.unsqueeze(dim=1), size=h.shape[1:], mode="nearest").squeeze(dim=1)
        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  flow_cache):
        if self.fp16 is True:
            prompt_feat = prompt_feat.half()
            embedding = embedding.half()

        # Support batch processing - removed batch_size=1 constraint
        batch_size = token.shape[0]
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text - handle batch dimension mismatch carefully

        # Handle embedding dimension issues - similar to prompt_token
        if len(embedding.shape) == 2 and embedding.shape[0] > 1 and token.shape[0] == 1:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Fixing embedding dimensions: {embedding.shape} -> batch format for single inference")
            # Take the first embedding if it's multi-dimensional
            embedding = embedding[0:1, :]
            logger.info(f"Fixed embedding to {embedding.shape}")

        # First, handle prompt_token dimension issues similar to the second inference method
        if len(prompt_token.shape) == 2 and prompt_token.shape[0] > 1 and token.shape[0] == 1:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Fixing prompt_token dimensions: {prompt_token.shape} -> batch format for single inference")

            # For 2D prompt_token, we need to determine which dimension is sequence length
            # Compare with token.shape[1] which we know is the correct sequence length
            if prompt_token.shape[1] == token.shape[1]:
                # [batch_wrong, seq_len] case - first dimension is wrong batch size, second is correct seq_len
                prompt_token = prompt_token[0:1, :]  # [batch_wrong, seq_len] -> [1, seq_len]
                logger.info(f"Fixed prompt_token from [batch_wrong, seq_len] to {prompt_token.shape}")
            elif prompt_token.shape[0] == token.shape[1]:
                # [seq_len, batch_wrong] case - first dimension is seq_len, second is wrong batch
                prompt_token = prompt_token.transpose(0, 1)[0:1, :]  # [seq_len, batch_wrong] -> [batch_wrong, seq_len] -> [1, seq_len]
                logger.info(f"Fixed prompt_token from [seq_len, batch_wrong] to {prompt_token.shape}")
            else:
                # Fallback: assume the larger dimension is sequence length
                if prompt_token.shape[0] < prompt_token.shape[1]:
                    prompt_token = prompt_token[0:1, :]
                else:
                    prompt_token = prompt_token.transpose(0, 1)[0:1, :]
                logger.info(f"Fixed prompt_token using fallback logic to {prompt_token.shape}")

            # Also fix prompt_token_len to match the new prompt_token sequence length
            # The prompt_token_len should be the actual sequence length of the reshaped prompt_token
            prompt_token_len = torch.tensor([prompt_token.shape[1]], device=prompt_token.device)

        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]

        # Handle case 1: prompt_token has batch_size=1 and token has larger batch_size
        if prompt_token.shape[0] == 1 and token.shape[0] > 1:
            # Broadcast prompt_token to match token batch size
            prompt_token = prompt_token.expand(token.shape[0], -1)
            # Handle prompt_token_len expansion safely
            if isinstance(prompt_token_len, torch.Tensor):
                if prompt_token_len.numel() == 1:
                    prompt_token_len = prompt_token_len.expand(token.shape[0])
                elif len(prompt_token_len.shape) == 0:
                    prompt_token_len = prompt_token_len.unsqueeze(0).expand(token.shape[0])
            else:
                # Handle case where prompt_token_len is a scalar
                prompt_token_len = torch.tensor([prompt_token_len] * token.shape[0], device=token.device)

        # Handle case 2: token has batch_size=1 and prompt_token has larger batch_size
        elif token.shape[0] == 1 and prompt_token.shape[0] > 1:
            # Broadcast token to match prompt_token batch size
            token = token.expand(prompt_token.shape[0], -1)
            # Handle token_len expansion safely
            if isinstance(token_len, torch.Tensor):
                if token_len.numel() == 1:
                    token_len = token_len.expand(prompt_token.shape[0])
                elif len(token_len.shape) == 0:
                    token_len = token_len.unsqueeze(0).expand(prompt_token.shape[0])
            else:
                # Handle case where token_len is a scalar
                token_len = torch.tensor([token_len] * prompt_token.shape[0], device=token.device)

        # Handle case 3: Both have different batch sizes and neither is 1 (error case)
        elif prompt_token.shape[0] != token.shape[0]:
            # This is a true mismatch that we can't handle
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Batch dimension mismatch: prompt_token.shape={prompt_token.shape}, token.shape={token.shape}")
            raise ValueError(f"Incompatible batch dimensions: prompt_token batch_size={prompt_token.shape[0]}, token batch_size={token.shape[0]}. "
                           f"One tensor must have batch_size=1 for broadcasting, or both must have the same batch_size.")

        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        
        # Check for out-of-bounds token IDs
        vocab_size = self.input_embedding.num_embeddings
        if token.max() >= vocab_size or token.min() < 0:
            logging.warning(f"S3Gen: Token IDs out of bounds: min={token.min().item()}, max={token.max().item()}, vocab_size={vocab_size}")
        
        token = self.input_embedding(torch.clamp(token, min=0, max=vocab_size-1)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / self.input_frame_rate * 22050 / 256)
        h, h_lengths = self.length_regulator.inference(h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2, self.input_frame_rate)

        # get conditions - handle batch processing
        conds = torch.zeros([batch_size, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)

        # Handle prompt_feat batch dimension
        if prompt_feat.shape[0] != batch_size:
            if prompt_feat.shape[0] == 1:
                # Expand prompt_feat to match batch_size
                prompt_feat = prompt_feat.expand(batch_size, -1, -1)
            else:
                # Mismatch we can't handle with expand - this is an error
                raise ValueError(f"Incompatible prompt_feat batch dimension: prompt_feat.shape[0]={prompt_feat.shape[0]}, "
                               f"batch_size={batch_size}. For expansion, prompt_feat must have batch_size=1.")

        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2] * batch_size))).to(h)
        feat, flow_cache = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            prompt_len=mel_len1,
            flow_cache=flow_cache
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), flow_cache


class CausalMaskedDiffWithXvec(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 6561,
        input_frame_rate: int = 25,
        only_mask_loss: bool = True,
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
        encoder: torch.nn.Module = None,
        decoder: torch.nn.Module = None,
        decoder_conf: Dict = {
            'in_channels': 240,
            'out_channel': 80,
            'spk_emb_dim': 80,
            'n_spks': 1,
            'cfm_params': CFM_PARAMS,
            'decoder_params': {
                'channels': [256, 256],
                'dropout': 0.0,
                'attention_head_dim': 64,
                'n_blocks': 4,
                'num_mid_blocks': 12,
                'num_heads': 8,
                'act_fn': 'gelu',
            }
        },
        mel_feat_conf: Dict = {
            'n_fft': 1024,
            'num_mels': 80,
            'sampling_rate': 22050,
            'hop_size': 256,
            'win_size': 1024,
            'fmin': 0,
            'fmax': 8000
        }
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len

        # FIXME: this was missing - just putting it in as false
        self.fp16 = False

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  finalize):
        if self.fp16 is True:
            prompt_feat = prompt_feat.half()
            embedding = embedding.half()

        # Support batch processing - removed batch_size=1 constraint
        # Handle potential dimension confusion where tokens might come in wrong shapes
        # For single inference, ensure both prompt_token and token have consistent batch dimensions

        # Log the shapes for debugging
        logger.info(f"Input shapes: prompt_token={prompt_token.shape}, token={token.shape}, prompt_feat={prompt_feat.shape}, embedding={embedding.shape}")

        # Handle embedding dimension issues - similar to prompt_token
        if len(embedding.shape) == 2 and embedding.shape[0] > 1 and token.shape[0] == 1:
            logger.warning(f"Fixing embedding dimensions: {embedding.shape} -> batch format for single inference")
            # Take the first embedding if it's multi-dimensional
            embedding = embedding[0:1, :]
            logger.info(f"Fixed embedding to {embedding.shape}")

        # Handle prompt_token dimension issues
        if len(prompt_token.shape) == 2 and prompt_token.shape[0] > 1 and token.shape[0] == 1:
            logger.warning(f"Fixing prompt_token dimensions: {prompt_token.shape} -> batch format for single inference")

            # For 2D prompt_token, we need to determine which dimension is sequence length
            # Compare with token.shape[1] which we know is the correct sequence length
            if prompt_token.shape[1] == token.shape[1]:
                # [batch_wrong, seq_len] case - first dimension is wrong batch size, second is correct seq_len
                # Take the first element of the wrong batch dimension
                prompt_token = prompt_token[0:1, :]  # [batch_wrong, seq_len] -> [1, seq_len]
                logger.info(f"Fixed prompt_token from [batch_wrong, seq_len] to {prompt_token.shape}")
            elif prompt_token.shape[0] == token.shape[1]:
                # [seq_len, batch_wrong] case - first dimension is seq_len, second is wrong batch
                # Transpose to get correct shape
                prompt_token = prompt_token.transpose(0, 1)[0:1, :]  # [seq_len, batch_wrong] -> [batch_wrong, seq_len] -> [1, seq_len]
                logger.info(f"Fixed prompt_token from [seq_len, batch_wrong] to {prompt_token.shape}")
            else:
                # Fallback: assume the larger dimension is sequence length
                if prompt_token.shape[0] < prompt_token.shape[1]:
                    # [batch_wrong, seq_len] case
                    prompt_token = prompt_token[0:1, :]
                else:
                    # [seq_len, batch_wrong] case
                    prompt_token = prompt_token.transpose(0, 1)[0:1, :]
                logger.info(f"Fixed prompt_token using fallback logic to {prompt_token.shape}")

            # Also fix prompt_token_len to match the new prompt_token sequence length
            # The prompt_token_len should be the actual sequence length of the reshaped prompt_token
            prompt_token_len = torch.tensor([prompt_token.shape[1]], device=prompt_token.device)

        # Handle prompt_feat dimension issues
        if len(prompt_feat.shape) >= 2 and prompt_feat.shape[0] > 1 and prompt_feat.shape[0] != token.shape[0]:
            logger.warning(f"Fixing prompt_feat dimensions: {prompt_feat.shape} -> reshaping for single inference")

            # prompt_feat likely has shape [seq_len, feature_len, embed_dim] but should be [batch_size=1, feature_len, embed_dim]
            # Take the average or first element across the sequence dimension
            if len(prompt_feat.shape) == 3:
                # [seq_len, feature_len, embed_dim] -> [1, feature_len, embed_dim]
                # Use mean across sequence dimension for stability
                prompt_feat = prompt_feat.mean(dim=0, keepdim=True)  # [seq_len, feature_len, embed_dim] -> [1, feature_len, embed_dim]
            else:
                # For 2D case, just add batch dimension
                prompt_feat = prompt_feat.unsqueeze(0)  # Add batch dimension

        # Log the fixed shapes
        logger.info(f"Fixed shapes: prompt_token={prompt_token.shape}, token={token.shape}, prompt_feat={prompt_feat.shape}")

        batch_size = token.shape[0]
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text - handle batch dimension mismatch carefully
        # Handle case 1: prompt_token has batch_size=1 and token has larger batch_size
        if prompt_token.shape[0] == 1 and token.shape[0] > 1:
            # Broadcast prompt_token to match token batch size
            prompt_token = prompt_token.expand(token.shape[0], -1)
            # Handle prompt_token_len expansion safely
            if isinstance(prompt_token_len, torch.Tensor):
                if prompt_token_len.numel() == 1:
                    prompt_token_len = prompt_token_len.expand(token.shape[0])
                elif len(prompt_token_len.shape) == 0:
                    prompt_token_len = prompt_token_len.unsqueeze(0).expand(token.shape[0])
            else:
                # Handle case where prompt_token_len is a scalar
                prompt_token_len = torch.tensor([prompt_token_len] * token.shape[0], device=token.device)

        # Handle case 2: token has batch_size=1 and prompt_token has larger batch_size
        elif token.shape[0] == 1 and prompt_token.shape[0] > 1:
            # Broadcast token to match prompt_token batch size
            token = token.expand(prompt_token.shape[0], -1)
            # Handle token_len expansion safely
            if isinstance(token_len, torch.Tensor):
                if token_len.numel() == 1:
                    token_len = token_len.expand(prompt_token.shape[0])
                elif len(token_len.shape) == 0:
                    token_len = token_len.unsqueeze(0).expand(prompt_token.shape[0])
            else:
                # Handle case where token_len is a scalar
                token_len = torch.tensor([token_len] * prompt_token.shape[0], device=token.device)

        # Handle case 3: Both have different batch sizes and neither is 1 (error case)
        elif prompt_token.shape[0] != token.shape[0]:
            # This is a true mismatch that we can't handle
            logger.warning(f"Batch dimension mismatch: prompt_token.shape={prompt_token.shape}, token.shape={token.shape}")
            raise ValueError(f"Incompatible batch dimensions: prompt_token batch_size={prompt_token.shape[0]}, token batch_size={token.shape[0]}. "
                           f"One tensor must have batch_size=1 for broadcasting, or both must have the same batch_size.")

        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0, max=self.input_embedding.num_embeddings-1)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        if finalize is False:
            h = h[:, :-self.pre_lookahead_len * self.token_mel_ratio]

        # Calculate mel_len1 and mel_len2 based on the actual token proportions
        # prompt_token_len and token_len contain the actual sequence lengths
        total_tokens = prompt_token_len + token_len
        if isinstance(prompt_token_len, torch.Tensor):
            prompt_ratio = prompt_token_len.float() / total_tokens.float()
        else:
            prompt_ratio = float(prompt_token_len) / float(total_tokens)

        # mel_len1 should be proportional to the prompt tokens in the encoded sequence
        mel_len1 = int((prompt_ratio * h.shape[1]).item() if isinstance(prompt_ratio, torch.Tensor) else prompt_ratio * h.shape[1])
        mel_len2 = h.shape[1] - mel_len1
        h = self.encoder_proj(h)

        # get conditions - handle batch processing
        conds = torch.zeros([batch_size, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)

        # Handle prompt_feat batch dimension and resize to match mel_len1
        if prompt_feat.shape[0] != batch_size:
            if prompt_feat.shape[0] == 1:
                # Expand prompt_feat to match batch_size
                prompt_feat = prompt_feat.expand(batch_size, -1, -1)
            else:
                # Mismatch we can't handle with expand - this is an error
                raise ValueError(f"Incompatible prompt_feat batch dimension: prompt_feat.shape[0]={prompt_feat.shape[0]}, "
                               f"batch_size={batch_size}. For expansion, prompt_feat must have batch_size=1.")

        # Adjust prompt_feat to match mel_len1 if necessary
        if prompt_feat.shape[1] != mel_len1:
            if prompt_feat.shape[1] > mel_len1:
                # Truncate prompt_feat to mel_len1
                prompt_feat = prompt_feat[:, :mel_len1, :]
            else:
                # Pad prompt_feat to mel_len1 (repeat last frame)
                padding_needed = mel_len1 - prompt_feat.shape[1]
                last_frame = prompt_feat[:, -1:, :].expand(-1, padding_needed, -1)
                prompt_feat = torch.cat([prompt_feat, last_frame], dim=1)

        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2] * batch_size))).to(h)
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), None  # NOTE jrm: why are they returning None here?
