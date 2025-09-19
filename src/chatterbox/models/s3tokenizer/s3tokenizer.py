from typing import List, Tuple, Union, Optional
import warnings

import numpy as np
import librosa
import torch
import torch.nn.functional as F
from s3tokenizer.utils import padding
from s3tokenizer.model_v2 import (
    S3TokenizerV2,
    ModelConfig,
)


# Sampling rate of the inputs to S3TokenizerV2
S3_SR = 16_000
S3_HOP = 160  # 100 frames/sec
S3_TOKEN_HOP = 640  # 25 tokens/sec
S3_TOKEN_RATE = 25
SPEECH_VOCAB_SIZE = 6561


class S3Tokenizer(S3TokenizerV2):
    """
    s3tokenizer.S3TokenizerV2 with the following changes:
    - a more integrated `forward`
    - compute `log_mel_spectrogram` using `_mel_filters` and `window` in `register_buffers`
    """

    ignore_state_dict_missing = ("_mel_filters", "window")

    def __init__(
        self,
        name: str="speech_tokenizer_v2_25hz",
        config: ModelConfig = ModelConfig()
    ):
        super().__init__(name)

        self.n_fft = 400
        _mel_filters = librosa.filters.mel(
            sr=S3_SR,
            n_fft=self.n_fft,
            n_mels=config.n_mels
        )
        self.register_buffer(
            "_mel_filters",
            torch.FloatTensor(_mel_filters),
        )

        self.register_buffer(
            "window",
            torch.hann_window(self.n_fft),
        )

    def pad(self, wavs, sr) -> List[torch.Tensor]:
        """
        Given a list of wavs with the same `sample_rate`, pad them so that the length is multiple of 40ms (S3 runs at 25 token/sec).
        """
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            n_tokens = (wav.shape[1] / sr) * S3_TOKEN_RATE
            n_tokens = np.ceil(n_tokens)
            intended_wav_len = n_tokens * (sr / S3_TOKEN_RATE)
            intended_wav_len = int(intended_wav_len)
            wav = torch.nn.functional.pad(
                wav,
                (0, intended_wav_len - wav.shape[-1]),
                mode="constant",
                value=0
            )
            processed_wavs.append(wav)
        return processed_wavs

    def _prepare_audio(self, wavs):
        """Prepare a list of audios for s3tokenizer processing."""
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            processed_wavs.append(wav)
        return processed_wavs

    @torch.no_grad()
    def forward(
        self,
        wavs: torch.Tensor,
        accelerator: 'Accelerator'=None,
        max_len: int=None,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        NOTE: mel-spec has a hop size of 160 points (100 frame/sec).
        FIXME: this class inherits `nn.Module` but doesn't accept `torch.Tensor` and handles a list of wavs one by one, which is unexpected.

        Args
        ----
        - `wavs`: 16 kHz speech audio
        - `max_len` max length to truncate the output sequence to (25 token/sec).
        NOTE: please pad the waveform if longer sequence is needed.
        """
        processed_wavs = self._prepare_audio(wavs)
        mels, mel_lens = [], []
        for wav in processed_wavs:
            wav = wav.to(self.device)
            mel = self.log_mel_spectrogram(wav)  # [B=1, F, T]
            if max_len is not None:
                mel = mel[..., :max_len * 4]  # num_mel_frames = 4 * num_tokens
            mels.append(mel.squeeze(0))

        mels, mel_lens = padding(mels)
        if accelerator is None:
            tokenizer = self
        else:
            tokenizer = accelerator.unwrap_model(self)

        speech_tokens, speech_token_lens = tokenizer.quantize(mels, mel_lens.to(self.device))
        return (
            speech_tokens.long().detach(),
            speech_token_lens.long().detach(),
        )

    def log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        padding: int = 0,
    ):
        """
        Compute the log-Mel spectrogram of

        Parameters
        ----------
        audio: torch.Tensor, shape = (*)
            The path to audio or either a NumPy array or Tensor containing the
            audio waveform in 16 kHz

        padding: int
            Number of zero samples to pad to the right

        Returns
        -------
        torch.Tensor, shape = (128, n_frames)
            A Tensor that contains the Mel spectrogram
        """
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)

        audio = audio.to(self.device)
        if padding > 0:
            audio = F.pad(audio, (0, padding))
        stft = torch.stft(
            audio, self.n_fft, S3_HOP,
            window=self.window.to(self.device),
            return_complex=True
        )
        magnitudes = stft[..., :-1].abs()**2

        mel_spec = self._mel_filters.to(self.device) @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def _prepare_audio_batch(self, wavs: List[Union[torch.Tensor, np.ndarray]]) -> List[torch.Tensor]:
        """Prepare a batch of audio files for processing."""
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            processed_wavs.append(wav)
        return processed_wavs

    def batch_log_mel_spectrogram(
        self,
        batch_audio: List[torch.Tensor],
        padding_value: int = 0,
    ) -> List[torch.Tensor]:
        """
        Compute log-Mel spectrograms for a batch of audio tensors efficiently.

        Parameters
        ----------
        batch_audio: List[torch.Tensor]
            List of audio tensors with shape (1, audio_length) or (audio_length,)

        padding_value: int
            Value to use for padding

        Returns
        -------
        List[torch.Tensor]
            List of log-Mel spectrograms with shape (n_mels, n_frames)
        """
        if not batch_audio:
            return []

        device = self.device
        mels = []

        # Group similar-length audio files for more efficient batching
        length_groups = {}
        for i, audio in enumerate(batch_audio):
            if not torch.is_tensor(audio):
                audio = torch.from_numpy(audio)
            audio = audio.to(device)

            if padding_value > 0:
                audio = F.pad(audio, (0, padding_value))

            audio_len = audio.shape[-1]
            length_key = (audio_len // 1000) * 1000  # Group by 1000-sample chunks

            if length_key not in length_groups:
                length_groups[length_key] = []
            length_groups[length_key].append((i, audio))

        # Process each length group as a batch
        mel_results = [None] * len(batch_audio)

        for length_key, audio_group in length_groups.items():
            if len(audio_group) == 1:
                # Single audio - process normally
                idx, audio = audio_group[0]
                mel = self._compute_single_mel(audio)
                mel_results[idx] = mel
            else:
                # Multiple audios with similar length - batch process
                indices, audios = zip(*audio_group)
                max_len = max(audio.shape[-1] for audio in audios)

                # Pad to same length and stack
                padded_audios = []
                for audio in audios:
                    if audio.shape[-1] < max_len:
                        audio = F.pad(audio, (0, max_len - audio.shape[-1]))
                    padded_audios.append(audio)

                batch_tensor = torch.stack(padded_audios, dim=0)  # [B, 1, T] or [B, T]
                if batch_tensor.dim() == 2:
                    batch_tensor = batch_tensor.unsqueeze(1)  # Ensure [B, 1, T]

                # Batch STFT computation
                batch_mels = self._compute_batch_mel(batch_tensor)

                # Store results in original order
                for i, idx in enumerate(indices):
                    mel_results[idx] = batch_mels[i]

        return mel_results

    def _compute_single_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram for a single audio tensor."""
        stft = torch.stft(
            audio, self.n_fft, S3_HOP,
            window=self.window.to(self.device),
            return_complex=True
        )
        magnitudes = stft[..., :-1].abs()**2
        mel_spec = self._mel_filters.to(self.device) @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec.squeeze(0) if log_spec.dim() > 2 else log_spec

    def _compute_batch_mel(self, batch_audio: torch.Tensor) -> List[torch.Tensor]:
        """Compute mel spectrograms for a batch of audio tensors."""
        # batch_audio: [B, 1, T]
        batch_audio = batch_audio.squeeze(1)  # [B, T]

        # Vectorized STFT
        batch_stft = torch.stft(
            batch_audio, self.n_fft, S3_HOP,
            window=self.window.to(self.device),
            return_complex=True
        )  # [B, F, T, 2] or [B, F, T] (complex)

        batch_magnitudes = batch_stft[..., :-1].abs()**2  # [B, F, T]

        # Batch mel filtering
        mel_filters = self._mel_filters.to(self.device)  # [n_mels, F]
        batch_mel_spec = torch.einsum('mf,bft->bmt', mel_filters, batch_magnitudes)  # [B, n_mels, T]

        # Batch log computation
        batch_log_spec = torch.clamp(batch_mel_spec, min=1e-10).log10()
        batch_log_spec = torch.maximum(batch_log_spec, batch_log_spec.max(dim=-1, keepdim=True)[0] - 8.0)
        batch_log_spec = (batch_log_spec + 4.0) / 4.0

        # Convert back to list
        return [batch_log_spec[i] for i in range(batch_log_spec.shape[0])]

    def forward_batch(
        self,
        wavs: List[Union[torch.Tensor, np.ndarray]],
        accelerator: Optional['Accelerator'] = None,
        max_len: Optional[int] = None,
        batch_size: int = 8,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Vectorized batch processing for multiple audio files.

        Args
        ----
        - `wavs`: List of 16 kHz speech audio tensors/arrays
        - `max_len`: max length to truncate the output sequence to (25 token/sec)
        - `batch_size`: maximum number of audio files to process simultaneously

        Returns
        -------
        - speech_tokens: tokenized speech
        - speech_token_lens: lengths of tokenized sequences
        """
        if not wavs:
            return torch.empty(0, 0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        processed_wavs = self._prepare_audio_batch(wavs)

        # Process in sub-batches to manage memory
        all_speech_tokens = []
        all_speech_token_lens = []

        for i in range(0, len(processed_wavs), batch_size):
            sub_batch_wavs = processed_wavs[i:i + batch_size]

            try:
                # Batch mel spectrogram computation
                mels = self.batch_log_mel_spectrogram(sub_batch_wavs)

                # Apply max_len truncation if specified
                if max_len is not None:
                    mels = [mel[..., :max_len * 4] for mel in mels]  # num_mel_frames = 4 * num_tokens

                # Pad and prepare for quantization
                try:
                    padded_mels, mel_lens = padding(mels)
                except Exception as pad_error:
                    warnings.warn(f"Padding failed: {pad_error}, falling back to individual processing")
                    # Fallback to individual processing for this sub-batch
                    raise pad_error

                # Quantize
                if accelerator is None:
                    tokenizer = self
                else:
                    tokenizer = accelerator.unwrap_model(self)

                speech_tokens, speech_token_lens = tokenizer.quantize(padded_mels, mel_lens.to(self.device))

                all_speech_tokens.append(speech_tokens.long().detach())
                all_speech_token_lens.append(speech_token_lens.long().detach())

            except Exception as e:
                warnings.warn(f"Failed to process sub-batch {i//batch_size}: {str(e)}")
                # Fallback to single-item processing for this sub-batch
                for wav in sub_batch_wavs:
                    try:
                        single_tokens, single_lens = self.forward([wav], accelerator, max_len)
                        all_speech_tokens.append(single_tokens)
                        all_speech_token_lens.append(single_lens)
                    except Exception as e2:
                        warnings.warn(f"Failed to process single item in fallback: {str(e2)}")
                        # Create empty tensor as ultimate fallback
                        empty_tokens = torch.zeros(1, 1, dtype=torch.long)
                        empty_lens = torch.zeros(1, dtype=torch.long)
                        all_speech_tokens.append(empty_tokens)
                        all_speech_token_lens.append(empty_lens)

        # Concatenate all results
        if all_speech_tokens:
            final_tokens = torch.cat(all_speech_tokens, dim=0)
            final_lens = torch.cat(all_speech_token_lens, dim=0)
        else:
            final_tokens = torch.empty(0, 0, dtype=torch.long)
            final_lens = torch.empty(0, dtype=torch.long)

        return final_tokens, final_lens
