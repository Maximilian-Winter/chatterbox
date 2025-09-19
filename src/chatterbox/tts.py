from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import concurrent.futures
import warnings

import librosa
import torch
import perth
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond

REPO_ID = "ResembleAI/chatterbox"


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        (""", "\""),
        (""", "\""),
        ("'", "'"),
        ("'", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
            self,
            t3: T3,
            s3gen: S3Gen,
            ve: VoiceEncoder,
            tokenizer: EnTokenizer,
            device: str,
            conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(ckpt_dir / "ve.safetensors")
        )
        ve.to(device).eval()

        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(
            str(ckpt_dir / "tokenizer.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTTS':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print(
                    "MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def _validate_batch_inputs(self, texts: List[str], audio_prompt_paths: Optional[List[str]] = None) -> None:
        if not isinstance(texts, list) or len(texts) == 0:
            raise ValueError("texts must be a non-empty list of strings")

        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All items in texts must be strings")

        if audio_prompt_paths is not None:
            if not isinstance(audio_prompt_paths, list):
                raise ValueError("audio_prompt_paths must be a list or None")
            if len(audio_prompt_paths) != len(texts):
                raise ValueError("audio_prompt_paths must have the same length as texts")

    def _batch_tokenize_texts(self, texts: List[str]) -> torch.Tensor:
        normalized_texts = [punc_norm(text) for text in texts]
        token_lists = [self.tokenizer.text_to_tokens(text) for text in normalized_texts]

        # Extract the actual sequence length from each tokenized output
        # tokenizer returns [1, seq_len], so we need to squeeze and get the actual length
        max_len = max(tokens.shape[1] for tokens in token_lists)
        batch_tokens = torch.zeros((len(texts), max_len), dtype=torch.long, device=self.device)

        for i, tokens in enumerate(token_lists):
            # tokens has shape [1, seq_len], squeeze to get [seq_len]
            tokens_1d = tokens.squeeze(0).to(self.device)
            batch_tokens[i, :tokens_1d.shape[0]] = tokens_1d

        return batch_tokens

    def _estimate_memory_usage(self, batch_size: int, max_text_len: int) -> float:
        text_tokens_mem = batch_size * max_text_len * 4  # int32
        model_forward_mem = batch_size * 1024 * max_text_len * 4  # float32, estimated model dim
        speech_tokens_mem = batch_size * 1000 * 4  # approximate speech sequence length

        total_mb = (text_tokens_mem + model_forward_mem + speech_tokens_mem) / (1024 * 1024) * 2
        return total_mb

    def _adaptive_batch_size(self, texts: List[str], max_batch_size: int) -> int:
        normalized_texts = [punc_norm(text) for text in texts]
        max_text_len = max(len(self.tokenizer.text_to_tokens(text)) for text in normalized_texts)

        for batch_size in range(max_batch_size, 0, -1):
            estimated_mem = self._estimate_memory_usage(batch_size, max_text_len)

            if torch.cuda.is_available() and self.device != 'cpu':
                try:
                    free_mem = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)  # GB
                    if estimated_mem < free_mem * 0.8 * 1024:  # Use 80% of available memory
                        return batch_size
                except:
                    pass

            if estimated_mem < 2048:  # 2GB limit for non-CUDA or fallback
                return batch_size

        return 1

    def prepare_conditionals_batch(
            self,
            wav_fpaths: List[str],
            exaggerations: Union[float, List[float]] = 0.5
    ) -> List[Conditionals]:
        if isinstance(exaggerations, (int, float)):
            exaggerations = [float(exaggerations)] * len(wav_fpaths)
        elif len(exaggerations) != len(wav_fpaths):
            raise ValueError("exaggerations must be a scalar or list with same length as wav_fpaths")

        def process_single_conditional(args):
            wav_fpath, exaggeration = args
            s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
            ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

            s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
            s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

            t3_cond_prompt_tokens = None
            if plen := self.t3.hp.speech_cond_prompt_len:
                s3_tokzr = self.s3gen.tokenizer
                t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
                t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

            ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
            ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

            t3_cond = T3Cond(
                speaker_emb=ve_embed,
                cond_prompt_speech_tokens=t3_cond_prompt_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

            return Conditionals(t3_cond, s3gen_ref_dict)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            conditionals = list(executor.map(process_single_conditional, zip(wav_fpaths, exaggerations)))

        return conditionals

    def generate_batch(
            self,
            texts: List[str],
            audio_prompt_paths: Optional[List[str]] = None,
            exaggerations: Union[float, List[float]] = 0.5,
            cfg_weights: Union[float, List[float]] = 0.5,
            temperatures: Union[float, List[float]] = 0.8,
            repetition_penalties: Union[float, List[float]] = 2.0,
            min_ps: Union[float, List[float]] = 0.05,
            top_ps: Union[float, List[float]] = 1.0,
            max_batch_size: int = 8,
            return_intermediates: bool = False,
    ) -> List[torch.Tensor]:
        self._validate_batch_inputs(texts, audio_prompt_paths)

        batch_size = len(texts)

        def ensure_list(param, name):
            if isinstance(param, (int, float)):
                return [float(param)] * batch_size
            elif len(param) != batch_size:
                raise ValueError(f"{name} must be a scalar or list with same length as texts")
            return param

        exaggerations = ensure_list(exaggerations, "exaggerations")
        cfg_weights = ensure_list(cfg_weights, "cfg_weights")
        temperatures = ensure_list(temperatures, "temperatures")
        repetition_penalties = ensure_list(repetition_penalties, "repetition_penalties")
        min_ps = ensure_list(min_ps, "min_ps")
        top_ps = ensure_list(top_ps, "top_ps")

        if audio_prompt_paths:
            batch_conditionals = self.prepare_conditionals_batch(audio_prompt_paths, exaggerations)
        else:
            if self.conds is None:
                raise ValueError("Please prepare_conditionals first or specify audio_prompt_paths")
            batch_conditionals = [self.conds] * batch_size

        optimal_batch_size = min(max_batch_size, self._adaptive_batch_size(texts, max_batch_size))

        results = []
        for i in range(0, batch_size, optimal_batch_size):
            end_idx = min(i + optimal_batch_size, batch_size)
            batch_texts = texts[i:end_idx]
            batch_conds = batch_conditionals[i:end_idx]
            batch_cfg = cfg_weights[i:end_idx]
            batch_temp = temperatures[i:end_idx]
            batch_rep_pen = repetition_penalties[i:end_idx]
            batch_min_p = min_ps[i:end_idx]
            batch_top_p = top_ps[i:end_idx]

            sub_batch_results = self._process_sub_batch(
                batch_texts, batch_conds, batch_cfg, batch_temp,
                batch_rep_pen, batch_min_p, batch_top_p
            )
            results.extend(sub_batch_results)

        return results

    def _process_sub_batch(
            self,
            texts: List[str],
            conditionals: List[Conditionals],
            cfg_weights: List[float],
            temperatures: List[float],
            repetition_penalties: List[float],
            min_ps: List[float],
            top_ps: List[float]
    ) -> List[torch.Tensor]:
        batch_size = len(texts)

        try:
            # Prepare batch text tokens
            batch_text_tokens = self._batch_tokenize_texts(texts)

            # Prepare individual text token sequences for T3
            t3_text_tokens_batch = []
            t3_cond_batch = []

            for i in range(batch_size):
                text_tokens = batch_text_tokens[i]
                text_tokens = text_tokens[text_tokens != 0]  # Remove padding

                if cfg_weights[i] > 0.0:
                    text_tokens = torch.cat([text_tokens.unsqueeze(0), text_tokens.unsqueeze(0)], dim=0)
                else:
                    text_tokens = text_tokens.unsqueeze(0)

                sot = self.t3.hp.start_text_token
                eot = self.t3.hp.stop_text_token
                text_tokens = F.pad(text_tokens, (1, 0), value=sot)
                text_tokens = F.pad(text_tokens, (0, 1), value=eot)

                t3_text_tokens_batch.append(text_tokens)
                t3_cond_batch.append(conditionals[i].t3)

            # Batch T3 inference
            with torch.inference_mode():
                speech_tokens_batch = self.t3.inference_batch(
                    t3_cond_batch=t3_cond_batch,
                    text_tokens_batch=t3_text_tokens_batch,
                    max_new_tokens=1000,
                    temperatures=temperatures,
                    cfg_weights=cfg_weights,
                    repetition_penalties=repetition_penalties,
                    min_ps=min_ps,
                    top_ps=top_ps,
                    max_batch_size=min(batch_size, 4),
                    use_parallel_generation=batch_size > 1,
                )

                # Process speech tokens and prepare for S3Gen
                processed_speech_tokens = []
                s3gen_ref_dicts = []

                for i, speech_tokens in enumerate(speech_tokens_batch):
                    try:
                        # For CFG, take the first sequence
                        if cfg_weights[i] > 0.0 and speech_tokens.dim() > 1 and speech_tokens.shape[0] > 1:
                            speech_tokens = speech_tokens[0]

                        # Ensure speech_tokens is 1D
                        if speech_tokens.dim() > 1:
                            speech_tokens = speech_tokens.squeeze(0)

                        # Apply drop_invalid_tokens
                        # Note: drop_invalid_tokens expects a 1D tensor and returns processed tokens
                        speech_tokens_processed = drop_invalid_tokens(speech_tokens)

                        # Handle the return value from drop_invalid_tokens
                        if isinstance(speech_tokens_processed, list):
                            # If it returns a list, take the first element
                            speech_tokens = speech_tokens_processed[0] if speech_tokens_processed else speech_tokens
                        elif isinstance(speech_tokens_processed, tuple):
                            # If it returns a tuple, take the first element
                            speech_tokens = speech_tokens_processed[0] if speech_tokens_processed else speech_tokens
                        elif torch.is_tensor(speech_tokens_processed):
                            # If it returns a tensor, use it directly
                            speech_tokens = speech_tokens_processed
                        else:
                            # Fallback to original if unexpected return type
                            warnings.warn(
                                f"Unexpected return type from drop_invalid_tokens: {type(speech_tokens_processed)}")

                        # Ensure tensor and apply token range filter
                        if not torch.is_tensor(speech_tokens):
                            speech_tokens = torch.tensor(speech_tokens, device=self.device)

                        # Filter tokens to valid range (less than vocabulary size)
                        # Note: Only apply this filter if we have valid tokens
                        if speech_tokens.numel() > 0:
                            valid_mask = speech_tokens < 6561
                            speech_tokens = speech_tokens[valid_mask]

                        # Ensure we have at least some tokens
                        if speech_tokens.numel() == 0:
                            warnings.warn(f"No valid speech tokens for item {i}, using fallback")
                            speech_tokens = torch.tensor([self.t3.hp.start_speech_token, self.t3.hp.stop_speech_token],
                                                         device=self.device)

                        speech_tokens = speech_tokens.to(self.device)
                        processed_speech_tokens.append(speech_tokens)
                        s3gen_ref_dicts.append(conditionals[i].gen)

                    except Exception as e:
                        warnings.warn(f"Failed to process speech tokens for item {i}: {str(e)}")
                        # Create fallback speech tokens
                        fallback_tokens = torch.tensor([self.t3.hp.start_speech_token, self.t3.hp.stop_speech_token],
                                                       device=self.device)
                        processed_speech_tokens.append(fallback_tokens)
                        s3gen_ref_dicts.append(conditionals[i].gen)

                # Batch S3Gen inference
                s3gen_results = self.s3gen.inference_batch(
                    speech_tokens_batch=processed_speech_tokens,
                    ref_dicts_batch=s3gen_ref_dicts,
                    finalize=True,
                    max_batch_size=min(batch_size, 4),
                )

                # Process final results
                results = []
                for i, result in enumerate(s3gen_results):
                    try:
                        # Handle different return formats from S3Gen
                        if isinstance(result, tuple):
                            wav, _ = result
                        else:
                            wav = result

                        wav = wav.squeeze(0).detach().cpu().numpy()
                        watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
                        results.append(torch.from_numpy(watermarked_wav).unsqueeze(0))
                    except Exception as e:
                        warnings.warn(f"Failed to process final output for item {i}: {str(e)}")
                        results.append(torch.zeros(1, 1000))

                return results

        except Exception as e:
            warnings.warn(f"Batch processing failed, falling back to individual processing: {str(e)}")
            # Fallback to individual processing
            return self._process_sub_batch_fallback(texts, conditionals, cfg_weights, temperatures,
                                                    repetition_penalties, min_ps, top_ps)

    def _process_sub_batch_fallback(
            self,
            texts: List[str],
            conditionals: List[Conditionals],
            cfg_weights: List[float],
            temperatures: List[float],
            repetition_penalties: List[float],
            min_ps: List[float],
            top_ps: List[float]
    ) -> List[torch.Tensor]:
        """Fallback to individual processing when batch processing fails."""
        batch_size = len(texts)
        batch_text_tokens = self._batch_tokenize_texts(texts)

        results = []
        for i in range(batch_size):
            try:
                text_tokens = batch_text_tokens[i:i + 1]
                text_tokens = text_tokens[text_tokens != 0]  # Remove padding

                if cfg_weights[i] > 0.0:
                    text_tokens = torch.cat([text_tokens.unsqueeze(0), text_tokens.unsqueeze(0)], dim=0)
                else:
                    text_tokens = text_tokens.unsqueeze(0)

                sot = self.t3.hp.start_text_token
                eot = self.t3.hp.stop_text_token
                text_tokens = F.pad(text_tokens, (1, 0), value=sot)
                text_tokens = F.pad(text_tokens, (0, 1), value=eot)

                with torch.inference_mode():
                    speech_tokens = self.t3.inference(
                        t3_cond=conditionals[i].t3,
                        text_tokens=text_tokens,
                        max_new_tokens=1000,
                        temperature=temperatures[i],
                        cfg_weight=cfg_weights[i],
                        repetition_penalty=repetition_penalties[i],
                        min_p=min_ps[i],
                        top_p=top_ps[i],
                    )

                    speech_tokens = speech_tokens[0] if cfg_weights[i] > 0.0 else speech_tokens
                    speech_tokens = drop_invalid_tokens(speech_tokens)
                    speech_tokens = speech_tokens[speech_tokens < 6561]
                    speech_tokens = speech_tokens.to(self.device)

                    wav, _ = self.s3gen.inference(
                        speech_tokens=speech_tokens,
                        ref_dict=conditionals[i].gen,
                    )
                    wav = wav.squeeze(0).detach().cpu().numpy()
                    watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
                    results.append(torch.from_numpy(watermarked_wav).unsqueeze(0))

            except Exception as e:
                warnings.warn(f"Failed to process text {i}: {texts[i][:50]}... Error: {str(e)}")
                results.append(torch.zeros(1, 1000))  # Fallback empty tensor

        return results

    def generate(
            self,
            text,
            repetition_penalty=1.2,
            min_p=0.05,
            top_p=1.0,
            audio_prompt_path=None,
            exaggeration=0.5,
            cfg_weight=0.5,
            temperature=0.8,
    ):
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,  # TODO: use the value in config
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)

            speech_tokens = speech_tokens[speech_tokens < 6561]

            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)