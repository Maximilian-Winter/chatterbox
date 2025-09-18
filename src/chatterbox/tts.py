from dataclasses import dataclass
from pathlib import Path

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
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
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
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
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

    def generate_batch(
        self,
        texts,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        audio_prompt_paths=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        max_batch_size=8,
    ):
        """
        Generate audio for multiple texts in batch.

        Args:
            texts: List of text strings to synthesize
            audio_prompt_paths: List of audio prompt paths (one per text) or single path for all,
                               or None to use built-in voice
            max_batch_size: Maximum number of texts to process in a single batch
            **kwargs: Other generation parameters

        Returns:
            List of audio tensors, one per input text
        """
        if not isinstance(texts, list):
            texts = [texts]

        if len(texts) == 0:
            return []

        # Handle audio prompt paths
        if audio_prompt_paths is None:
            audio_prompt_paths = [None] * len(texts)
        elif isinstance(audio_prompt_paths, str):
            # Single path for all texts
            audio_prompt_paths = [audio_prompt_paths] * len(texts)
        elif len(audio_prompt_paths) == 1 and len(texts) > 1:
            # Single path for all texts
            audio_prompt_paths = audio_prompt_paths * len(texts)
        else:
            assert len(audio_prompt_paths) == len(texts), "audio_prompt_paths must match texts length"

        # Process in chunks if batch too large
        all_results = []
        for i in range(0, len(texts), max_batch_size):
            batch_texts = texts[i:i + max_batch_size]
            batch_prompts = audio_prompt_paths[i:i + max_batch_size]

            batch_results = self._generate_batch_chunk(
                batch_texts,
                batch_prompts,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )
            all_results.extend(batch_results)

        return all_results

    def _generate_batch_chunk(
        self,
        texts,
        audio_prompt_paths,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
    ):
        """Generate audio for a chunk of texts that fit in memory."""
        batch_size = len(texts)

        # Prepare conditionals for each text
        batch_conds = []
        for i, audio_prompt_path in enumerate(audio_prompt_paths):
            if audio_prompt_path:
                self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
            else:
                assert self.conds is not None, f"Please `prepare_conditionals` first or specify `audio_prompt_path` for text {i}"

            # Update exaggeration if needed
            if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
                _cond = self.conds.t3
                self.conds.t3 = T3Cond(
                    speaker_emb=_cond.speaker_emb,
                    cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                    emotion_adv=exaggeration * torch.ones(1, 1, 1),
                ).to(device=self.device)

            batch_conds.append((self.conds.t3, self.conds.gen))

        # Normalize and tokenize all texts
        normalized_texts = [punc_norm(text) for text in texts]
        text_tokens_list = [self.tokenizer.text_to_tokens(text).to(self.device) for text in normalized_texts]

        # Pad text tokens to same length
        max_text_len = max(tokens.size(1) for tokens in text_tokens_list)
        padded_text_tokens = []
        text_lens = []

        for tokens in text_tokens_list:
            text_lens.append(tokens.size(1))
            if tokens.size(1) < max_text_len:
                pad_size = max_text_len - tokens.size(1)
                tokens = F.pad(tokens, (0, pad_size), value=0)
            padded_text_tokens.append(tokens)

        # Stack into batch
        batch_text_tokens = torch.cat(padded_text_tokens, dim=0)  # (B, T)

        if cfg_weight > 0.0:
            batch_text_tokens = torch.cat([batch_text_tokens, batch_text_tokens], dim=0)  # CFG

        # Add start/stop tokens
        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        batch_text_tokens = F.pad(batch_text_tokens, (1, 0), value=sot)
        batch_text_tokens = F.pad(batch_text_tokens, (0, 1), value=eot)

        # Generate speech tokens for the batch
        all_speech_tokens = []
        for i in range(batch_size):
            # Get conditionals for this sample
            t3_cond, gen_cond = batch_conds[i]

            # Extract text tokens for this sample
            if cfg_weight > 0.0:
                sample_text_tokens = torch.stack([batch_text_tokens[i], batch_text_tokens[i + batch_size]])
            else:
                sample_text_tokens = batch_text_tokens[i:i+1]

            with torch.inference_mode():
                speech_tokens = self.t3.inference(
                    t3_cond=t3_cond,
                    text_tokens=sample_text_tokens,
                    max_new_tokens=1000,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    repetition_penalty=repetition_penalty,
                    min_p=min_p,
                    top_p=top_p,
                )
                speech_tokens = speech_tokens[0]  # Extract conditional batch
                speech_tokens = drop_invalid_tokens(speech_tokens)
                speech_tokens = speech_tokens[speech_tokens < 6561]
                all_speech_tokens.append((speech_tokens.to(self.device), gen_cond))

        # Generate waveforms
        results = []
        for speech_tokens, gen_cond in all_speech_tokens:
            with torch.inference_mode():
                wav, _ = self.s3gen.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=gen_cond,
                )
                wav = wav.squeeze(0).detach().cpu().numpy()
                watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
                results.append(torch.from_numpy(watermarked_wav).unsqueeze(0))

        return results