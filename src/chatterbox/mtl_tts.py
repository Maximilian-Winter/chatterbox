from dataclasses import dataclass
from pathlib import Path
import os

import librosa
import torch
import perth
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download

from .models.t3 import T3
from .models.t3.modules.t3_config import T3Config
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import MTLTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/chatterbox"

# Supported languages for the multilingual model
SUPPORTED_LANGUAGES = {
  "ar": "Arabic",
  "da": "Danish",
  "de": "German",
  "el": "Greek",
  "en": "English",
  "es": "Spanish",
  "fi": "Finnish",
  "fr": "French",
  "he": "Hebrew",
  "hi": "Hindi",
  "it": "Italian",
  "ja": "Japanese",
  "ko": "Korean",
  "ms": "Malay",
  "nl": "Dutch",
  "no": "Norwegian",
  "pl": "Polish",
  "pt": "Portuguese",
  "ru": "Russian",
  "sv": "Swedish",
  "sw": "Swahili",
  "tr": "Turkish",
  "zh": "Chinese",
}


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
    sentence_enders = {".", "!", "?", "-", ",","、","，","。","？","！"}
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
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxMultilingualTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: MTLTokenizer,
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
    def get_supported_languages(cls):
        """Return dictionary of supported language codes and names."""
        return SUPPORTED_LANGUAGES.copy()

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(ckpt_dir)

        ve = VoiceEncoder()
        ve.load_state_dict(
            torch.load(ckpt_dir / "ve.pt", weights_only=True)
        )
        ve.to(device).eval()

        t3 = T3(T3Config.multilingual())
        t3_state = load_safetensors(ckpt_dir / "t3_23lang.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            torch.load(ckpt_dir / "s3gen.pt", weights_only=True)
        )
        s3gen.to(device).eval()

        tokenizer = MTLTokenizer(
            str(ckpt_dir / "mtl_tokenizer.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device: torch.device) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main", 
                allow_patterns=["ve.pt", "t3_23lang.safetensors", "s3gen.pt", "mtl_tokenizer.json", "conds.pt", "Cangjie5_TC.json"],
                token=os.getenv("HF_TOKEN"),
            )
        )
        return cls.from_local(ckpt_dir, device)
    
    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        t3_cond_prompt_tokens = None
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
        language_id,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=2.0,
        min_p=0.05,
        top_p=1.0,
    ):
        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )
        
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text, language_id=language_id.lower() if language_id else None).to(self.device)
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
        language_id,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=2.0,
        min_p=0.05,
        top_p=1.0,
    ):
        """
        Generate speech for multiple text inputs with the same voice and parameters.
        Uses true batch processing for improved performance.

        Args:
            texts (list): List of text strings to convert to speech
            language_id (str): Language code for all texts (e.g., 'en', 'fr', 'es')
            audio_prompt_path (str, optional): Path to audio file for voice conditioning
            exaggeration (float): Emotion exaggeration factor
            cfg_weight (float): Classifier-free guidance weight
            temperature (float): Sampling temperature
            repetition_penalty (float): Repetition penalty for text generation
            min_p (float): Minimum probability threshold for token sampling
            top_p (float): Top-p (nucleus) sampling parameter

        Returns:
            list: List of torch tensors containing the generated audio waveforms
        """
        if not isinstance(texts, list):
            raise ValueError("texts must be a list of strings")

        if len(texts) == 0:
            return []

        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        # Prepare conditionals once for the entire batch
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Batch tokenize all texts
        normalized_texts = [punc_norm(text) for text in texts]
        all_text_tokens = []

        for text in normalized_texts:
            text_tokens = self.tokenizer.text_to_tokens(text, language_id=language_id.lower() if language_id else None).to(self.device)
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

            sot = self.t3.hp.start_text_token
            eot = self.t3.hp.stop_text_token
            text_tokens = F.pad(text_tokens, (1, 0), value=sot)
            text_tokens = F.pad(text_tokens, (0, 1), value=eot)
            all_text_tokens.append(text_tokens)

        # Pad to same length for batch processing
        max_len = max(tokens.size(1) for tokens in all_text_tokens)
        padded_tokens = []
        for tokens in all_text_tokens:
            if tokens.size(1) < max_len:
                pad_size = max_len - tokens.size(1)
                tokens = F.pad(tokens, (0, pad_size), value=self.t3.hp.stop_text_token)
            padded_tokens.append(tokens)

        # Stack into batch tensor
        batch_text_tokens = torch.cat(padded_tokens, dim=0)  # (batch_size * cfg_multiplier, max_len)

        # Expand conditioning for batch size (create contiguous tensors)
        # Handle emotion_adv expansion based on its actual dimensions
        emotion_adv_expanded = self.conds.t3.emotion_adv
        if emotion_adv_expanded.dim() == 2:
            # [orig_batch, features] -> [new_batch, features]
            emotion_adv_expanded = emotion_adv_expanded.expand(len(texts), -1)
        elif emotion_adv_expanded.dim() == 3:
            # [orig_batch, seq, features] -> [new_batch, seq, features]
            emotion_adv_expanded = emotion_adv_expanded.expand(len(texts), -1, -1)
        # else: keep as-is for other dimensions

        batch_conds = T3Cond(
            speaker_emb=self.conds.t3.speaker_emb.expand(len(texts), -1).contiguous(),
            cond_prompt_speech_tokens=self.conds.t3.cond_prompt_speech_tokens.expand(len(texts), -1).contiguous() if self.conds.t3.cond_prompt_speech_tokens is not None else None,
            emotion_adv=emotion_adv_expanded.contiguous(),
        ).to(device=self.device)

        results = []

        # Use true batch inference for T3 model
        batch_t3_conds = []
        for i in range(len(texts)):
            individual_conds = T3Cond(
                speaker_emb=batch_conds.speaker_emb[i:i+1],
                cond_prompt_speech_tokens=batch_conds.cond_prompt_speech_tokens[i:i+1] if batch_conds.cond_prompt_speech_tokens is not None else None,
                emotion_adv=batch_conds.emotion_adv[i:i+1],
            ).to(device=self.device)
            batch_t3_conds.append(individual_conds)

        with torch.inference_mode():
            # True batch T3 inference - process all sequences in parallel
            all_speech_tokens = self.t3.batch_inference(
                batch_text_tokens=padded_tokens,
                batch_t3_conds=batch_t3_conds,
                max_new_tokens=1000,  # TODO: use the value in config
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )

            # Post-process speech tokens and run S3Gen inference
            for i, speech_tokens in enumerate(all_speech_tokens):
                # Skip empty sequences
                if speech_tokens.numel() == 0:
                    print(f"Warning: Sequence {i} generated no tokens, creating silent audio")
                    # Create a short silent audio instead
                    silent_duration = 1.0  # 1 second
                    silent_wav = torch.zeros(1, int(silent_duration * self.sr))
                    results.append(silent_wav)
                    continue

                # Apply post-processing - ensure proper shape for drop_invalid_tokens
                if speech_tokens.dim() == 1:
                    speech_tokens = speech_tokens.unsqueeze(0)  # Add batch dim

                # Remove stop token (6562) if it exists - it's not a valid speech token
                # Find where stop token appears and truncate
                if (speech_tokens == self.t3.hp.stop_speech_token).any():
                    stop_idx = (speech_tokens == self.t3.hp.stop_speech_token).nonzero(as_tuple=True)
                    if len(stop_idx[0]) > 0 and len(stop_idx[1]) > 0:
                        first_stop = stop_idx[1][0].item()
                        speech_tokens = speech_tokens[:, :first_stop]
                        print(f"Sequence {i}: Truncated at stop token, keeping {first_stop} tokens")

                speech_tokens = drop_invalid_tokens(speech_tokens)

                # Skip if all tokens were dropped
                if speech_tokens.numel() == 0:
                    print(f"Warning: Sequence {i} had all tokens dropped, creating silent audio")
                    silent_duration = 1.0
                    silent_wav = torch.zeros(1, int(silent_duration * self.sr))
                    results.append(silent_wav)
                    continue

                # Ensure proper shape for S3Gen (expects batch dimension)
                if speech_tokens.dim() == 1:
                    speech_tokens = speech_tokens.unsqueeze(0)
                speech_tokens = speech_tokens.to(self.device)

                # S3Gen inference (still individual due to constraints)
                wav, _ = self.s3gen.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=self.conds.gen,
                )
                wav = wav.squeeze(0).detach().cpu().numpy()
                watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
                results.append(torch.from_numpy(watermarked_wav).unsqueeze(0))

        return results
