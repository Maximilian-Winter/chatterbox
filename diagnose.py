"""
Diagnostic script for ChatterboxTTS batch processing
Traces data flow through the pipeline to identify where audio generation fails
"""

from chatterbox import ChatterboxMultilingualTTS
import torchaudio as ta
import torch
import numpy as np


class DiagnosticTTS(ChatterboxMultilingualTTS):
    """Wrapper class to add diagnostic logging to batch processing"""

    def _process_sub_batch(self, texts, language_ids, conditionals, cfg_weights, temperatures,
                           repetition_penalties, min_ps, top_ps):
        batch_size = len(texts)
        print(f"\n=== Processing sub-batch of {batch_size} items ===")

        try:
            # Tokenize texts
            batch_text_tokens = self._batch_tokenize_texts(texts)
            print(f"Text tokens shape: {batch_text_tokens.shape}")
            print(f"Text tokens sample: {batch_text_tokens[0][:20].tolist()}")

            # Prepare T3 inputs
            t3_text_tokens_batch = []
            t3_cond_batch = []

            for i in range(batch_size):
                text_tokens = batch_text_tokens[i]
                text_tokens = text_tokens[text_tokens != 0]

                if cfg_weights[i] > 0.0:
                    text_tokens = torch.cat([text_tokens.unsqueeze(0), text_tokens.unsqueeze(0)], dim=0)
                else:
                    text_tokens = text_tokens.unsqueeze(0)

                sot = self.t3.hp.start_text_token
                eot = self.t3.hp.stop_text_token
                text_tokens = torch.nn.functional.pad(text_tokens, (1, 0), value=sot)
                text_tokens = torch.nn.functional.pad(text_tokens, (0, 1), value=eot)

                print(f"T3 input {i} shape: {text_tokens.shape}")
                t3_text_tokens_batch.append(text_tokens)
                t3_cond_batch.append(conditionals[i].t3)

            # T3 inference
            print("\nCalling T3 inference_batch...")
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

                print(f"T3 output: {len(speech_tokens_batch)} sequences")
                for i, tokens in enumerate(speech_tokens_batch):
                    print(f"  Speech tokens {i}: shape={tokens.shape if hasattr(tokens, 'shape') else 'N/A'}, "
                          f"type={type(tokens)}, len={len(tokens) if hasattr(tokens, '__len__') else 'N/A'}")
                    if hasattr(tokens, 'shape') and tokens.numel() > 0:
                        print(f"    Sample values: {tokens.flatten()[:10].tolist()}")

                # Process speech tokens
                processed_speech_tokens = []
                s3gen_ref_dicts = []

                for i, speech_tokens in enumerate(speech_tokens_batch):
                    try:
                        original_shape = speech_tokens.shape if hasattr(speech_tokens, 'shape') else None

                        if cfg_weights[i] > 0.0 and speech_tokens.dim() > 1 and speech_tokens.shape[0] > 1:
                            speech_tokens = speech_tokens[0]
                            print(f"  CFG adjustment for {i}: {original_shape} -> {speech_tokens.shape}")

                        # Import drop_invalid_tokens function
                        from chatterbox.models.s3tokenizer import drop_invalid_tokens
                        speech_tokens = drop_invalid_tokens(speech_tokens)

                        if isinstance(speech_tokens, list):
                            speech_tokens = speech_tokens[0] if speech_tokens else torch.tensor([], device=self.device)
                            print(f"  List handling for {i}: converted to tensor with shape {speech_tokens.shape}")

                        speech_tokens = speech_tokens[speech_tokens < 6561]
                        speech_tokens = speech_tokens.to(self.device)

                        print(f"  Processed tokens {i}: shape={speech_tokens.shape}, "
                              f"min={speech_tokens.min().item() if speech_tokens.numel() > 0 else 'empty'}, "
                              f"max={speech_tokens.max().item() if speech_tokens.numel() > 0 else 'empty'}")

                        processed_speech_tokens.append(speech_tokens)
                        s3gen_ref_dicts.append(conditionals[i].gen)

                    except Exception as e:
                        print(f"  ERROR processing speech tokens {i}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        fallback_tokens = torch.tensor([self.t3.hp.start_speech_token, self.t3.hp.stop_speech_token],
                                                       device=self.device)
                        processed_speech_tokens.append(fallback_tokens)
                        s3gen_ref_dicts.append(conditionals[i].gen)

                # S3Gen inference
                print("\nCalling S3Gen inference_batch...")
                print(f"Input: {len(processed_speech_tokens)} token sequences")
                for i, tokens in enumerate(processed_speech_tokens):
                    print(f"  Sequence {i}: shape={tokens.shape}, len={tokens.numel()}")

                s3gen_results = self.s3gen.inference_batch(
                    speech_tokens_batch=processed_speech_tokens,
                    ref_dicts_batch=s3gen_ref_dicts,
                    finalize=True,
                    max_batch_size=min(batch_size, 4),
                )

                print(f"S3Gen output: {len(s3gen_results)} results")

                # Process results
                results = []
                for i, result in enumerate(s3gen_results):
                    try:
                        if isinstance(result, tuple):
                            wav, _ = result
                        else:
                            wav = result

                        print(f"  Result {i}: type={type(wav)}, shape={wav.shape if hasattr(wav, 'shape') else 'N/A'}")

                        if hasattr(wav, 'shape'):
                            print(f"    Stats: min={wav.min().item():.4f}, max={wav.max().item():.4f}, "
                                  f"mean={wav.mean().item():.4f}, std={wav.std().item():.4f}")

                        wav = wav.squeeze(0).detach().cpu().numpy()
                        print(f"    After processing: shape={wav.shape}, dtype={wav.dtype}")

                        watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
                        results.append(torch.from_numpy(watermarked_wav).unsqueeze(0))

                    except Exception as e:
                        print(f"  ERROR processing result {i}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        results.append(torch.zeros(1, 1000))

                return results

        except Exception as e:
            print(f"\nBatch processing failed: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Falling back to individual processing...")
            return self._process_sub_batch_fallback(texts, language_ids,conditionals, cfg_weights,
                                                    temperatures, repetition_penalties, min_ps, top_ps)


def main():
    print("Initializing ChatterboxMultilingualTTS with diagnostics...")

    # Monkey-patch the TTS class
    import chatterbox
    original_tts = chatterbox.ChatterboxMultilingualTTS
    chatterbox.ChatterboxMultilingualTTS = DiagnosticTTS

    tts = DiagnosticTTS.from_pretrained("cuda")

    texts = ["Hello world!", "How are you?", "Batch processing test."]
    audio_paths = ["voice.wav", "voice.wav", "voice.wav"]

    print("\n" + "=" * 50)
    print("Starting batch generation...")
    print("=" * 50)

    outputs = tts.generate_batch(
        texts=texts,
        language_ids=["en"] * len(texts),
        audio_prompt_paths=audio_paths,
        max_batch_size=4
    )

    print("\n" + "=" * 50)
    print("Batch generation complete")
    print("=" * 50)

    for idx, audio in enumerate(outputs):
        print(f"\nOutput {idx}:")
        print(f"  Shape: {audio.shape}")
        print(f"  Min: {audio.min().item():.4f}, Max: {audio.max().item():.4f}")
        print(f"  Mean: {audio.mean().item():.4f}, Std: {audio.std().item():.4f}")
        print(f"  Non-zero elements: {(audio != 0).sum().item()} / {audio.numel()}")

        ta.save(f"test-diag-{idx}.wav", audio, tts.sr)
        print(f"  Saved to test-diag-{idx}.wav")


if __name__ == "__main__":
    main()