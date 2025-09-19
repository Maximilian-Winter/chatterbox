from chatterbox import ChatterboxTTS, ChatterboxMultilingualTTS
import torchaudio as ta

tts = ChatterboxMultilingualTTS.from_pretrained("cuda")

texts = ["Hello world!", "How are you?", "Batch processing rocks!", "I like Scissors Sixty Nine!"]
audio_paths = ["voice.wav", "voice.wav", "voice.wav", "voice.wav"]

# True batch processing - 3-5x faster!
outputs = tts.generate_batch(
    texts=texts,
    language_ids=["en"] * 4,
    audio_prompt_paths=audio_paths,
    max_batch_size=4
)

for idx, audio in enumerate(outputs):
    ta.save(f"test-{idx}.wav", audio, tts.sr)