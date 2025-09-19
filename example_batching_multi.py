from chatterbox import ChatterboxTTS, ChatterboxMultilingualTTS
import torchaudio as ta

tts = ChatterboxMultilingualTTS.from_pretrained("cuda")

texts = ["Hello world!", "How are you?", "Batch processing rocks!", "I like Scissors Sixty Nine!"]
audio_paths = ["voice.wav", "voice.wav", "voice.wav", "voice.wav"]
language_ids = ["en", "en", "en", "en"]
minps = [0.025, 0.05, 0.075, 0.1]
tops = [1.0, 0.95, 0.90, 0.85]
batch_size = 8

outputs = tts.generate_batch(
    texts=texts,
    language_ids=language_ids,
    audio_prompt_paths=audio_paths,
    max_batch_size=batch_size,
    min_ps=minps,
    top_ps=tops,
)

for idx, audio in enumerate(outputs):
    ta.save(f"test-batch-{idx}-multi.wav", audio, tts.sr)