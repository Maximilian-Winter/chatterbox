from chatterbox import ChatterboxTTS
import torchaudio as ta

tts = ChatterboxTTS.from_pretrained("cuda")

texts = ["Hello world!", "How are you?", "Batch processing rocks!", "I like Scissors Sixty Nine!"]
audio_paths = ["voice.wav", "voice.wav", "voice.wav", "voice.wav"]
minps = [0.1, 0.1, 0.1, 0.1]
tops = [1.0, 0.95, 0.90, 0.85]
batch_size = 8

outputs = tts.generate_batch(
    texts=texts,
    audio_prompt_paths=audio_paths,
    max_batch_size=batch_size,
    min_ps=minps,
    top_ps=tops,
)

for idx, audio in enumerate(outputs):
    ta.save(f"test-batch-{idx}.wav", audio, tts.sr)