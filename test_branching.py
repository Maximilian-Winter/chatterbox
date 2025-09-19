from chatterbox import ChatterboxTTS

tts = ChatterboxTTS.from_pretrained("cuda")

texts = ["Hello world!", "How are you?", "Batch processing rocks!"]
audio_paths = ["voice.wav", "voice.wav", "voice.wav"]

# True batch processing - 3-5x faster!
outputs = tts.generate_batch(
    texts=texts,
    audio_prompt_paths=audio_paths,
    max_batch_size=8
)