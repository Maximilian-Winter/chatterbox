#!/usr/bin/env python3
"""
Example script demonstrating batch processing with Chatterbox TTS.

This script shows how to efficiently generate multiple audio files
using the new batch processing capabilities.
"""

import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# Example 1: Basic English batch processing
print("\n=== Example 1: English Batch Processing ===")
model = ChatterboxTTS.from_pretrained(device=device)

# Multiple texts to synthesize
texts = [
    "Welcome to the world of efficient text-to-speech synthesis.",
    "Batch processing allows you to generate multiple audio files quickly.",
    "This is much faster than processing each text individually.",
    "Enjoy the improved performance and convenience!"
]

print(f"Generating audio for {len(texts)} texts...")

# Generate all audio files in batch
batch_results = model.generate_batch(
    texts=texts,
    temperature=0.8,
    cfg_weight=0.5,
    exaggeration=0.5
)

# Save all results
for i, wav in enumerate(batch_results):
    ta.save(f"batch_example_{i+1}.wav", wav, model.sr)

print(f"✅ Generated {len(batch_results)} English audio files")


# Example 2: Multilingual batch processing
print("\n=== Example 2: Multilingual Batch Processing ===")
multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)

# Texts in different languages
multilingual_texts = [
    "Hello, welcome to our international TTS demo.",           # English
    "Bonjour, bienvenue dans notre démo TTS internationale.", # French
    "Hola, bienvenido a nuestra demostración TTS internacional.", # Spanish
    "こんにちは、国際的なTTSデモへようこそ。",                  # Japanese
    "你好，欢迎来到我们的国际TTS演示。"                        # Chinese
]

language_ids = ["en", "fr", "es", "ja", "zh"]

print(f"Generating audio for {len(multilingual_texts)} texts in {len(set(language_ids))} languages...")

# Generate multilingual batch
multilingual_results = multilingual_model.generate_batch(
    texts=multilingual_texts,
    language_ids=language_ids,
    temperature=0.8,
    cfg_weight=0.5,
    exaggeration=0.5
)

# Save multilingual results
for i, (wav, lang) in enumerate(zip(multilingual_results, language_ids)):
    ta.save(f"multilingual_batch_{lang}_{i+1}.wav", wav, multilingual_model.sr)

print(f"✅ Generated {len(multilingual_results)} multilingual audio files")


# Example 3: Batch processing with chunking (for large datasets)
print("\n=== Example 3: Large Batch with Chunking ===")

# Large list of texts (simulated)
large_text_list = [
    f"This is sentence number {i+1} in our large batch processing test."
    for i in range(25)  # 25 sentences
]

print(f"Processing {len(large_text_list)} texts with automatic chunking...")

# The max_batch_size parameter automatically chunks large batches
large_batch_results = model.generate_batch(
    texts=large_text_list,
    max_batch_size=8,  # Process in chunks of 8
    temperature=0.8
)

print(f"✅ Generated {len(large_batch_results)} audio files from large batch")

# Save only first few examples
for i in range(min(5, len(large_batch_results))):
    ta.save(f"large_batch_sample_{i+1}.wav", large_batch_results[i], model.sr)


# Example 4: Performance comparison
print("\n=== Example 4: Performance Comparison ===")
import time

# Small test set for timing
test_texts = [
    "Speed test sentence one.",
    "Speed test sentence two.",
    "Speed test sentence three."
]

# Time individual processing
print("Timing individual processing...")
start_time = time.time()
individual_results = []
for text in test_texts:
    wav = model.generate(text)
    individual_results.append(wav)
individual_time = time.time() - start_time

# Time batch processing
print("Timing batch processing...")
start_time = time.time()
batch_results = model.generate_batch(test_texts)
batch_time = time.time() - start_time

print(f"Individual processing: {individual_time:.2f}s")
print(f"Batch processing: {batch_time:.2f}s")
print(f"Speedup: {individual_time / batch_time:.2f}x")

print("\n🎉 All examples completed successfully!")
print("\nTo use batch processing in your own code:")
print("1. Load your model: model = ChatterboxTTS.from_pretrained(device)")
print("2. Prepare your texts: texts = ['text1', 'text2', ...]")
print("3. Generate batch: results = model.generate_batch(texts)")
print("4. Save results: [ta.save(f'file_{i}.wav', wav, model.sr) for i, wav in enumerate(results)]")