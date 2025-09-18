import torchaudio as ta
import torch
import time
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

# Test batch processing with ChatterboxTTS
print("Testing batch processing with ChatterboxTTS...")
model = ChatterboxTTS.from_pretrained(device=device)

# Multiple texts for batch processing
texts = [
    "Hello, this is the first sentence.",
    "This is the second sentence in the batch.",
    "And here we have the third and final sentence.",
    "Fourth sentence to make the batch larger.",
    "Fifth and final sentence for performance testing.",
]

print(f"\n=== Performance Comparison ===")
print(f"Testing with {len(texts)} texts...")

# Time individual generation (baseline)
print("\n1. Individual generation (baseline):")
start_time = time.time()
individual_wavs = []
for i, text in enumerate(texts):
    wav = model.generate(text)
    individual_wavs.append(wav)
    ta.save(f"individual_test_{i+1}.wav", wav, model.sr)
individual_time = time.time() - start_time
print(f"Individual generation time: {individual_time:.2f}s")

# Time batch generation
print("\n2. Batch generation:")
start_time = time.time()
batch_wavs = model.generate_batch(texts)
batch_time = time.time() - start_time
print(f"Batch generation time: {batch_time:.2f}s")

# Save batch results and check content
for i, wav in enumerate(batch_wavs):
    ta.save(f"batch_test_{i+1}.wav", wav, model.sr)
    # Debug: Check if audio has content
    wav_tensor = wav if torch.is_tensor(wav) else torch.tensor(wav)
    print(f"  batch_test_{i+1}.wav - Shape: {wav_tensor.shape}, Max: {wav_tensor.abs().max():.4f}, Non-zero: {(wav_tensor != 0).sum().item()}")

# Performance improvement
speedup = individual_time / batch_time if batch_time > 0 else float('inf')
print(f"\nSpeedup: {speedup:.2f}x")
print(f"Time saved: {individual_time - batch_time:.2f}s ({(individual_time - batch_time)/individual_time*100:.1f}%)")

# Test batch processing with ChatterboxMultilingualTTS
print("\n" + "="*50)
print("Testing batch processing with ChatterboxMultilingualTTS...")
multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)

# Multiple texts in French for batch processing
french_texts = [
    "Bonjour, comment ça va?",
    "Ceci est le modèle de synthèse vocale multilingue Chatterbox.",
    "Il prend en charge vingt-trois langues différentes.",
]

print(f"\n=== Multilingual Performance Comparison ===")
print(f"Testing with {len(french_texts)} French texts...")

# Time individual generation (baseline)
print("\n1. Individual generation (baseline):")
start_time = time.time()
individual_french_wavs = []
for i, text in enumerate(french_texts):
    wav = multilingual_model.generate(text, language_id="fr")
    individual_french_wavs.append(wav)
    ta.save(f"individual_french_test_{i+1}.wav", wav, multilingual_model.sr)
individual_french_time = time.time() - start_time
print(f"Individual generation time: {individual_french_time:.2f}s")

# Time batch generation
print("\n2. Batch generation:")
start_time = time.time()
batch_french_wavs = multilingual_model.generate_batch(french_texts, language_id="fr")
batch_french_time = time.time() - start_time
print(f"Batch generation time: {batch_french_time:.2f}s")

# Save batch results
for i, wav in enumerate(batch_french_wavs):
    ta.save(f"batch_french_test_{i+1}.wav", wav, multilingual_model.sr)

# Performance improvement
french_speedup = individual_french_time / batch_french_time if batch_french_time > 0 else float('inf')
print(f"\nSpeedup: {french_speedup:.2f}x")
print(f"Time saved: {individual_french_time - batch_french_time:.2f}s ({(individual_french_time - batch_french_time)/individual_french_time*100:.1f}%)")

print("\n" + "="*50)
print("Batch processing test completed successfully!")
print(f"English batch speedup: {speedup:.2f}x")
print(f"French batch speedup: {french_speedup:.2f}x")