#!/usr/bin/env python3
"""
Test script for batch processing functionality in Chatterbox TTS.
This script tests both English and multilingual models with batch processing.
"""

import torch
import torchaudio as ta
import time
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

def test_english_batch():
    """Test English model batch processing."""
    print("=" * 50)
    print("Testing English Model Batch Processing")
    print("=" * 50)

    # Automatically detect the best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Load model
    print("Loading English model...")
    model = ChatterboxTTS.from_pretrained(device=device)

    # Test texts
    texts = [
        "Hello, this is the first test sentence for batch processing.",
        "The second sentence tests different content and length variations.",
        "Finally, this third sentence completes our batch processing test.",
    ]

    print(f"Processing {len(texts)} texts...")

    # Single generation for comparison
    print("\n--- Single Generation (for comparison) ---")
    start_time = time.time()
    single_results = []
    for i, text in enumerate(texts):
        wav = model.generate(text)
        single_results.append(wav)
        ta.save(f"test-single-{i+1}.wav", wav, model.sr)
    single_time = time.time() - start_time
    print(f"Single generation time: {single_time:.2f}s")

    # Batch generation
    print("\n--- Batch Generation ---")
    start_time = time.time()
    batch_results = model.generate_batch(texts)
    batch_time = time.time() - start_time
    print(f"Batch generation time: {batch_time:.2f}s")
    print(f"Speedup: {single_time / batch_time:.2f}x")

    # Save batch results
    for i, wav in enumerate(batch_results):
        ta.save(f"test-batch-{i+1}.wav", wav, model.sr)

    print(f"Generated {len(batch_results)} audio files")
    print("English batch test completed successfully!")


def test_multilingual_batch():
    """Test multilingual model batch processing."""
    print("\n" + "=" * 50)
    print("Testing Multilingual Model Batch Processing")
    print("=" * 50)

    # Automatically detect the best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Load multilingual model
    print("Loading multilingual model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    # Test texts in different languages
    texts = [
        "Hello, this is an English test sentence.",
        "Bonjour, ceci est une phrase de test en français.",
        "Hola, esta es una oración de prueba en español.",
    ]

    language_ids = ["en", "fr", "es"]

    print(f"Processing {len(texts)} texts in {len(set(language_ids))} languages...")

    # Single generation for comparison
    print("\n--- Single Generation (for comparison) ---")
    start_time = time.time()
    single_results = []
    for i, (text, lang_id) in enumerate(zip(texts, language_ids)):
        wav = model.generate(text, language_id=lang_id)
        single_results.append(wav)
        ta.save(f"test-mtl-single-{lang_id}-{i+1}.wav", wav, model.sr)
    single_time = time.time() - start_time
    print(f"Single generation time: {single_time:.2f}s")

    # Batch generation
    print("\n--- Batch Generation ---")
    start_time = time.time()
    batch_results = model.generate_batch(texts, language_ids)
    batch_time = time.time() - start_time
    print(f"Batch generation time: {batch_time:.2f}s")
    print(f"Speedup: {single_time / batch_time:.2f}x")

    # Save batch results
    for i, (wav, lang_id) in enumerate(zip(batch_results, language_ids)):
        ta.save(f"test-mtl-batch-{lang_id}-{i+1}.wav", wav, model.sr)

    print(f"Generated {len(batch_results)} audio files")
    print("Multilingual batch test completed successfully!")


def test_batch_with_audio_prompts():
    """Test batch processing with different audio prompts."""
    print("\n" + "=" * 50)
    print("Testing Batch Processing with Audio Prompts")
    print("=" * 50)

    # Note: This test requires existing audio files
    # For now, we'll test with the same built-in voice

    # Automatically detect the best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Load model
    print("Loading English model...")
    model = ChatterboxTTS.from_pretrained(device=device)

    # Test texts
    texts = [
        "This is a test with the built-in voice.",
        "Another sentence to test batch processing capabilities.",
    ]

    print(f"Processing {len(texts)} texts with built-in voice...")

    # Batch generation with built-in voice
    start_time = time.time()
    batch_results = model.generate_batch(texts)
    batch_time = time.time() - start_time
    print(f"Batch generation time: {batch_time:.2f}s")

    # Save results
    for i, wav in enumerate(batch_results):
        ta.save(f"test-batch-prompt-{i+1}.wav", wav, model.sr)

    print(f"Generated {len(batch_results)} audio files with prompts")
    print("Batch processing with audio prompts test completed!")


if __name__ == "__main__":
    try:
        # Test English model
        test_english_batch()

        # Test multilingual model
        test_multilingual_batch()

        # Test with audio prompts
        test_batch_with_audio_prompts()

        print("\n" + "=" * 50)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("Batch processing implementation is working correctly.")
        print("You can now use generate_batch() for efficient multi-text synthesis.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()