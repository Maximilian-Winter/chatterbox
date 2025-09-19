"""
ChatterboxTTS Benchmark Script
Compares single vs batch processing performance
"""

import time
import torch
import torchaudio as ta
from chatterbox import ChatterboxTTS
from typing import List, Dict, Any
import json
from datetime import datetime
import numpy as np


class TTSBenchmark:
    def __init__(self, device: str = "cuda", voice_file: str = "voice.wav"):
        """Initialize benchmark with TTS model and test data."""
        self.device = device
        self.voice_file = voice_file
        self.tts = None

        # Test sentences of varying lengths and complexity
        self.test_texts = [
            # Short sentences (< 10 words)
            "Hello world!",
            "How are you today?",

            # Medium sentences (10-20 words)
            "The quick brown fox jumps over the lazy dog in the sunny afternoon.",
            "Artificial intelligence is transforming how we interact with technology every single day.",

            # Long sentences (20-30 words)
            "In the midst of winter, I found there was within me an invincible summer that kept me warm through the coldest days.",
            "The development of advanced neural networks has revolutionized natural language processing, enabling machines to understand and generate human-like text with unprecedented accuracy.",

            # Complex sentences with punctuation
            "Wait, let me think about this: should we go to the park, the museum, or stay home?",
            "Wow! I can't believe it's already 2025 - time flies when you're having fun, doesn't it?"
        ]

        self.results = {
            "device": device,
            "timestamp": None,
            "test_sentences": len(self.test_texts),
            "single_processing": {},
            "batch_processing": {},
            "speedup": {},
            "quality_metrics": {}
        }

    def setup(self):
        """Initialize the TTS model."""
        print(f"Initializing ChatterboxTTS on {self.device}...")
        self.tts = ChatterboxTTS.from_pretrained(self.device)
        print("Model loaded successfully!\n")

    def warmup(self):
        """Perform warmup runs to ensure stable timing."""
        print("Performing warmup runs...")
        warmup_text = ["Warmup text"] * 2

        # Warmup single processing
        for text in warmup_text[:1]:
            _ = self.tts.generate(
                text=text,
                audio_prompt_path=self.voice_file
            )

        # Warmup batch processing
        _ = self.tts.generate_batch(
            texts=warmup_text,
            audio_prompt_paths=[self.voice_file] * 2,
            max_batch_size=2
        )

        print("Warmup complete!\n")

    def benchmark_single_processing(self) -> Dict[str, Any]:
        """Benchmark single text processing (sequential)."""
        print("=" * 60)
        print("BENCHMARKING SINGLE PROCESSING (Sequential)")
        print("=" * 60)

        outputs = []
        times = []

        total_start = time.perf_counter()

        for i, text in enumerate(self.test_texts):
            print(f"Processing text {i + 1}/{len(self.test_texts)}: '{text[:30]}...'")

            start = time.perf_counter()
            output = self.tts.generate(
                text=text,
                audio_prompt_path=self.voice_file,
                temperature=0.8,
                cfg_weight=0.5,
                repetition_penalty=1.2
            )
            end = time.perf_counter()

            elapsed = end - start
            times.append(elapsed)
            outputs.append(output)

            print(f"  Time: {elapsed:.2f}s")

            # Save audio
            ta.save(f"benchmark_single_{i}.wav", output, self.tts.sr)

        total_end = time.perf_counter()
        total_time = total_end - total_start

        results = {
            "total_time": total_time,
            "average_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": min(times),
            "max_time": max(times),
            "individual_times": times,
            "throughput": len(self.test_texts) / total_time  # texts per second
        }

        print(f"\nSingle Processing Summary:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average per text: {results['average_time']:.2f}s")
        print(f"  Throughput: {results['throughput']:.3f} texts/second\n")

        return results, outputs

    def benchmark_batch_processing(self, batch_sizes: List[int] = [2, 4, 8]) -> Dict[str, Any]:
        """Benchmark batch processing with different batch sizes."""
        print("=" * 60)
        print("BENCHMARKING BATCH PROCESSING")
        print("=" * 60)

        results = {}
        best_outputs = None
        best_time = float('inf')

        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")

            # Prepare batch inputs
            audio_paths = [self.voice_file] * len(self.test_texts)

            # Time the batch processing
            start = time.perf_counter()
            outputs = self.tts.generate_batch(
                texts=self.test_texts,
                audio_prompt_paths=audio_paths,
                temperatures=0.8,
                cfg_weights=0.5,
                repetition_penalties=1.2,
                max_batch_size=batch_size
            )
            end = time.perf_counter()

            elapsed = end - start

            results[f"batch_{batch_size}"] = {
                "total_time": elapsed,
                "average_time": elapsed / len(self.test_texts),
                "throughput": len(self.test_texts) / elapsed,
                "batch_size": batch_size
            }

            print(f"  Total time: {elapsed:.2f}s")
            print(f"  Average per text: {elapsed / len(self.test_texts):.2f}s")
            print(f"  Throughput: {len(self.test_texts) / elapsed:.3f} texts/second")

            # Save outputs from best batch size
            if elapsed < best_time:
                best_time = elapsed
                best_outputs = outputs

                # Save audio files
                for i, output in enumerate(outputs):
                    ta.save(f"benchmark_batch_{i}.wav", output, self.tts.sr)

        # Find best batch size
        best_batch = min(results.items(), key=lambda x: x[1]["total_time"])
        results["best_configuration"] = best_batch[0]
        results["best_metrics"] = best_batch[1]

        print(f"\nBest Batch Configuration: {best_batch[0]}")
        print(f"  Time: {best_batch[1]['total_time']:.2f}s")
        print(f"  Throughput: {best_batch[1]['throughput']:.3f} texts/second\n")

        return results, best_outputs

    def compare_audio_quality(self, single_outputs: List[torch.Tensor],
                              batch_outputs: List[torch.Tensor]) -> Dict[str, Any]:
        """Compare audio quality between single and batch processing."""
        print("=" * 60)
        print("COMPARING AUDIO QUALITY")
        print("=" * 60)

        quality_metrics = {
            "length_differences": [],
            "amplitude_differences": [],
            "silence_ratios": []
        }

        for i, (single, batch) in enumerate(zip(single_outputs, batch_outputs)):
            # Compare lengths
            len_diff = abs(single.shape[-1] - batch.shape[-1])
            quality_metrics["length_differences"].append(len_diff)

            # Compare amplitudes (RMS)
            single_rms = torch.sqrt(torch.mean(single ** 2)).item()
            batch_rms = torch.sqrt(torch.mean(batch ** 2)).item()
            amp_diff = abs(single_rms - batch_rms)
            quality_metrics["amplitude_differences"].append(amp_diff)

            # Check for silence (ratio of near-zero samples)
            threshold = 0.001
            single_silence = (torch.abs(single) < threshold).float().mean().item()
            batch_silence = (torch.abs(batch) < threshold).float().mean().item()
            quality_metrics["silence_ratios"].append({
                "single": single_silence,
                "batch": batch_silence
            })

            print(f"Text {i + 1}:")
            print(f"  Length diff: {len_diff} samples")
            print(f"  Amplitude diff: {amp_diff:.6f}")
            print(f"  Silence ratio - Single: {single_silence:.3f}, Batch: {batch_silence:.3f}")

        # Summary statistics
        avg_len_diff = np.mean(quality_metrics["length_differences"])
        avg_amp_diff = np.mean(quality_metrics["amplitude_differences"])

        print(f"\nQuality Summary:")
        print(f"  Average length difference: {avg_len_diff:.1f} samples")
        print(f"  Average amplitude difference: {avg_amp_diff:.6f}")

        return quality_metrics

    def run_benchmark(self):
        """Run the complete benchmark suite."""
        print("\n" + "=" * 60)
        print("CHATTERBOX TTS BENCHMARK")
        print("=" * 60 + "\n")

        # Setup
        self.setup()
        self.warmup()

        # Record timestamp
        self.results["timestamp"] = datetime.now().isoformat()

        # Run single processing benchmark
        single_results, single_outputs = self.benchmark_single_processing()
        self.results["single_processing"] = single_results

        # Run batch processing benchmark
        batch_results, batch_outputs = self.benchmark_batch_processing()
        self.results["batch_processing"] = batch_results

        # Calculate speedup
        best_batch = batch_results["best_metrics"]
        speedup = single_results["total_time"] / best_batch["total_time"]
        throughput_improvement = best_batch["throughput"] / single_results["throughput"]

        self.results["speedup"] = {
            "time_speedup": speedup,
            "throughput_improvement": throughput_improvement,
            "percent_faster": (speedup - 1) * 100
        }

        # Compare audio quality
        self.results["quality_metrics"] = self.compare_audio_quality(
            single_outputs, batch_outputs
        )

        # Print final summary
        self.print_summary()

        # Save results to JSON
        self.save_results()

        return self.results

    def print_summary(self):
        """Print a summary of benchmark results."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        single = self.results["single_processing"]
        batch = self.results["batch_processing"]["best_metrics"]
        speedup = self.results["speedup"]

        print(f"\nDevice: {self.results['device']}")
        print(f"Test sentences: {self.results['test_sentences']}")

        print(f"\nSingle Processing:")
        print(f"  Total time: {single['total_time']:.2f}s")
        print(f"  Throughput: {single['throughput']:.3f} texts/second")

        print(f"\nBest Batch Processing ({self.results['batch_processing']['best_configuration']}):")
        print(f"  Total time: {batch['total_time']:.2f}s")
        print(f"  Throughput: {batch['throughput']:.3f} texts/second")

        print(f"\nPerformance Improvement:")
        print(f"  Time speedup: {speedup['time_speedup']:.2f}x")
        print(f"  Throughput improvement: {speedup['throughput_improvement']:.2f}x")
        print(f"  Percent faster: {speedup['percent_faster']:.1f}%")

        print("\n" + "=" * 60)

    def save_results(self):
        """Save benchmark results to JSON file."""
        filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert numpy values to Python native types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            return obj

        results_native = convert_to_native(self.results)

        with open(filename, 'w') as f:
            json.dump(results_native, f, indent=2)

        print(f"\nResults saved to: {filename}")


def main():
    """Run the benchmark with command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark ChatterboxTTS")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--voice", default="voice.wav", help="Path to voice reference file")

    args = parser.parse_args()

    # Create and run benchmark
    benchmark = TTSBenchmark(device=args.device, voice_file=args.voice)
    results = benchmark.run_benchmark()

    # Quick results access
    speedup = results["speedup"]["time_speedup"]
    print(f"\n🎯 Final Result: Batch processing is {speedup:.2f}x faster!")


if __name__ == "__main__":
    main()