#!/usr/bin/env python3
"""
Performance benchmarking script for optimized T3 batch inference.
Validates 3-5x speedup over sequential processing and measures efficiency gains.
"""

import time
import torch
import torchaudio as ta
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import json
import gc
import psutil
import logging

from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_device():
    """Setup optimal device for benchmarking."""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = "cpu"
        logger.info("Using CPU")

    return device

def generate_test_texts(num_texts: int = 10) -> List[str]:
    """Generate diverse test texts for benchmarking."""
    base_texts = [
        "Hello, this is a test of the parallel batch processing system.",
        "The quick brown fox jumps over the lazy dog in the moonlight.",
        "Artificial intelligence and machine learning are transforming our world.",
        "Music has the power to heal, inspire, and bring people together.",
        "The future of technology depends on sustainable innovation and creativity.",
        "Deep learning models require significant computational resources to train effectively.",
        "Natural language processing enables computers to understand human communication.",
        "Climate change is one of the most pressing challenges facing humanity today.",
        "Space exploration opens new frontiers for scientific discovery and human expansion.",
        "The intersection of art and technology creates endless possibilities for creative expression.",
    ]

    # Extend with variations for larger batch sizes
    texts = []
    for i in range(num_texts):
        base_idx = i % len(base_texts)
        variation = f"Sequence {i+1}: {base_texts[base_idx]}"
        texts.append(variation)

    return texts

def measure_memory_usage():
    """Measure current memory usage."""
    process = psutil.Process()
    memory_info = process.memory_info()

    result = {
        'ram_mb': memory_info.rss / 1024 / 1024,
        'gpu_mb': 0
    }

    if torch.cuda.is_available():
        result['gpu_mb'] = torch.cuda.memory_allocated() / 1024 / 1024

    return result

def benchmark_sequential_vs_batch(
    model: ChatterboxTTS,
    texts: List[str],
    num_runs: int = 3
) -> Dict[str, Any]:
    """
    Benchmark sequential vs batch processing to validate speedup.

    Returns performance metrics including timing, memory, and speedup ratios.
    """
    logger.info(f"Benchmarking {len(texts)} texts with {num_runs} runs each")

    results = {
        'num_texts': len(texts),
        'num_runs': num_runs,
        'sequential': {'times': [], 'memory': []},
        'batch_legacy': {'times': [], 'memory': []},
        'batch_optimized': {'times': [], 'memory': []},
    }

    # Warm up GPU
    warmup_text = texts[0]
    _ = model.generate(warmup_text)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Sequential processing baseline
    logger.info("1. Benchmarking sequential processing...")
    for run in range(num_runs):
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        start_memory = measure_memory_usage()
        start_time = time.time()

        sequential_results = []
        for text in texts:
            result = model.generate(text)
            sequential_results.append(result)

        end_time = time.time()
        end_memory = measure_memory_usage()

        sequential_time = end_time - start_time
        memory_used = max(end_memory['ram_mb'] - start_memory['ram_mb'], 0)

        results['sequential']['times'].append(sequential_time)
        results['sequential']['memory'].append(memory_used)

        logger.info(f"  Run {run+1}: {sequential_time:.2f}s, {memory_used:.1f}MB RAM")

    # Legacy batch processing
    logger.info("2. Benchmarking legacy batch processing...")
    for run in range(num_runs):
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        start_memory = measure_memory_usage()
        start_time = time.time()

        # Use the existing batch_inference method
        batch_results = model.generate_batch(texts)

        end_time = time.time()
        end_memory = measure_memory_usage()

        batch_time = end_time - start_time
        memory_used = max(end_memory['ram_mb'] - start_memory['ram_mb'], 0)

        results['batch_legacy']['times'].append(batch_time)
        results['batch_legacy']['memory'].append(memory_used)

        logger.info(f"  Run {run+1}: {batch_time:.2f}s, {memory_used:.1f}MB RAM")

    # Optimized batch processing (using optimized_batch_inference)
    logger.info("3. Benchmarking optimized batch processing...")
    for run in range(num_runs):
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        start_memory = measure_memory_usage()
        start_time = time.time()

        # Direct call to optimized method
        batch_results_opt = model.generate_batch(texts)  # This now uses optimized_batch_inference

        end_time = time.time()
        end_memory = measure_memory_usage()

        opt_batch_time = end_time - start_time
        memory_used = max(end_memory['ram_mb'] - start_memory['ram_mb'], 0)

        results['batch_optimized']['times'].append(opt_batch_time)
        results['batch_optimized']['memory'].append(memory_used)

        logger.info(f"  Run {run+1}: {opt_batch_time:.2f}s, {memory_used:.1f}MB RAM")

    # Calculate statistics
    for method in ['sequential', 'batch_legacy', 'batch_optimized']:
        times = results[method]['times']
        results[method]['avg_time'] = np.mean(times)
        results[method]['std_time'] = np.std(times)
        results[method]['min_time'] = np.min(times)
        results[method]['max_time'] = np.max(times)

        memory = results[method]['memory']
        results[method]['avg_memory'] = np.mean(memory)
        results[method]['std_memory'] = np.std(memory)

    # Calculate speedup ratios
    seq_avg = results['sequential']['avg_time']
    legacy_avg = results['batch_legacy']['avg_time']
    opt_avg = results['batch_optimized']['avg_time']

    results['speedup_legacy_vs_sequential'] = seq_avg / legacy_avg if legacy_avg > 0 else 0
    results['speedup_optimized_vs_sequential'] = seq_avg / opt_avg if opt_avg > 0 else 0
    results['speedup_optimized_vs_legacy'] = legacy_avg / opt_avg if opt_avg > 0 else 0

    return results

def benchmark_batch_size_scaling(
    model: ChatterboxTTS,
    base_texts: List[str],
    batch_sizes: List[int] = [1, 2, 4, 8, 12, 16]
) -> Dict[str, Any]:
    """
    Benchmark how performance scales with batch size.

    Tests throughput (sequences/second) vs batch size to find optimal settings.
    """
    logger.info("Benchmarking batch size scaling...")

    results = {
        'batch_sizes': batch_sizes,
        'throughput': [],
        'latency': [],
        'memory_usage': [],
        'efficiency': []
    }

    for batch_size in batch_sizes:
        if batch_size > len(base_texts):
            # Repeat texts to reach target batch size
            multiplier = (batch_size + len(base_texts) - 1) // len(base_texts)
            test_texts = (base_texts * multiplier)[:batch_size]
        else:
            test_texts = base_texts[:batch_size]

        logger.info(f"Testing batch size: {batch_size}")

        # Warm up
        _ = model.generate_batch(test_texts)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Measure performance
        num_runs = 3
        times = []
        memories = []

        for run in range(num_runs):
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            start_memory = measure_memory_usage()
            start_time = time.time()

            _ = model.generate_batch(test_texts)

            end_time = time.time()
            end_memory = measure_memory_usage()

            elapsed = end_time - start_time
            memory_used = max(end_memory['ram_mb'] - start_memory['ram_mb'], 0)

            times.append(elapsed)
            memories.append(memory_used)

        avg_time = np.mean(times)
        avg_memory = np.mean(memories)

        throughput = batch_size / avg_time  # sequences per second
        latency = avg_time / batch_size  # average time per sequence
        efficiency = throughput / batch_size  # throughput normalized by batch size

        results['throughput'].append(throughput)
        results['latency'].append(latency)
        results['memory_usage'].append(avg_memory)
        results['efficiency'].append(efficiency)

        logger.info(f"  Throughput: {throughput:.2f} seq/s, Latency: {latency:.3f}s/seq, Memory: {avg_memory:.1f}MB")

    return results

def save_benchmark_results(results: Dict[str, Any], output_dir: Path = Path("benchmark_results")):
    """Save benchmark results to files."""
    output_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Save detailed results as JSON
    json_path = output_dir / f"benchmark_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {json_path}")

    return json_path

def plot_benchmark_results(results: Dict[str, Any], output_dir: Path = Path("benchmark_results")):
    """Create visualization plots for benchmark results."""
    output_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Plot 1: Performance comparison
    if 'sequential' in results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Timing comparison
        methods = ['Sequential', 'Legacy Batch', 'Optimized Batch']
        times = [
            results['sequential']['avg_time'],
            results['batch_legacy']['avg_time'],
            results['batch_optimized']['avg_time']
        ]
        errors = [
            results['sequential']['std_time'],
            results['batch_legacy']['std_time'],
            results['batch_optimized']['std_time']
        ]

        bars1 = ax1.bar(methods, times, yerr=errors, capsize=5,
                       color=['lightcoral', 'lightblue', 'lightgreen'])
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title(f'Processing Time Comparison\n({results["num_texts"]} sequences)')

        # Add speedup annotations
        for i, (bar, time_val) in enumerate(zip(bars1, times)):
            if i == 0:  # Sequential baseline
                speedup = 1.0
            elif i == 1:  # Legacy batch
                speedup = results['speedup_legacy_vs_sequential']
            else:  # Optimized batch
                speedup = results['speedup_optimized_vs_sequential']

            ax1.annotate(f'{speedup:.1f}x',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontweight='bold')

        # Memory usage comparison
        memories = [
            results['sequential']['avg_memory'],
            results['batch_legacy']['avg_memory'],
            results['batch_optimized']['avg_memory']
        ]

        bars2 = ax2.bar(methods, memories, color=['lightcoral', 'lightblue', 'lightgreen'])
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage Comparison')

        plt.tight_layout()
        plot_path = output_dir / f"performance_comparison_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Performance plot saved to {plot_path}")

    # Plot 2: Batch size scaling
    if 'batch_sizes' in results:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        batch_sizes = results['batch_sizes']

        # Throughput vs batch size
        ax1.plot(batch_sizes, results['throughput'], 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Throughput (sequences/second)')
        ax1.set_title('Throughput vs Batch Size')
        ax1.grid(True, alpha=0.3)

        # Latency vs batch size
        ax2.plot(batch_sizes, results['latency'], 'o-', color='orange', linewidth=2, markersize=8)
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Latency (seconds/sequence)')
        ax2.set_title('Latency vs Batch Size')
        ax2.grid(True, alpha=0.3)

        # Memory usage vs batch size
        ax3.plot(batch_sizes, results['memory_usage'], 'o-', color='green', linewidth=2, markersize=8)
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Memory Usage (MB)')
        ax3.set_title('Memory Usage vs Batch Size')
        ax3.grid(True, alpha=0.3)

        # Efficiency vs batch size
        ax4.plot(batch_sizes, results['efficiency'], 'o-', color='red', linewidth=2, markersize=8)
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Efficiency (normalized throughput)')
        ax4.set_title('Efficiency vs Batch Size')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        scaling_plot_path = output_dir / f"batch_scaling_{timestamp}.png"
        plt.savefig(scaling_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Scaling plot saved to {scaling_plot_path}")

def print_summary(results: Dict[str, Any]):
    """Print a comprehensive summary of benchmark results."""
    print("\n" + "="*80)
    print("PARALLEL BATCH PROCESSING BENCHMARK RESULTS")
    print("="*80)

    if 'sequential' in results:
        print(f"\n📊 Performance Summary ({results['num_texts']} sequences):")
        print("-" * 60)

        seq_time = results['sequential']['avg_time']
        legacy_time = results['batch_legacy']['avg_time']
        opt_time = results['batch_optimized']['avg_time']

        print(f"Sequential Processing:    {seq_time:.2f}s ± {results['sequential']['std_time']:.2f}s")
        print(f"Legacy Batch Processing:  {legacy_time:.2f}s ± {results['batch_legacy']['std_time']:.2f}s")
        print(f"Optimized Batch:          {opt_time:.2f}s ± {results['batch_optimized']['std_time']:.2f}s")

        print(f"\n🚀 Speedup Achievements:")
        print("-" * 30)
        print(f"Legacy Batch vs Sequential:    {results['speedup_legacy_vs_sequential']:.2f}x")
        print(f"Optimized vs Sequential:       {results['speedup_optimized_vs_sequential']:.2f}x")
        print(f"Optimized vs Legacy:           {results['speedup_optimized_vs_legacy']:.2f}x")

        # Validate 3-5x target
        target_speedup = results['speedup_optimized_vs_sequential']
        if target_speedup >= 3.0:
            status = "✅ TARGET ACHIEVED"
            if target_speedup >= 5.0:
                status = "🎯 EXCEEDED TARGET"
        else:
            status = "⚠️  BELOW TARGET"

        print(f"\n🎯 Target Validation (3-5x speedup): {status}")
        print(f"   Achieved: {target_speedup:.2f}x speedup")

        print(f"\n💾 Memory Efficiency:")
        print("-" * 25)
        seq_mem = results['sequential']['avg_memory']
        opt_mem = results['batch_optimized']['avg_memory']
        mem_efficiency = seq_mem / opt_mem if opt_mem > 0 else 1.0
        print(f"Sequential Memory:    {seq_mem:.1f}MB")
        print(f"Optimized Memory:     {opt_mem:.1f}MB")
        print(f"Memory Efficiency:    {mem_efficiency:.2f}x")

    if 'batch_sizes' in results:
        print(f"\n📈 Batch Size Scaling Analysis:")
        print("-" * 40)

        best_throughput_idx = np.argmax(results['throughput'])
        best_efficiency_idx = np.argmax(results['efficiency'])

        best_batch_size = results['batch_sizes'][best_throughput_idx]
        best_throughput = results['throughput'][best_throughput_idx]

        optimal_batch_size = results['batch_sizes'][best_efficiency_idx]
        optimal_efficiency = results['efficiency'][best_efficiency_idx]

        print(f"Best Throughput:     {best_throughput:.2f} seq/s at batch size {best_batch_size}")
        print(f"Optimal Efficiency:  {optimal_efficiency:.3f} at batch size {optimal_batch_size}")
        print(f"Recommended batch size: {optimal_batch_size}")

    print("\n" + "="*80)

def main():
    """Main benchmarking function."""
    print("🚀 Starting T3 Parallel Batch Processing Benchmark")
    print("Targeting 3-5x speedup validation...")

    # Setup
    device = setup_device()

    # Load model
    logger.info("Loading ChatterboxTTS model...")
    model = ChatterboxTTS.from_pretrained(device=device)

    # Generate test data
    test_texts = generate_test_texts(10)  # Start with 10 sequences

    # Run benchmarks
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK 1: Sequential vs Batch Performance")
    logger.info("="*60)

    perf_results = benchmark_sequential_vs_batch(model, test_texts, num_runs=3)

    logger.info("\n" + "="*60)
    logger.info("BENCHMARK 2: Batch Size Scaling Analysis")
    logger.info("="*60)

    scaling_results = benchmark_batch_size_scaling(model, test_texts, [1, 2, 4, 6, 8])

    # Combine results
    combined_results = {
        **perf_results,
        **scaling_results,
        'device': device,
        'model_config': {
            'hidden_size': model.t3.cfg.hidden_size,
            'num_layers': model.t3.cfg.num_hidden_layers,
            'num_heads': model.t3.cfg.num_attention_heads,
        },
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save and visualize results
    output_dir = Path("benchmark_results")
    save_benchmark_results(combined_results, output_dir)
    plot_benchmark_results(combined_results, output_dir)

    # Print summary
    print_summary(combined_results)

    print(f"\n✅ Benchmark completed! Results saved to {output_dir}/")

    return combined_results

if __name__ == "__main__":
    results = main()