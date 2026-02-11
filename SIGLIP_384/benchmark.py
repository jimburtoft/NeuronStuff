#!/usr/bin/env python3
"""
Benchmark SigLIP 384px performance
Tests single core and DataParallel configurations
"""
import torch
import torch_neuronx
import time
import os

os.environ['NEURON_RT_LOG_LEVEL'] = 'ERROR'

MODEL_FILE = 'siglip_384_neuron.pt'
NUM_WARMUP = 10
NUM_ITERATIONS = 50

def benchmark_single_core():
    """Benchmark single core performance"""
    print("\n" + "="*80)
    print("Single Core Benchmark")
    print("="*80)
    
    model = torch.jit.load(MODEL_FILE)
    model.eval()
    
    input_tensor = torch.randn(1, 3, 384, 384)
    
    # Warmup
    print(f"Warming up ({NUM_WARMUP} iterations)...")
    for _ in range(NUM_WARMUP):
        with torch.no_grad():
            _ = model(input_tensor)
    
    # Benchmark
    print(f"Benchmarking ({NUM_ITERATIONS} iterations)...")
    times = []
    for _ in range(NUM_ITERATIONS):
        start = time.time()
        with torch.no_grad():
            output = model(input_tensor)
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    throughput = 1 / avg_time
    latency_ms = avg_time * 1000
    
    print(f"\nResults:")
    print(f"  Throughput: {throughput:.2f} img/s")
    print(f"  Latency: {latency_ms:.2f} ms")
    print(f"  Output shape: {output.shape}")
    
    return throughput

def benchmark_dataparallel():
    """Benchmark DataParallel (2 cores) performance"""
    print("\n" + "="*80)
    print("DataParallel Benchmark (2 cores)")
    print("="*80)
    
    model = torch.jit.load(MODEL_FILE)
    model.eval()
    
    # Wrap with DataParallel
    model_parallel = torch_neuronx.DataParallel(model)
    
    # Batch size 2 (one per core)
    input_tensor = torch.randn(2, 3, 384, 384)
    
    # Warmup
    print(f"Warming up ({NUM_WARMUP} iterations)...")
    for _ in range(NUM_WARMUP):
        with torch.no_grad():
            _ = model_parallel(input_tensor)
    
    # Benchmark
    print(f"Benchmarking ({NUM_ITERATIONS} iterations)...")
    times = []
    for _ in range(NUM_ITERATIONS):
        start = time.time()
        with torch.no_grad():
            output = model_parallel(input_tensor)
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    throughput = 2 / avg_time  # 2 images per batch
    latency_ms = avg_time * 1000
    per_image_latency = latency_ms / 2
    
    print(f"\nResults:")
    print(f"  Throughput: {throughput:.2f} img/s")
    print(f"  Batch latency: {latency_ms:.2f} ms")
    print(f"  Per-image latency: {per_image_latency:.2f} ms")
    print(f"  Output shape: {output.shape}")
    
    return throughput

def main():
    print("="*80)
    print("SigLIP 384px Performance Benchmark")
    print("="*80)
    
    # Check model exists
    if not os.path.exists(MODEL_FILE):
        print(f"\n✗ Error: {MODEL_FILE} not found")
        print("Run: python3 compile_model.py")
        return
    
    # Run benchmarks
    single_throughput = benchmark_single_core()
    dataparallel_throughput = benchmark_dataparallel()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Single core: {single_throughput:.2f} img/s")
    print(f"DataParallel (2 cores): {dataparallel_throughput:.2f} img/s")
    print(f"Speedup: {dataparallel_throughput/single_throughput:.2f}x")
    print(f"Efficiency: {(dataparallel_throughput/single_throughput/2)*100:.1f}%")
    print("\nNote: For maximum throughput, use separate processes:")
    print("  ./run_dual_core.sh (~26.5 img/s, 1.98x speedup)")
    print("="*80)

if __name__ == '__main__':
    main()
