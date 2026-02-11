#!/usr/bin/env python3
"""
Single-core inference for SigLIP 384px
Measured performance: 13.38 img/s, 74.74 ms latency
"""
import torch
import torch_neuronx
import time
import os

os.environ['NEURON_RT_LOG_LEVEL'] = 'ERROR'

MODEL_FILE = 'siglip_384_neuron.pt'

print("="*80)
print("SigLIP 384px - Single Core Inference")
print("="*80)

# Check model exists
if not os.path.exists(MODEL_FILE):
    print(f"✗ Error: {MODEL_FILE} not found")
    print("Run: python3 compile_model.py")
    exit(1)

# Load model
print(f"\nLoading model from {MODEL_FILE}...")
model = torch.jit.load(MODEL_FILE)
model.eval()
print("✓ Model loaded")

# Create test input
input_tensor = torch.randn(1, 3, 384, 384)
print(f"✓ Input shape: {input_tensor.shape}")

# Warmup
print('\nWarming up (10 iterations)...')
for _ in range(10):
    with torch.no_grad():
        _ = model(input_tensor)
print("✓ Warmup complete")

# Benchmark
print('\nBenchmarking (50 iterations)...')
times = []
for _ in range(50):
    start = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    times.append(time.time() - start)

# Calculate metrics
avg_time = sum(times) / len(times)
throughput = 1 / avg_time
latency_ms = avg_time * 1000

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Throughput: {throughput:.2f} img/s")
print(f"Latency: {latency_ms:.2f} ms/image")
print(f"Output shape: {output.shape}")
print("="*80)
