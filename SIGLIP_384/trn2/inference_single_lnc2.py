#!/usr/bin/env python3
"""
Single-core inference benchmark for SigLIP 384px on Trainium2 with LNC=2.

Expected performance: ~21.95 img/s, 45.56 ms latency

Usage:
    NEURON_LOGICAL_NC_CONFIG=2 python3 inference_single_lnc2.py
"""

import torch
import torch_neuronx
import time
import os

os.environ["NEURON_RT_LOG_LEVEL"] = "ERROR"

MODEL_FILE = "siglip_384_neuron.pt"
NUM_WARMUP = 10
NUM_ITERATIONS = 50

print("=" * 80)
print("SigLIP 384px - Single Core Inference (LNC=2)")
print("=" * 80)
print(f"Configuration: LNC=2 (4 logical cores on trn2.3xlarge)")
print(f"Model: {MODEL_FILE}")

# Check model exists
if not os.path.exists(MODEL_FILE):
    print(f"\nError: {MODEL_FILE} not found")
    print("Run: NEURON_LOGICAL_NC_CONFIG=2 python3 compile_model_lnc2.py")
    exit(1)

# Load model
print(f"\nLoading model...")
model = torch.jit.load(MODEL_FILE)
model.eval()
print("Model loaded")

# Create test input
input_tensor = torch.randn(1, 3, 384, 384)
print(f"Input shape: {input_tensor.shape}")

# Warmup
print(f"\nWarming up ({NUM_WARMUP} iterations)...")
for _ in range(NUM_WARMUP):
    with torch.no_grad():
        _ = model(input_tensor)
print("Warmup complete")

# Benchmark
print(f"\nBenchmarking ({NUM_ITERATIONS} iterations)...")
times = []
for _ in range(NUM_ITERATIONS):
    start = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    times.append(time.time() - start)

# Calculate metrics
avg_time = sum(times) / len(times)
throughput = 1 / avg_time
latency_ms = avg_time * 1000

print("\n" + "=" * 80)
print("RESULTS - LNC=2")
print("=" * 80)
print(f"Throughput: {throughput:.2f} img/s")
print(f"Latency: {latency_ms:.2f} ms/image")
print(f"Output shape: {output.shape}")
print("=" * 80)
