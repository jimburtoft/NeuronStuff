#!/usr/bin/env python3
"""
Worker process for multi-core inference
Run with NEURON_RT_VISIBLE_CORES environment variable
"""
import torch
import torch_neuronx
import time
import sys
import os

os.environ['NEURON_RT_LOG_LEVEL'] = 'ERROR'

MODEL_FILE = 'siglip_384_neuron.pt'

if len(sys.argv) != 2:
    print("Usage: NEURON_RT_VISIBLE_CORES=X python3 inference_worker.py <core_id>")
    sys.exit(1)

core_id = sys.argv[1]
visible_cores = os.environ.get('NEURON_RT_VISIBLE_CORES', 'not set')

print(f"[Core {core_id}] Starting on NeuronCore {visible_cores}")

# Load model
print(f"[Core {core_id}] Loading model...")
start_load = time.time()
model = torch.jit.load(MODEL_FILE)
model.eval()
load_time = time.time() - start_load
print(f"[Core {core_id}] Model loaded in {load_time:.1f}s")

# Create input
input_tensor = torch.randn(1, 3, 384, 384)

# Warmup
print(f"[Core {core_id}] Warming up...")
for _ in range(10):
    with torch.no_grad():
        _ = model(input_tensor)

# Continuous inference loop
print(f"[Core {core_id}] Ready for inference")
print(f"[Core {core_id}] Press Ctrl+C to stop")

try:
    count = 0
    start_time = time.time()
    
    while True:
        with torch.no_grad():
            output = model(input_tensor)
        
        count += 1
        
        # Report every 100 inferences
        if count % 100 == 0:
            elapsed = time.time() - start_time
            throughput = count / elapsed
            print(f"[Core {core_id}] {count} inferences, {throughput:.2f} img/s")

except KeyboardInterrupt:
    elapsed = time.time() - start_time
    throughput = count / elapsed
    print(f"\n[Core {core_id}] Stopped")
    print(f"[Core {core_id}] Total: {count} inferences in {elapsed:.1f}s")
    print(f"[Core {core_id}] Throughput: {throughput:.2f} img/s")
