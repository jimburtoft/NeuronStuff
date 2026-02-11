#!/usr/bin/env python3
"""
Compile SigLIP 384px model for AWS Neuron (inf2.8xlarge)
Configuration: batch size 1, optlevel 3, model-type transformer
Expected compile time: 239 seconds
"""
import torch
import torch_neuronx
from transformers import AutoModel, AutoImageProcessor
import time
import os

print("="*80)
print("SigLIP 384px Model Compilation for Neuron")
print("="*80)

# Configuration
MODEL_PATH = '/home/ubuntu/timm/ViT-SO400M-14-SigLIP-384'
OUTPUT_FILE = 'siglip_384_neuron.pt'
BATCH_SIZE = 1
IMAGE_SIZE = 384

print(f"\nConfiguration:")
print(f"  Model: {MODEL_PATH}")
print(f"  Output: {OUTPUT_FILE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"  Compiler flags: --optlevel 3 --model-type transformer")

# Check if already compiled
if os.path.exists(OUTPUT_FILE):
    print(f"\n⚠ Warning: {OUTPUT_FILE} already exists")
    response = input("Overwrite? (yes/no): ")
    if response.lower() != 'yes':
        print("Compilation cancelled")
        exit(0)

# Load model
print("\nLoading model from HuggingFace...")
model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True
)
model.eval()
print("✓ Model loaded")

# Create example input
example_input = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
print(f"✓ Example input created: {example_input.shape}")

# Compile for Neuron
print("\nCompiling for Neuron (expected: 239 seconds)...")
start_time = time.time()

model_neuron = torch_neuronx.trace(
    model.vision_model,
    example_input,
    compiler_workdir='./neuron_compile_384',
    compiler_args=[
        '--verbose', 'error',
        '--optlevel', '3',
        '--model-type', 'transformer'
    ]
)

compile_time = time.time() - start_time
print(f"✓ Compilation complete in {compile_time:.1f} seconds")

# Save compiled model
print(f"\nSaving compiled model to {OUTPUT_FILE}...")
torch.jit.save(model_neuron, OUTPUT_FILE)
file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
print(f"✓ Model saved ({file_size_mb:.1f} MB)")

print("\n" + "="*80)
print("Compilation successful!")
print("="*80)
print(f"\nNext steps:")
print(f"  1. Test single core: python3 inference_single.py")
print(f"  2. Test dual core: ./run_dual_core.sh")
print(f"  3. Benchmark: python3 benchmark.py")
