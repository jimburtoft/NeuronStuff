#!/usr/bin/env python3
"""
Compile SigLIP 384px model for AWS Trainium2 with LNC=1 configuration.

This script compiles the model for LNC=1, which creates 8 logical cores
on trn2.3xlarge with smaller model footprint.

Usage:
    NEURON_LOGICAL_NC_CONFIG=1 python3 compile_model_lnc1.py

Expected performance: ~14.38 img/s single-core, 69.56 ms latency
"""

import torch
import torch_neuronx
import time
import os

print("=" * 80)
print("SigLIP 384px Model Compilation for Trainium2 - LNC=1")
print("=" * 80)

# Configuration
MODEL_PATH = "/home/ubuntu/timm/ViT-SO400M-14-SigLIP-384"
OUTPUT_FILE = "siglip_384_neuron_lnc1.pt"
BATCH_SIZE = 1
IMAGE_SIZE = 384

print(f"\nConfiguration:")
print(f"  Model: {MODEL_PATH}")
print(f"  Output: {OUTPUT_FILE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"  LNC: 1 (8 logical cores on trn2.3xlarge)")
print(
    f"  Compiler flags: --optlevel 3 --model-type transformer --lnc 1 --auto-cast matmult"
)
print(f"  Note: --auto-cast=matmult enables BF16 optimization for better performance")

# Check if already compiled
if os.path.exists(OUTPUT_FILE):
    print(f"\nWarning: {OUTPUT_FILE} already exists")
    response = input("Overwrite? (yes/no): ")
    if response.lower() != "yes":
        print("Compilation cancelled")
        exit(0)

# Load model
print("\nLoading model...")
import open_clip

state_dict = torch.load(f"{MODEL_PATH}/model.pt")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-SO400M-14-SigLIP-384", pretrained=False
)
model.load_state_dict(state_dict)
model.eval()
print(f"Model loaded ({len(state_dict)} parameters)")

# Get vision model
visual_model = model.visual
visual_model.eval()

# Create example input
example_input = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
print(f"Example input created: {example_input.shape}")

# Compile for Neuron
print("\nCompiling for Trainium2 with LNC=1...")
print("Expected compile time: ~200 seconds")
start_time = time.time()

model_neuron = torch_neuronx.trace(
    visual_model,
    example_input,
    compiler_workdir="./neuron_compile_lnc1",
    compiler_args=[
        "--verbose",
        "error",
        "--optlevel",
        "3",
        "--model-type",
        "transformer",
        "--lnc",
        "1",
        "--auto-cast",
        "matmult",
    ],
)

compile_time = time.time() - start_time
print(f"Compilation complete in {compile_time:.1f} seconds")

# Save compiled model
print(f"\nSaving compiled model to {OUTPUT_FILE}...")
torch.jit.save(model_neuron, OUTPUT_FILE)
file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
print(f"Model saved ({file_size_mb:.1f} MB)")

print("\n" + "=" * 80)
print("Compilation successful!")
print("=" * 80)
print(f"\nNext steps:")
print(
    f"  1. Test single core: NEURON_LOGICAL_NC_CONFIG=1 python3 inference_single_lnc1.py"
)
print(f"  2. Run multi-core: NEURON_LOGICAL_NC_CONFIG=1 ./run_multi_core_lnc1.sh")
print(f"  3. Compare: python3 benchmark_lnc_comparison.py")
