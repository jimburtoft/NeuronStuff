#!/usr/bin/env python3
"""
Compile SigLIP 384px model for AWS Trainium2 with LNC=2 (DEFAULT settings).

This script compiles the model WITHOUT the --auto-cast=matmult flag for comparison
purposes. Use this to compare against the optimized version.

Usage:
    NEURON_LOGICAL_NC_CONFIG=2 python3 compile_model_lnc2_default.py

Expected performance: ~21.6 img/s single-core, 46.3 ms latency
Model size: ~3.0 GB

Compare with compile_model_lnc2.py (with --auto-cast=matmult):
- Optimized: 36.1 img/s, 27.7 ms latency, 1.3 GB
"""

import torch
import torch_neuronx
import time
import os

print("=" * 80)
print("SigLIP 384px Model Compilation for Trainium2 - LNC=2 (DEFAULT)")
print("=" * 80)
print("\n⚠️  This compiles WITHOUT --auto-cast=matmult for comparison purposes")
print("    For production use, use compile_model_lnc2.py instead")
print("=" * 80)

# Configuration
MODEL_PATH = "/home/ubuntu/timm/ViT-SO400M-14-SigLIP-384"
OUTPUT_FILE = "siglip_384_neuron_default.pt"
BATCH_SIZE = 1
IMAGE_SIZE = 384

print(f"\nConfiguration:")
print(f"  Model: {MODEL_PATH}")
print(f"  Output: {OUTPUT_FILE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"  LNC: 2 (4 logical cores on trn2.3xlarge)")
print(f"  Compiler flags: --optlevel 3 --model-type transformer --lnc 2")
print(f"  ⚠️  NO --auto-cast flag (PyTorch 2.9 default)")

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

# Compile for Neuron (WITHOUT --auto-cast flag)
print("\nCompiling for Trainium2 with LNC=2 (DEFAULT settings)...")
print("Expected compile time: ~380 seconds")
print("Expected model size: ~3.0 GB")
start_time = time.time()

model_neuron = torch_neuronx.trace(
    visual_model,
    example_input,
    compiler_workdir="./neuron_compile_lnc2_default",
    compiler_args=[
        "--verbose",
        "error",
        "--optlevel",
        "3",
        "--model-type",
        "transformer",
        "--lnc",
        "2",
        # NOTE: No --auto-cast flag - using PyTorch 2.9 default
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
print("Compilation successful! (DEFAULT settings)")
print("=" * 80)
print(f"\nNext steps:")
print(f"  1. Compare with optimized version:")
print(
    f"     python3 test_accuracy.py --model1 {OUTPUT_FILE} --model2 siglip_384_neuron.pt"
)
print(f"\n  2. For production use, compile with optimization:")
print(f"     NEURON_LOGICAL_NC_CONFIG=2 python3 compile_model_lnc2.py")
