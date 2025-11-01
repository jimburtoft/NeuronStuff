#!/usr/bin/env python3
"""
Quick test to verify Neuron integration works with OpenEMMA.

This tests the openemma_neuron module independently before integrating into main.py.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate
    python test_neuron_integration.py
"""

import os
import sys
from PIL import Image

# Test imports
print("Testing imports...")
try:
    from openemma_neuron import load_neuron_model, neuron_vlm_inference, NEURON_AVAILABLE
    print("✓ openemma_neuron module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import openemma_neuron: {e}")
    sys.exit(1)

if not NEURON_AVAILABLE:
    print("✗ Neuron libraries not available")
    print("  Activate Neuron venv: source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate")
    sys.exit(1)

print("✓ Neuron libraries available")

# Test model loading
print("\nTesting model loading...")
try:
    model, processor, tokenizer = load_neuron_model("meta-llama/Llama-3.2-11B-Vision-Instruct")
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test inference
print("\nTesting inference...")
test_image = "/home/ubuntu/OpenEmmaInference/nuscenes/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg"

if not os.path.exists(test_image):
    print(f"✗ Test image not found: {test_image}")
    print("  Please update the path to a valid image")
    sys.exit(1)

try:
    prompt = "Describe what you see in this driving scene."
    print(f"  Prompt: {prompt}")
    print(f"  Image: {test_image}")
    
    result = neuron_vlm_inference(
        text=prompt,
        images=test_image,
        model=model,
        processor=processor,
        max_new_tokens=128
    )
    
    print(f"\n✓ Inference completed successfully")
    print(f"\nResponse:\n{result}")
    
except Exception as e:
    print(f"✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✓✓✓ ALL TESTS PASSED ✓✓✓")
print("="*80)
print("\nNeuron integration is working correctly!")
print("You can now integrate into main.py following NEURON_INTEGRATION_GUIDE.md")
