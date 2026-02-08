#!/usr/bin/env python3
"""
Run ViT-SO400M-14-SigLIP model on AWS Trainium2 using Neuron SDK
"""
import torch
import torch_neuronx
from PIL import Image
from urllib.request import urlopen
import time
import os
from safetensors.torch import load_file
import open_clip

# Set environment variables for better logging
os.environ['NEURON_RT_LOG_LEVEL'] = 'INFO'

def main():
    print("=" * 80)
    print("ViT-SO400M-14-SigLIP on AWS Trainium2")
    print("=" * 80)
    
    model_path = '/home/ubuntu/timm/ViT-SO400M-14-SigLIP'
    
    # Load model
    print("\n1. Loading model...")
    try:
        # Load the safetensors file
        state_dict = load_file(f'{model_path}/open_clip_model.safetensors')
        print(f"✓ Loaded {len(state_dict)} parameters from safetensors")
        
        # Create model with correct architecture (224 resolution, not 384)
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-SO400M-14-SigLIP',
            pretrained=False
        )
        
        # Load the state dict
        model.load_state_dict(state_dict)
        model.eval()
        print("✓ Model loaded successfully")
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Prepare test image
    print("\n2. Preparing test image...")
    try:
        image_url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
        image = Image.open(urlopen(image_url))
        print(f"✓ Image loaded: {image.size}")
        
        # Preprocess image
        image_tensor = preprocess(image).unsqueeze(0)
        print(f"✓ Input tensor shape: {image_tensor.shape}")
        
    except Exception as e:
        print(f"✗ Failed to prepare image: {e}")
        return
    
    # Run baseline inference on CPU
    print("\n3. Running baseline inference on CPU...")
    try:
        with torch.no_grad():
            start_time = time.time()
            cpu_output = model.encode_image(image_tensor)
            cpu_time = time.time() - start_time
        
        print(f"✓ CPU inference completed in {cpu_time:.4f}s")
        print(f"   Output shape: {cpu_output.shape}")
        print(f"   Output range: [{cpu_output.min():.4f}, {cpu_output.max():.4f}]")
        print(f"   First 5 values: {cpu_output[0, :5]}")
        
    except Exception as e:
        print(f"✗ CPU inference failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Compile model for Neuron
    print("\n4. Compiling model for Neuron...")
    print("   This will take several minutes (10-30 min for large models)...")
    print("   Compilation artifacts will be saved to ./neuron_compile_siglip/")
    
    try:
        # Extract just the visual encoder for compilation
        visual_model = model.visual
        visual_model.eval()
        
        # Trace the model for Neuron
        model_neuron = torch_neuronx.trace(
            visual_model,
            image_tensor,
            compiler_workdir='./neuron_compile_siglip',
            compiler_args=[
                '--verbose', 'info',
                '--model-type', 'transformer'
            ]
        )
        
        # Save compiled model
        compiled_model_path = 'siglip_visual_neuron.pt'
        torch.jit.save(model_neuron, compiled_model_path)
        print(f"✓ Model compiled and saved to {compiled_model_path}")
        
    except Exception as e:
        print(f"✗ Model compilation failed: {e}")
        print("\nCompilation error details:")
        import traceback
        traceback.print_exc()
        print("\nNote: Vision transformers may have operations not fully supported by Neuron.")
        print("      Check the compiler logs in ./neuron_compile_siglip/ for details.")
        return
    
    # Run inference on Neuron
    print("\n5. Running inference on Neuron...")
    try:
        with torch.no_grad():
            start_time = time.time()
            neuron_output = model_neuron(image_tensor)
            neuron_time = time.time() - start_time
        
        print(f"✓ Neuron inference completed in {neuron_time:.4f}s")
        print(f"   Output shape: {neuron_output.shape}")
        print(f"   Output range: [{neuron_output.min():.4f}, {neuron_output.max():.4f}]")
        print(f"   First 5 values: {neuron_output[0, :5]}")
        
        # Compare outputs
        diff = torch.abs(cpu_output - neuron_output).max().item()
        mean_diff = torch.abs(cpu_output - neuron_output).mean().item()
        
        print(f"\n   Accuracy comparison:")
        print(f"   - Max difference: {diff:.6f}")
        print(f"   - Mean difference: {mean_diff:.6f}")
        
        if diff < 0.01:
            print("   ✓ Outputs match within tolerance")
        else:
            print("   ⚠ Outputs differ (may be due to precision differences)")
        
        # Performance comparison
        print(f"\n   Performance:")
        print(f"   - CPU time: {cpu_time:.4f}s")
        print(f"   - Neuron time: {neuron_time:.4f}s")
        if neuron_time < cpu_time:
            speedup = cpu_time / neuron_time
            print(f"   - Speedup: {speedup:.2f}x")
        else:
            print(f"   - Note: First inference includes model loading overhead")
        
    except Exception as e:
        print(f"✗ Neuron inference failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run multiple inferences to measure steady-state performance
    print("\n6. Running warmup and performance test...")
    try:
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model_neuron(image_tensor)
        
        # Measure
        num_runs = 10
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model_neuron(image_tensor)
        total_time = time.time() - start_time
        avg_time = total_time / num_runs
        
        print(f"✓ Average inference time over {num_runs} runs: {avg_time:.4f}s")
        print(f"   Throughput: {1/avg_time:.2f} images/second")
        
    except Exception as e:
        print(f"⚠ Performance test failed: {e}")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)

if __name__ == '__main__':
    main()
