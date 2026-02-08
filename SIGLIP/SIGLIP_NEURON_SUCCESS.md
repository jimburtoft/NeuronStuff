# ViT-SO400M-14-SigLIP on AWS Trainium2 - Success Report

## Summary

Successfully ran the ViT-SO400M-14-SigLIP vision transformer model on AWS Trainium2 using the Neuron SDK. The model was compiled and executed with excellent accuracy and performance.

## Test Results

### Model Information
- **Model**: ViT-SO400M-14-SigLIP (400M parameter vision transformer)
- **Architecture**: Vision Transformer with SigLIP training
- **Input Size**: 224x224 RGB images
- **Output**: 1152-dimensional embeddings

### Compilation
- **Status**: ✓ Successful
- **Compiler**: torch_neuronx.trace()
- **Artifacts**: Saved to `./neuron_compile_siglip/`
- **Compiled Model**: `siglip_visual_neuron.pt`

### Accuracy
- **Max Difference**: 0.000024 (CPU vs Neuron)
- **Mean Difference**: 0.000001
- **Status**: ✓ Outputs match within tolerance

### Performance

#### CPU Baseline
- **Inference Time**: 0.2566s per image
- **Throughput**: ~3.9 images/second

#### Neuron (Trainium2)
- **First Inference**: 29.29s (includes model loading overhead)
- **Steady-State**: 0.0120s per image (after warmup)
- **Throughput**: 83.26 images/second
- **Speedup**: ~21x faster than CPU

### Hardware Utilization
- **Neuron Cores Used**: 2 cores (NC4, NC5 on ND0)
- **Memory Usage**: 1.632GB per core
  - Model code: 10.6MB
  - Model constants: 1.615GB
  - DMA rings: 4.9MB
  - Runtime: 190KB

## Files Created

1. **run_siglip_neuron_v2.py** - Main test script with compilation and inference
2. **siglip_visual_neuron.pt** - Compiled model for Neuron
3. **siglip_neuron_test.log** - Full execution log
4. **neuron_compile_siglip/** - Compilation artifacts directory

## Key Findings

1. **Compilation Success**: The vision transformer architecture is well-supported by Neuron SDK
2. **Numerical Accuracy**: Excellent match between CPU and Neuron outputs (< 0.00003 difference)
3. **Performance**: Significant speedup (21x) over CPU inference after warmup
4. **Memory Efficiency**: Model fits comfortably in Neuron device memory

## Usage Instructions

### Prerequisites
```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
```

### Run Inference
```bash
python run_siglip_neuron_v2.py
```

### Load Pre-compiled Model
```python
import torch

# Load compiled model
model_neuron = torch.jit.load('siglip_visual_neuron.pt')
model_neuron.eval()

# Run inference
with torch.no_grad():
    output = model_neuron(input_tensor)
```

## Technical Details

### Neuron SDK Configuration
- **Runtime Version**: 2.29.40.0
- **Firmware Version**: 1.19.1.0
- **Driver Version**: 2.25.4.0
- **Visible Cores**: 64 (0-63)

### Model Architecture
- **Embedding Dimension**: 1152
- **Image Size**: 224x224
- **Patch Size**: 14x14
- **Parameters**: ~400M

## Recommendations

1. **Batch Processing**: For higher throughput, consider batching multiple images
2. **Model Caching**: Reuse the compiled model (`siglip_visual_neuron.pt`) to avoid recompilation
3. **Warmup**: Always run 2-3 warmup inferences before measuring performance
4. **Memory**: Model uses ~1.6GB per core, leaving plenty of room for larger batch sizes

## Next Steps

Potential optimizations:
- Test with larger batch sizes
- Explore multi-core deployment for parallel processing
- Integrate with text encoder for full CLIP functionality
- Benchmark against other instance types

## Conclusion

The ViT-SO400M-14-SigLIP model runs successfully on AWS Trainium2 with excellent accuracy and performance. The 21x speedup over CPU makes it suitable for production workloads requiring high-throughput image embedding generation.
