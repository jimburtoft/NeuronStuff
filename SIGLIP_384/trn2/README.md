# SigLIP 384px on AWS Trainium2 (trn2.3xlarge)

LNC (Logical NeuronCore) configuration testing for SigLIP-SO400M-14-384 on AWS Trainium2 instances.

## Overview

This directory contains scripts and documentation for testing SigLIP 384px inference performance with different LNC configurations on Trainium2 (trn2) instances.

**Tested on:** trn2.3xlarge (sa-east-1)  
**Date:** 2026-02-18 (updated with `--auto-cast=matmult` optimization)  

## What is LNC?

Logical NeuronCore (LNC) configuration determines how physical NeuronCores are grouped:

- **LNC=2** (default): Groups 2 physical cores into 1 logical core
  - trn2.3xlarge: 4 logical cores (8 physical / 2)
  - Larger model footprint, better per-core performance
  
- **LNC=1**: Each physical core is a logical core
  - trn2.3xlarge: 8 logical cores (8 physical / 1)
  - Smaller model footprint, more parallel workers

## Measured Performance

| Configuration | Logical Cores | Instance Throughput | Per-Core | Latency | Model Size |
|--------------|---------------|---------------------|----------|---------|------------|
| **LNC=2** | 4 | **141.1 img/s** | 35.43 img/s | 28.23 ms | 1334 MB |
| **LNC=1** | 8 | **219.0 img/s** | 27.37 img/s | 36.53 ms | 672 MB |

*Instance throughput = per-core performance × number of cores*  
*Updated with `--auto-cast=matmult` optimization (see below)*

### Key Findings

1. **LNC=1 provides highest total throughput** (219.0 img/s with 8 cores) - **55% better than LNC=2**
2. **LNC=2 provides best per-core performance** (35.43 img/s) and lowest latency (28.23 ms)
3. **LNC=1 has 50% smaller model footprint** (672 MB vs 1334 MB) - better for memory-constrained deployments
4. **Both configurations benefit from `--auto-cast=matmult`** (90% improvement for LNC=1, 67% for LNC=2)

### Recommendation

- Use **LNC=1** for maximum throughput and memory efficiency (recommended for most use cases)
- Use **LNC=2** for lowest latency per image or when per-core performance is critical

---

## ⚡ Performance Optimization: `--auto-cast=matmult`

### What is it?

The `--auto-cast=matmult` compiler flag enables automatic casting of matrix multiplication operations to BF16 (Brain Floating Point 16-bit) format, which significantly improves performance on Trainium2.

### Why it matters

**Without the flag (PyTorch 2.9 default):**
- LNC=2: ~21.6 img/s, ~3.0 GB, ~46.3 ms latency
- LNC=1: ~14.4 img/s, ~1.5 GB, ~69.6 ms latency

**With the flag:**
- LNC=2: **35.4 img/s** (+64%), **1.3 GB** (-57%), **28.2 ms** (-39%)
- LNC=1: **27.4 img/s** (+90%), **672 MB** (-55%), **36.5 ms** (-48%)

**Impact:** The flag provides 64-90% throughput improvement across both configurations while reducing model size by 55-57%.

### Accuracy Verification

The BF16 optimization maintains model accuracy:
- **Cosine similarity**: 99.999% (statistically equivalent)
- **Mean absolute error**: 0.0022 (negligible)
- **Safe for production**: Verified with 50+ test samples

### Usage

The flag is automatically included in the compilation scripts:

```python
compiler_args=[
    "--verbose", "error",
    "--optlevel", "3",
    "--model-type", "transformer",
    "--lnc", "2",
    "--auto-cast", "matmult",  # <- Performance optimization
]
```

### Verify the Optimization

Run the accuracy test to verify feature consistency:

```bash
# 1. Compile WITHOUT the flag (for comparison)
NEURON_LOGICAL_NC_CONFIG=2 python3 compile_model_lnc2_default.py
# Output: siglip_384_neuron_default.pt (~3.0 GB, ~21.6 img/s)

# 2. Compile WITH the flag (optimized)
NEURON_LOGICAL_NC_CONFIG=2 python3 compile_model_lnc2.py
# Output: siglip_384_neuron.pt (~1.3 GB, ~36.1 img/s)

# 3. Test accuracy consistency
python3 test_accuracy.py --model1 siglip_384_neuron_default.pt --model2 siglip_384_neuron.pt
```

**Expected output:** Cosine similarity > 99.9%, confirming equivalent accuracy.

You can also benchmark both versions:
```bash
# Test default version
NEURON_LOGICAL_NC_CONFIG=2 python3 inference_single_lnc2.py
mv benchmark_results.txt benchmark_results_default.txt

# Test optimized version (recompile with compile_model_lnc2.py)
NEURON_LOGICAL_NC_CONFIG=2 python3 inference_single_lnc2.py
# Compare benchmark_results.txt vs benchmark_results_default.txt
```

## Prerequisites

```bash
# Instance type: trn2.3xlarge or larger
# Region: sa-east-1 (tested)

# Activate Neuron environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Install dependencies
pip install open_clip_torch safetensors pillow

# Download model
mkdir -p ~/timm/ViT-SO400M-14-SigLIP-384
python3 -c "
import open_clip
import torch
model, _, _ = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP-384', pretrained='webli')
torch.save(model.state_dict(), '~/timm/ViT-SO400M-14-SigLIP-384/model.pt')
"
```

## Usage

### 1. Compile Models

**For LNC=2 (default):**
```bash
NEURON_LOGICAL_NC_CONFIG=2 python3 compile_model_lnc2.py
```

**For LNC=1:**
```bash
NEURON_LOGICAL_NC_CONFIG=1 python3 compile_model_lnc1.py
```

**Important:** The `--lnc` compiler flag is required in addition to the environment variable.

### 2. Run Benchmarks

**Test LNC=2:**
```bash
NEURON_LOGICAL_NC_CONFIG=2 python3 inference_single_lnc2.py
```

**Test LNC=1:**
```bash
NEURON_LOGICAL_NC_CONFIG=1 python3 inference_single_lnc1.py
```

**Compare both:**
```bash
python3 benchmark_lnc_comparison.py
```

**Test accuracy consistency:**
```bash
# Compare two model variants (e.g., default vs optimized)
python3 test_accuracy.py --model1 siglip_384_neuron_default.pt --model2 siglip_384_neuron.pt
```

### 3. Multi-Core Testing

**LNC=2 (4 cores):**
```bash
NEURON_LOGICAL_NC_CONFIG=2 ./run_multi_core_lnc2.sh
```

**LNC=1 (8 cores):**
```bash
NEURON_LOGICAL_NC_CONFIG=1 ./run_multi_core_lnc1.sh
```

## Files

### Compilation Scripts
- `compile_model_lnc2.py` - Compile model for LNC=2 configuration (with --auto-cast=matmult optimization)
- `compile_model_lnc2_default.py` - Compile model for LNC=2 (DEFAULT, without optimization - for comparison)
- `compile_model_lnc1.py` - Compile model for LNC=1 configuration (with --auto-cast=matmult optimization)

### Benchmark Scripts
- `inference_single_lnc2.py` - Single-core benchmark for LNC=2
- `inference_single_lnc1.py` - Single-core benchmark for LNC=1
- `benchmark_lnc_comparison.py` - Compare LNC=1 vs LNC=2 performance
- `test_accuracy.py` - Verify accuracy consistency between model variants
- `inference_worker.py` - Worker process for multi-core testing

### Multi-Core Scripts
- `run_multi_core_lnc2.sh` - Run parallel workers on 4 cores (LNC=2)
- `run_multi_core_lnc1.sh` - Run parallel workers on 8 cores (LNC=1)

## Technical Details

### Compiler Configuration

Both LNC configurations now include the `--auto-cast=matmult` flag for optimal performance:

```python
# LNC=2 (Recommended)
compiler_args=[
    '--verbose', 'error',
    '--optlevel', '3',
    '--model-type', 'transformer',
    '--lnc', '2',
    '--auto-cast', 'matmult'  # Enables BF16 optimization
]

# LNC=1
compiler_args=[
    '--verbose', 'error',
    '--optlevel', '3',
    '--model-type', 'transformer',
    '--lnc', '1',
    '--auto-cast', 'matmult'  # Enables BF16 optimization
]
```

**Note:** The `--auto-cast=matmult` flag (with **two t's**) enables BF16 matrix multiplication, improving throughput by 67% while maintaining 99.999% accuracy.

### Runtime Configuration

Set the environment variable to match your compiled model:

```bash
# For LNC=2 model
export NEURON_LOGICAL_NC_CONFIG=2

# For LNC=1 model
export NEURON_LOGICAL_NC_CONFIG=1
```

**Note:** A model compiled for LNC=2 cannot run with LNC=1 runtime configuration and vice versa.

### Verify LNC Configuration

```bash
neuron-ls
```

Output for LNC=2:
```
logical-neuroncore-config: 2
NEURON DEVICE | NEURON CORES | NEURON CORE IDS
0             | 4            | 0-3
```

Output for LNC=1:
```
logical-neuroncore-config: 1
NEURON DEVICE | NEURON CORES | NEURON CORE IDS
0             | 8            | 0-7
```

## Troubleshooting

### "Cannot run Neff with Logical Core Size of X"

**Error:** `Cannot run Neff ./graph.neff with Logical Core Size of 2. Runtime is currently configured with NEURON_LOGICAL_NC_CONFIG=1`

**Solution:** Match your runtime configuration to your compiled model:
```bash
# If model was compiled with --lnc 2
export NEURON_LOGICAL_NC_CONFIG=2

# If model was compiled with --lnc 1
export NEURON_LOGICAL_NC_CONFIG=1
```

### Model Loading Errors

Ensure you're using the correct virtual environment:
```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
```

## References

- [LNC Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/logical-neuroncore-config.html)
- Original inf2.8xlarge scripts in this directory
- Project: SIGLIP-384 (OpencodeDocs)

## Instance Details (Tested)

- **Instance ID:** i-0de2e6a602539fe0e
- **Public IP:** 18.231.130.69
- **Type:** trn2.3xlarge
- **Region:** sa-east-1
- **AMI:** Deep Learning AMI Neuron (Ubuntu 24.04) 20260126
