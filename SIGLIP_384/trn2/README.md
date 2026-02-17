# SigLIP 384px on AWS Trainium2 (trn2.3xlarge)

LNC (Logical NeuronCore) configuration testing for SigLIP-SO400M-14-384 on AWS Trainium2 instances.

## Overview

This directory contains scripts and documentation for testing SigLIP 384px inference performance with different LNC configurations on Trainium2 (trn2) instances.

**Tested on:** trn2.3xlarge (sa-east-1)  
**Date:** 2026-02-17  

## What is LNC?

Logical NeuronCore (LNC) configuration determines how physical NeuronCores are grouped:

- **LNC=2** (default): Groups 2 physical cores into 1 logical core
  - trn2.3xlarge: 4 logical cores (8 physical / 2)
  - Larger model footprint, better per-core performance
  
- **LNC=1**: Each physical core is a logical core
  - trn2.3xlarge: 8 logical cores (8 physical / 1)
  - Smaller model footprint, more parallel workers

## Measured Performance

| Configuration | Logical Cores | Instance Throughput | Per-Core | Latency |
|--------------|---------------|---------------------|----------|---------|
| **LNC=2** | 4 | **87.8 img/s** | 21.95 img/s | 45.56 ms |
| **LNC=1** | 8 | **115.0 img/s** | 14.38 img/s | 69.56 ms |

*Instance throughput = per-core performance × number of cores*

### Key Findings

1. **LNC=1 provides 31% higher instance throughput** (115.0 vs 87.8 img/s)
   - 8 cores × 14.38 img/s = 115.0 img/s total
   - 4 cores × 21.95 img/s = 87.8 img/s total
2. **LNC=2 has 34% lower latency** per image (45.56ms vs 69.56ms)
3. **LNC=1 utilizes all 8 cores** vs 4 cores with LNC=2

### Recommendation

- Use **LNC=1** for maximum instance throughput (batch processing, high-load scenarios)
- Use **LNC=2** for low-latency requirements (real-time inference)

## Prerequisites

```bash
# Instance type: trn2.3xlarge or larger
# Region: sa-east-1 (tested)

# Activate Neuron environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Install dependencies
pip install open_clip_torch safetensors

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
- `compile_model_lnc2.py` - Compile model for LNC=2 configuration
- `compile_model_lnc1.py` - Compile model for LNC=1 configuration

### Benchmark Scripts
- `inference_single_lnc2.py` - Single-core benchmark for LNC=2
- `inference_single_lnc1.py` - Single-core benchmark for LNC=1
- `benchmark_lnc_comparison.py` - Compare LNC=1 vs LNC=2 performance
- `inference_worker.py` - Worker process for multi-core testing

### Multi-Core Scripts
- `run_multi_core_lnc2.sh` - Run parallel workers on 4 cores (LNC=2)
- `run_multi_core_lnc1.sh` - Run parallel workers on 8 cores (LNC=1)

## Technical Details

### Compiler Configuration

Both LNC configurations require the `--lnc` flag:

```python
# LNC=2
compiler_args=[
    '--verbose', 'error',
    '--optlevel', '3',
    '--model-type', 'transformer',
    '--lnc', '2'
]

# LNC=1
compiler_args=[
    '--verbose', 'error',
    '--optlevel', '3',
    '--model-type', 'transformer',
    '--lnc', '1'
]
```

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
