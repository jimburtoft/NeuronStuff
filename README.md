# SigLIP 384px Production Deployment

High-performance inference for SigLIP-SO400M-14-384 on AWS inf2.8xlarge.

## Measured Performance

- **Single core:** 13.38 img/s, 74.74 ms latency
- **DataParallel (2 cores):** 20.68 img/s with batch size 2
- **Separate processes:** Each core runs at ~13.3 img/s independently

## Quick Start

### 1. Compile Model (one-time setup)

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python3 compile_model.py
```

This creates `siglip_384_neuron.pt` (1.5GB, 239 seconds compile time).

### 2. Run Inference

**Single Core:**
```bash
python3 inference_single.py
```

**Dual Core:**
```bash
./run_dual_core.sh
```

## Files

- `compile_model.py` - Compile SigLIP 384px for Neuron
- `inference_single.py` - Single-core inference
- `inference_worker.py` - Worker for multi-core deployment
- `run_dual_core.sh` - Run dual-core inference
- `benchmark.py` - Performance benchmarking tool

## Configuration

**Model:** `/home/ubuntu/timm/ViT-SO400M-14-SigLIP-384`

**Optimal settings (tested):**
- Batch size: 1
- Compiler flags: `--optlevel 3 --model-type transformer`
- Input shape: [1, 3, 384, 384]

## Requirements

- AWS inf2.8xlarge instance
- Virtual environment: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`
- PyTorch with Neuron support
- Model weights in `/home/ubuntu/timm/ViT-SO400M-14-SigLIP-384`
