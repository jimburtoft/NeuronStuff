# SigLIP 384px Production Deployment

High-performance inference for SigLIP-SO400M-14-384 on AWS inf2.8xlarge.

## Measured Performance

### inf2.8xlarge (Inferentia2)
- **Single core:** 13.38 img/s, 74.74 ms latency
- **DataParallel (2 cores):** 20.68 img/s with batch size 2
- **Separate processes:** Each core runs at ~13.3 img/s independently

### trn2.3xlarge (Trainium2 with LNC)
Tested 2026-02-17 with LNC (Logical NeuronCore) configuration:

| Configuration | Logical Cores | Per-Core | Instance Throughput | Latency |
|--------------|---------------|----------|---------------------|---------|
| **LNC=2** | 4 | 21.95 img/s | **87.8 img/s** | 45.56 ms |
| **LNC=1** | 8 | 14.38 img/s | **115.0 img/s** | 69.56 ms |

**Key Findings:**
- LNC=1 provides **31% higher** total instance throughput (115.0 vs 87.8 img/s)
- LNC=2 has **34% lower** latency per image
- LNC=1 utilizes all 8 cores; LNC=2 uses 4 cores

**Recommendation:** 
- Use **LNC=1** for maximum instance throughput (batch processing, high-load scenarios)
- Use **LNC=2** for low-latency requirements (real-time inference)

See [trn2/README.md](trn2/README.md) for Trainium2-specific details and scripts.

## Quick Start

### 1. Setup Environment

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
pip install open_clip_torch
```

### 2. Download Model

```bash
huggingface-cli download timm/ViT-SO400M-14-SigLIP-384 --local-dir timm/ViT-SO400M-14-SigLIP-384/
```

### 3. Compile Model (one-time setup)

```bash
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

### For Trainium2 (trn2.3xlarge)

See the [`trn2/`](trn2/) subdirectory for Trainium2-specific scripts and LNC configuration:

```bash
cd trn2/
# Follow instructions in trn2/README.md
```

## Files

### inf2.8xlarge (Original)
- `compile_model.py` - Compile SigLIP 384px for Neuron
- `inference_single.py` - Single-core inference
- `inference_worker.py` - Worker for multi-core deployment
- `run_dual_core.sh` - Run dual-core inference
- `benchmark.py` - Performance benchmarking tool

### trn2.3xlarge (Trainium2 with LNC Testing)
Located in `trn2/` subdirectory:
- `trn2/README.md` - Complete guide for Trainium2 LNC configuration
- `trn2/compile_model_lnc2.py` - Compile for LNC=2 (4 cores, max throughput: 87.8 img/s)
- `trn2/compile_model_lnc1.py` - Compile for LNC=1 (8 cores, max throughput: 115.0 img/s)
- `trn2/inference_single_lnc2.py` - Benchmark LNC=2 single-core
- `trn2/inference_single_lnc1.py` - Benchmark LNC=1 single-core
- `trn2/benchmark_lnc_comparison.py` - Compare LNC configurations
- `trn2/run_multi_core_lnc2.sh` - Multi-core test with 4 cores
- `trn2/run_multi_core_lnc1.sh` - Multi-core test with 8 cores

## Configuration

**Model:** `/home/ubuntu/timm/ViT-SO400M-14-SigLIP-384`

**Optimal settings (tested):**
- Batch size: 1
- Compiler flags: `--optlevel 3 --model-type transformer`
- Input shape: [1, 3, 384, 384]

## Requirements

**For inf2.8xlarge (Inferentia2):**
- AWS inf2.8xlarge instance
- Virtual environment: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`
- PyTorch with Neuron support
- Model weights in `/home/ubuntu/timm/ViT-SO400M-14-SigLIP-384`

**For trn2.3xlarge (Trainium2):**
- AWS trn2.3xlarge instance (or larger)
- Same virtual environment and dependencies
- See [trn2/README.md](trn2/README.md) for LNC-specific setup and scripts
