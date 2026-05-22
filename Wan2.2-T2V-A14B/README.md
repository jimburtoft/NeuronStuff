# Wan 2.2 T2V-A14B Video Generation on AWS Trainium2

Generate 768x1280 (81 frames, ~5s at 16fps) video from text prompts using the Wan 2.2 T2V-A14B Mixture-of-Experts diffusion model on a trn2.48xlarge instance.

## Model

- **Name:** [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)
- **Architecture:** 27B parameter MoE with 14B active per denoising step
- **Two separate experts:** Expert 1 handles high-noise steps, Expert 2 handles low-noise steps. They share zero weights.
- **Weights:** ~118 GB (downloaded from Hugging Face)

## Performance

Measured on trn2.48xlarge (16 NeuronDevices, 64 NeuronCores with LNC=2) with Neuron SDK 2.29.1:

### Optimized (CP=16, Batched CFG)

| Phase | Time |
|-------|------|
| Pipeline load | ~3s |
| Text encoding (Neuron, TP=4) | 22s |
| Expert 1 load (TP=4, CP=16, 64 cores) | 179s |
| Expert 1 denoise (13 steps) | 67s |
| Expert 2 load (TP=4, CP=16, 64 cores) | 179s |
| Expert 2 denoise (27 steps) | 136s |
| VAE decode (Neuron tiled, 8 NCs) | ~35s |
| **Total wall time** | **~618s (~10.3 min)** |

- **Per-step average:** 5.06s (single batched forward pass with CFG, batch_size=2)
- **Per forward pass:** 5.06s (batch=2: conditional + unconditional in one pass)
- **Per-sample equivalent:** 2.53s (=5.06/2, computing one direction)
- **Resolution:** 768x1280, 81 frames, 40 denoising steps
- **Parallelism:** TP=4, CP=16 (world_size=64) per expert — all 64 NeuronCores
- **Compiler flags:** `-O1 --auto-cast=none --enable-native-kernel=1 --remat` + `--enable-ccop-compute-overlap`

### Baseline (CP=4, Sequential CFG)

| Phase | Time |
|-------|------|
| Expert 1 load (TP=4, CP=4, 16 cores) | 218s |
| Expert 1 denoise (13 steps) | 223s |
| Expert 2 load (TP=4, CP=4, 16 cores) | 216s |
| Expert 2 denoise (27 steps) | 463s |
| **Total wall time** | **~1224s (~20.4 min)** |

- **Per-step average:** 17.1s (2 sequential forward passes for CFG, batch_size=1)
- **Per forward pass:** 8.54s (batch=1, single direction)

### Optimization Impact

| Optimization | Per-step | Speedup |
|-------------|----------|---------|
| Baseline (CP=4, BS=1) | 17.1s | 1.0x |
| + CP=16 (4x more cores) | ~8.5s | ~2.0x |
| + Batched CFG (BS=2) | **5.06s** | **3.4x** |

### GPU Comparison

| Config | Time (s) | Notes |
|--------|----------|-------|
| **trn2.48xlarge (optimized)** | **~618** | CP=16, batch=2, all 64 cores |
| trn2.48xlarge (baseline) | ~1224 | CP=4, batch=1, 16 cores only |
| 1x H100 | 1042 | Warm start, 720P |
| 4x H100 | 289 | Warm start, 720P |
| 1x A100 | 2736 | Warm start, 720P |
| 4x A100 | 725 | Warm start, 720P |

GPU numbers from official Wan 2.2 benchmarks (warm start, 720P).

### Further Optimization Paths

1. **replace_weights() fix:** SDK SIGSEGV forces separate processes (2x model load). Fixing saves ~179s (15% E2E)
2. **Persistent/warm start:** Keep model loaded across requests — eliminates load time entirely
3. **Combine CP=16 + persistent:** Projected per-request time ~240s (denoising 203s + text 22s + VAE 35s) = **4 min**

## Files

| File | Description |
|------|-------------|
| `wan22_t2v_a14b_trn2.ipynb` | Source notebook (no outputs) |
| `wan22_t2v_a14b_trn2_executed.ipynb` | Executed notebook with all outputs |
| `worker_denoise.py` | Subprocess worker for expert denoising |

## Quick Start

Open and run `wan22_t2v_a14b_trn2.ipynb` on a trn2.48xlarge instance. The notebook covers:

1. Environment setup (NVMe mount, dependencies)
2. Model download (118 GB from Hugging Face)
3. Compilation of all four components (~15 min from scratch):
   - Text encoder
   - Transformer Expert 1 (high noise)
   - Transformer Expert 2 (low noise)
   - Tiled VAE decoder
4. End-to-end inference with timing breakdown

## Requirements

- **Instance:** trn2.48xlarge (16 NeuronDevices required)
- **LNC:** 2 (default, gives 64 logical cores with 24 GB HBM each)
- **AMI:** Deep Learning AMI Neuron (Ubuntu 24.04) 20260502 (SDK 2.29.1)
- **Venv:** `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`
- **Disk:** 300+ GB EBS + NVMe mount for model weights and compiled artifacts
- **diffusers:** 0.38.0+ (required for `WanPipeline` with Wan2.2 MoE support)

## Architecture Notes

The A14B MoE model uses two completely independent transformer experts. Because they share no weights, each must be loaded/unloaded separately. The pipeline uses subprocess isolation for clean HBM lifecycle:

1. Loads Expert 1 (TP=4, CP=16 on all 64 cores), runs 13 high-noise denoising steps with batched CFG
2. Subprocess exits (cleanly frees HBM), loads Expert 2
3. Runs 27 low-noise denoising steps with Expert 2 (same 64 cores), batched CFG
4. Decodes latents to video with the tiled VAE decoder (8 tiles on 8 NCs)

Context Parallelism (CP=16) splits the 80,640-token sequence across 16 ranks. Combined with batched CFG (batch_size=2), both classifier-free guidance passes are computed in a single forward call.

The compilation code is from [whn09/aws-neuron-samples](https://github.com/whn09/aws-neuron-samples) (torch-neuronx/inference/hf_pretrained_wan2.2_t2v_a14b/).

## SDK Version History

| SDK | DLAMI | Status | Notes |
|-----|-------|--------|-------|
| 2.29.1 | 20260502 | **Current** | neuronx-cc 2.24.8799.0, NKI 0.3.0 GA |
| 2.28 | 20260227 | Previous | Original implementation |
