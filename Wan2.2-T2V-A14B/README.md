# Wan 2.2 T2V-A14B Video Generation on AWS Trainium2

Generate 768x1280 (81 frames, ~5s at 16fps) video from text prompts using the Wan 2.2 T2V-A14B Mixture-of-Experts diffusion model on a trn2.48xlarge instance.

## Model

- **Name:** [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)
- **Architecture:** 27B parameter MoE with 14B active per denoising step
- **Two separate experts:** Expert 1 handles high-noise steps, Expert 2 handles low-noise steps. They share zero weights.
- **Weights:** ~118 GB (downloaded from Hugging Face)

## Performance

Measured on trn2.48xlarge (16 NeuronDevices, 64 NeuronCores with LNC=2) with Neuron SDK 2.29.1:

| Phase | Time |
|-------|------|
| Pipeline load | ~3s |
| Text encoding (Neuron, subprocess) | 22.1s |
| Expert 1 load (TP=4, CP=4, 16 cores) | 218s |
| Expert 1 denoise (13 steps) | 223s |
| Expert 2 load (TP=4, CP=4, 16 cores) | 216s |
| Expert 2 denoise (27 steps) | 463s |
| VAE decode (Neuron tiled, 8 NCs) | 35s |
| **Total wall time** | **~1224s (~20.4 min)** |

- **Per-step average:** 17.1s (2 sequential forward passes for CFG, batch_size=1)
- **Per forward pass:** 8.54s (steady state)
- **Resolution:** 768x1280, 81 frames, 40 denoising steps
- **Parallelism:** TP=4, CP=4 (world_size=16) per expert — 16 of 64 NeuronCores
- **Compiler flags:** `-O1 --auto-cast=none --enable-native-kernel=1 --remat`

### GPU Comparison

| Config | Time (s) | Notes |
|--------|----------|-------|
| **trn2.48xlarge (ours)** | **~1224** | TP=4, CP=4, batch=1, 16 cores only |
| 1x H100 | 1042 | Warm start, 720P |
| 4x H100 | 289 | Warm start, 720P |
| 1x A100 | 2736 | Warm start, 720P |
| 4x A100 | 725 | Warm start, 720P |


GPU numbers from official Wan 2.2 benchmarks (warm start, 720P).

### Optimization Opportunities

The current result (20.4 min) is a baseline. Key optimization paths:

1. **Batched CFG (batch_size=2):** Reduces per-step from 17.1s to ~9-10s (model compiled for batch=1, needs recompile)
2. **Higher CP degree (CP=16):** Use all 64 cores for ~4x theoretical speedup on denoising (3.35x empirical)
3. **replace_weights() fix:** Currently uses separate processes (2x model load overhead). Fixing the SIGSEGV saves ~216s
4. **Compiler flag tuning:** Test `-O2 --enable-ccop-compute-overlap` vs current `-O1`
5. **Warm start:** Model stays loaded across requests (persistent mode from upstream)

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

1. Loads Expert 1 (TP=4, CP=4 on 16 cores), runs 13 high-noise denoising steps
2. Subprocess exits (cleanly frees HBM), loads Expert 2
3. Runs 27 low-noise denoising steps with Expert 2 (same 16 cores)
4. Decodes latents to video with the tiled VAE decoder (8 tiles on 8 NCs)

Context Parallelism (CP=4) splits the 80,640-token sequence across 4 ranks. Higher CP degrees (CP=16 using all 64 cores) would provide further speedup and are an optimization target.

The compilation code is from [whn09/aws-neuron-samples](https://github.com/whn09/aws-neuron-samples) (torch-neuronx/inference/hf_pretrained_wan2.2_t2v_a14b/).

## SDK Version History

| SDK | DLAMI | Status | Notes |
|-----|-------|--------|-------|
| 2.29.1 | 20260502 | **Current** | neuronx-cc 2.24.8799.0, NKI 0.3.0 GA |
| 2.28 | 20260227 | Previous | Original implementation |
