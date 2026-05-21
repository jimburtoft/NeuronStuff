# Wan 2.2 T2V-A14B Video Generation on AWS Trainium2

Generate 768x1280 (81 frames, ~5s at 16fps) video from text prompts using the Wan 2.2 T2V-A14B Mixture-of-Experts diffusion model on a trn2.48xlarge instance.

## Model

- **Name:** [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)
- **Architecture:** 27B parameter MoE with 14B active per denoising step
- **Two separate experts:** Expert 1 handles high-noise steps (1-16), Expert 2 handles low-noise steps (17-50). They share zero weights.
- **Weights:** ~118 GB (downloaded from Hugging Face)

## Performance

Measured on trn2.48xlarge (16 NeuronDevices, 64 NeuronCores with LNC=2) with Neuron SDK 2.28:

| Phase | Time |
|-------|------|
| Pipeline load | ~3s |
| Text encoding (CPU) | ~5s |
| Expert 1 load (TP=4, CP=16) | ~121s |
| Expert 1 denoise (16 steps) | ~82s |
| Expert 2 load (TP=4, CP=16) | ~121s |
| Expert 2 denoise (34 steps) | ~175s |
| VAE decode (Neuron tiled) | ~33s |
| **Total wall time** | **~596s** |

- **Per-step average:** ~5.2s
- **Resolution:** 768x1280, 81 frames, 50 denoising steps
- **Parallelism:** TP=4, CP=16 (world_size=64) per expert — all 64 NeuronCores

### GPU Comparison

| Config | Time (s) | vs trn2.48xlarge |
|--------|----------|------------------|
| **trn2.48xlarge (ours)** | **~596** ||
| 1x H100 | 1042 |  |
| 4x H100 | 289 | |
| 1x A100 | 2736 |  |
| 4x A100 | 725 |  |

GPU numbers from official Wan 2.2 benchmarks (warm start, 720P).

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
- **AMI:** Deep Learning AMI Neuron (Ubuntu 24.04) 20260227 (SDK 2.28)
- **Venv:** `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`
- **Disk:** 300+ GB EBS + NVMe mount for model weights and compiled artifacts
- **diffusers:** 0.37.1+ (required for `transformer_2` attribute / A14B MoE support)

## Architecture Notes

The A14B MoE model uses two completely independent transformer experts. Because they share no weights, each requires all 64 NeuronCores (TP=4, CP=16). The pipeline uses subprocess isolation to cleanly load/unload each expert:

1. Loads Expert 1, runs 16 high-noise denoising steps on all 64 cores
2. Subprocess exits (cleanly frees HBM), loads Expert 2
3. Runs 34 low-noise denoising steps with Expert 2 on all 64 cores
4. Decodes latents to video with the tiled VAE decoder

Context Parallelism (CP=16) splits the 80,640-token sequence across all cores, achieving near-linear scaling: 3.35x faster per-step compared to CP=4. This uses all available hardware rather than leaving 48 of 64 cores idle.

The compilation code is from [whn09/aws-neuron-samples](https://github.com/whn09/aws-neuron-samples) (torch-neuronx/inference/hf_pretrained_wan2.2_t2v_a14b/).
