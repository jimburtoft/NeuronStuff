# Wan 2.2 T2V-A14B Video Generation on AWS Trainium2

Generate 768x1280 (81 frames, ~3.2s) video from text prompts using the Wan 2.2 T2V-A14B Mixture-of-Experts diffusion model on a trn2.48xlarge instance.

## Model

- **Name:** [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)
- **Architecture:** 27B parameter MoE with 14B active per denoising step
- **Two separate experts:** Expert 1 handles high-noise steps (1-16), Expert 2 handles low-noise steps (17-50). They share zero weights.
- **Weights:** ~118 GB (downloaded from Hugging Face)

## Performance

Measured on trn2.48xlarge (16 NeuronDevices, 32 NeuronCores, LNC=2) with Neuron SDK 2.28:

| Phase | Time |
|-------|------|
| Pipeline load | 2.9s |
| Text encoding (CPU) | 5.2s |
| Expert 1 load (TP=4, CP=4) | 213.0s |
| Expert 1 denoise (16 steps) | 276.8s |
| Expert 2 load (TP=4, CP=4) | 223.2s |
| Expert 2 denoise (34 steps) | 582.6s |
| VAE decode (Neuron tiled) | 33.4s |
| **Total wall time** | **1190.0s** |

- **Per-step average:** 17.2s
- **Resolution:** 768x1280, 81 frames, 50 denoising steps
- **Parallelism:** TP=4, CP=4 (world_size=16) per expert

## Files

| File | Description |
|------|-------------|
| `wan22_t2v_a14b_trn2.ipynb` | Source notebook (no outputs) |
| `wan22_t2v_a14b_trn2_executed.ipynb` | Executed notebook with all outputs |
| `worker_denoise.py` | Subprocess worker for expert denoising |
| `worker_denoise_preload.py` | Preloading variant that loads the model while the other expert runs |

## Quick Start

Open and run `wan22_t2v_a14b_trn2.ipynb` on a trn2.48xlarge instance. The notebook covers:

1. Environment setup (NVMe mount, dependencies)
2. Model download (118 GB from Hugging Face)
3. Compilation of all four components (~50 min from scratch):
   - Text encoder
   - Transformer Expert 1 (high noise)
   - Transformer Expert 2 (low noise)
   - Tiled VAE decoder
4. End-to-end inference with timing breakdown

## Requirements

- **Instance:** trn2.48xlarge (16 NeuronDevices required)
- **LNC:** 2 (default, gives 32 logical cores with 48 GB HBM each)
- **AMI:** Deep Learning AMI Neuron (Ubuntu 24.04) 20260227 (SDK 2.28)
- **Venv:** `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`
- **Disk:** 300+ GB EBS + NVMe mount for model weights and compiled artifacts
- **diffusers:** 0.37.1+ (required for `transformer_2` attribute / A14B MoE support)

## Architecture Notes

The A14B MoE model uses two completely independent transformer experts. Because they share no weights, they cannot be loaded simultaneously on a trn2.48xlarge (each requires 16 NeuronCores with TP=4/CP=4). The pipeline:

1. Loads Expert 1, runs 16 high-noise denoising steps
2. Unloads Expert 1, loads Expert 2 (preloaded in parallel during step 1)
3. Runs 34 low-noise denoising steps with Expert 2
4. Decodes latents to video with the tiled VAE decoder

The compilation code is from [whn09/aws-neuron-samples](https://github.com/whn09/aws-neuron-samples) (torch-neuronx/inference/hf_pretrained_wan2.2_t2v_a14b/).
