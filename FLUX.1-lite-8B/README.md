# FLUX.1-lite-8B Image Generation on AWS Trainium2

Generate 1024x1024 images from text prompts using FLUX.1-lite-8B on a trn2.3xlarge instance.

## Model

- **Name:** [Freepik/flux.1-lite-8B](https://huggingface.co/Freepik/flux.1-lite-8B)
- **Architecture:** 8B parameter diffusion transformer with 8 MMDiT double-stream blocks + 38 DiT single-stream blocks, dual text encoders (CLIP + T5-XXL)
- **License:** Check HuggingFace model card

## Performance

Measured on trn2.3xlarge (2 NeuronDevices, 4 NeuronCores with LNC=2) with Neuron SDK 2.29:

| Metric | Value |
|--------|-------|
| Mean denoise latency (28 steps) | **39.00s** |
| Steps/sec | **0.72** |
| Std dev | 0.00s |
| VAE decode | 0.29s |
| Total compile time | ~117 min |
| Transformer NEFF total | 25.0 GB (3 parts) |

## Key Technical Details

The full 8B transformer exceeds the Neuron compiler's 5M instruction limit (8.6M actual), so it must be split into 3 parts:

| Part | Layers | Instructions | Compile Time |
|------|--------|-------------|-------------|
| Part A | 8 MMDiT double blocks | ~4.5M | ~50 min |
| Part B | Single blocks 0-18 | ~3.5M | ~35 min |
| Part C | Single blocks 19-37 + output | ~3.5M | ~32 min |

**Important:** Do NOT use `--model-type=transformer` — it causes 37% mean error per step due to an incompatible custom softmax kernel, producing blank images. Compile with `-O1` only.

Staged model loading is required: text encoders run in a subprocess (core 0), then the transformer loads onto cores 1-3 after the subprocess exits (Neuron runtime doesn't release HBM on model delete within the same process).

## Files

| File | Description |
|------|-------------|
| `flux1_lite_8b_trn2.ipynb` | Source notebook (no outputs) |

## Quick Start

Open and run `flux1_lite_8b_trn2.ipynb` on a trn2.3xlarge instance. The notebook covers:

1. Environment setup (dependencies, model download)
2. 3-part transformer compilation with `-O1` flags (~117 min total)
3. Text encoder compilation (CLIP + T5-XXL in subprocess)
4. VAE decoder compilation
5. End-to-end inference with timing breakdown

## Requirements

- **Instance:** trn2.3xlarge (2 NeuronDevices, 4 NeuronCores with LNC=2)
- **AMI:** Deep Learning AMI Neuron (Ubuntu 24.04) 20260410 (SDK 2.29)
- **Venv:** `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`
- **Disk:** 300+ GB EBS for model weights and compiled artifacts
- **diffusers:** 0.37.1+

## Architecture Notes

Each denoising step executes 3 sequential NEFFs (~1.39s/step total). The bottleneck is the sequential chain — the model cannot be parallelized further without NxD Inference tensor parallelism, which would allow a single NEFF for the full transformer.

FLUX.1-lite uses guidance distillation (no classifier-free guidance), so only one forward pass is needed per step — unlike FLUX.2-klein which requires 2 passes per step for classic CFG.
