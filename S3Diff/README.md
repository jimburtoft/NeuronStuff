# S3Diff One-Step Super-Resolution on AWS Trainium2

Run S3Diff one-step 4x super-resolution (128x128 to 512x512) on a trn2.3xlarge instance. Achieves 2.12 images/sec — a 24.5x speedup over CPU.

## Model

- **Name:** [Yukang/S3Diff](https://github.com/ArcticHare105/S3Diff)
- **Architecture:** One-step diffusion model with degradation-aware LoRA modulation: DEResNet encoder, UNet with dynamic LoRA, VAE encoder/decoder
- **Parameters:** ~2 GB total (small enough for a single NeuronCore)
- **Paper:** "Degradation-Guided One-Step Image Super-Resolution with Diffusion Priors" (ECCV 2024)

## Performance

Measured on trn2.3xlarge (2 NeuronDevices, 4 NeuronCores with LNC=2) with Neuron SDK 2.29:

| Phase | Time |
|-------|------|
| DEResNet | 3.8ms |
| Modulation (CPU) | 0.5ms |
| VAE Encode | 83.2ms |
| UNet x2 (CFG) | 218.8ms |
| VAE Decode | 164.6ms |
| **Total** | **0.471s** |

| Metric | Value |
|--------|-------|
| Resolution | 128x128 → 512x512 (4x SR) |
| Inference steps | 1 (one-step model) |
| Throughput | **2.12 img/s** |
| Std dev | 0.000s |
| Total compile time | ~21 min |
| CPU baseline | 11.53s |
| **Speedup vs CPU** | **24.5x** |

## Key Technical Details

S3Diff uses dynamic LoRA modulation: the DEResNet encoder analyzes input degradation and produces per-layer LoRA scaling factors (`de_mod_all`). These modulate the UNet's LoRA layers at inference time. On Neuron, dynamic LoRA is handled by wrapper `nn.Module` classes that accept the modulation tensors as explicit function arguments (since `torch_neuronx.trace()` requires static computation graphs).

The LoRA operations use einsum patterns (`...khw,...kr->...rhw` for Conv2d, `...lk,...kr->...lr` for Linear) which compile correctly on Neuron.

## Files

| File | Description |
|------|-------------|
| `s3diff_trn2.ipynb` | Source notebook (no outputs) |

## Quick Start

Open and run `s3diff_trn2.ipynb` on a trn2.3xlarge instance. The notebook covers:

1. Environment setup (clone S3Diff repo, install dependencies)
2. Model download and weight loading
3. Compilation of all 5 components (~21 min total):
   - DEResNet encoder
   - VAE encoder
   - UNet (with LoRA wrappers)
   - VAE decoder
   - Modulation network (stays on CPU)
4. End-to-end inference with timing breakdown
5. Benchmark (10 warm iterations)

## Requirements

- **Instance:** trn2.3xlarge (model is small, single core is sufficient)
- **AMI:** Deep Learning AMI Neuron (Ubuntu 24.04) 20260410 (SDK 2.29)
- **Venv:** `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`
- **Disk:** 300+ GB EBS
- **Note:** S3Diff pip package only installs `basicsr` — must clone the GitHub repo for actual model code

## Architecture Notes

S3Diff is unusual among diffusion models because it uses only a single denoising step, making it extremely fast. The model compensates for the single step by conditioning the UNet on input-specific degradation information through dynamic LoRA layers.

Compiler flags: `--auto-cast=matmult` + `-O1` for DEResNet, VAE encoder, and UNet. VAE decoder uses `--model-type=unet-inference` instead (matmult flag causes compiler errors on the VAE decoder).
