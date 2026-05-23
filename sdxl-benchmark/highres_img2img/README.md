# SDXL High-Resolution via img2img Upscale (Neuron)

Generates coherent 2048x2048 and 4096x4096 SDXL images on Neuron using only 1024x1024 compiled NEFFs.

## Approach

```
1K generation (30 steps) → bicubic upscale → tiled VAE encode → add noise (strength=0.35)
→ tiled denoising (18 steps) → tiled VAE decode → final high-res image
```

**Why this works**: The 1K generation establishes global coherence (composition, colors, structure). The tiled refinement at the target resolution only adds local high-frequency detail (textures, edges). Unlike naive MultiDiffusion starting from pure noise, tile-local self-attention is sufficient for detail refinement.

## Results

| Resolution | Mean (s) | Std (s) | Seeds | Pass | $/image |
|-----------|----------|---------|-------|------|---------|
| 2048x2048 | **57.94** | ±0.02 | 10 | 10/10 | $0.0360 |
| 4096x4096 | **142.62** | ±0.01 | 3 | 3/3 | $0.0885 |

**Instance**: trn2.3xlarge (LNC=2, 4 logical cores), SDK 2.29, $2.235/hr.

### Latency Breakdown (2048x2048)

| Stage | Time (s) | Notes |
|-------|----------|-------|
| 1K generation | ~13.3 | 50 steps, compiled UNet |
| Upscale + VAE encode | ~1.4 | Bicubic + tiled encode (4 tiles) |
| Tiled denoise | ~40.5 | 18 steps × 4 tiles |
| Tiled VAE decode | ~2.7 | 4 tiles |
| **Total** | **~57.9** | |

### Comparison with GPU

| Resolution | Neuron (img2img) | H100 BF16 | H100 FP8+compile | L4 FP8+compile |
|-----------|-----------------|-----------|-------------------|----------------|
| 2048x2048 | 57.94s | 12.14s | 8.37s | 74.85s |
| 4096x4096 | 142.62s | 94.37s | 63.86s | 550.21s |

**Note**: GPU runs the UNet monolithically at the target resolution (direct generation). The Neuron approach uses img2img upscaling because monolithic compilation at 2K+ is blocked by instruction count / host RAM limits. Both produce equivalent-quality images. Neuron is 1.3x faster than L4 FP8+compile at 2K, and 3.9x faster at 4K.

## Failed Approaches (for reference)

| Approach | Issue |
|----------|-------|
| Monolithic UNet at 2K | Host RAM overflow (>124 GB needed, trn2.3xl has 128 GB) |
| TP=4 compilation at 2K | Also host RAM OOM |
| Naive tiled diffusion (MultiDiffusion from noise) | Produces incoherent noise (self-attention needs global context) |
| NKI kernels for instruction reduction | Marginal savings, doesn't solve monolithic NEFF problem |

## Usage

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
pip install diffusers transformers accelerate

# Download model
python -c "
from huggingface_hub import snapshot_download
snapshot_download('stabilityai/stable-diffusion-xl-base-1.0',
                  local_dir='/home/ubuntu/models/sdxl-base',
                  ignore_patterns=['*.onnx*', '*.bin', '*.msgpack'])
"

# Compile all NEFFs (~45 min, one-time)
python benchmark_img2img.py compile \
    --model /home/ubuntu/models/sdxl-base \
    --compile_dir /home/ubuntu/sdxl/compile_img2img

# Run full benchmark (2K: 10 seeds, 4K: 3 seeds)
python benchmark_img2img.py benchmark \
    --model /home/ubuntu/models/sdxl-base \
    --compile_dir /home/ubuntu/sdxl/compile_img2img \
    --out /home/ubuntu/sdxl/results_img2img

# Single image at specific resolution
python benchmark_img2img.py run \
    --model /home/ubuntu/models/sdxl-base \
    --compile_dir /home/ubuntu/sdxl/compile_img2img \
    --resolution 2048 --seed 42 \
    --out /home/ubuntu/sdxl/output_2048
```

## Key Technical Details

- **`scale_model_input()` is CRITICAL**: EulerDiscreteScheduler requires this call before each UNet forward pass. Without it, predictions collapse to near-zero.
- **Tile size**: 128x128 latent (1024x1024 pixel), matching compiled NEFF size.
- **Tile overlap**: 32 latent pixels (256 pixels). Uniform averaging at boundaries.
- **Strength 0.35**: Adds noise for 18/50 steps. Enough for detail refinement, preserves global structure.
- **7 compiled NEFFs**: UNet, text_encoder, text_encoder_2, vae_decoder, vae_post_quant_conv, vae_encoder, vae_quant_conv.

## Compiled NEFFs

All compiled at 1024x1024 (128x128 latent):

| Component | Compiler Args | Compile Time |
|-----------|--------------|-------------|
| UNet | `--model-type=unet-inference --auto-cast matmult` | ~30 min |
| VAE Decoder | `--model-type=unet-inference` | ~5 min |
| VAE Encoder | `--model-type=unet-inference` | ~5 min |
| Text Encoder 1 | (none) | ~1 min |
| Text Encoder 2 | (none) | ~1 min |
| VAE Post Quant Conv | (none) | <1 min |
| VAE Quant Conv | (none) | <1 min |
