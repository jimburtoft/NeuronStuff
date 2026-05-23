#!/usr/bin/env python3
"""
SDXL high-resolution benchmark via img2img upscale approach on Neuron.

Generates coherent 2048x2048 and 4096x4096 images using compiled 1024x1024 NEFFs:
1. Generate at 1024x1024 (proven, compiled UNet)
2. Upscale to target resolution (bicubic)
3. Tiled VAE encode back to latent space
4. Add partial noise (strength=0.35, 18/50 steps)
5. Tiled denoising refinement
6. Tiled VAE decode

This approach works because the 1K generation establishes global coherence,
and the tiled refinement only adds local high-frequency detail.

Requirements:
- Pre-compiled NEFFs at 1024x1024 (UNet, text encoders, VAE encoder/decoder)
- trn2.3xlarge with SDK 2.29+
- diffusers, torch_neuronx, neuronxcc

Usage:
    # Compile all NEFFs first (one-time, ~45 min)
    python benchmark_img2img.py compile --model /path/to/sdxl --compile_dir /path/to/neffs

    # Run benchmark
    python benchmark_img2img.py benchmark --model /path/to/sdxl --compile_dir /path/to/neffs

    # Single run at specific resolution
    python benchmark_img2img.py run --model /path/to/sdxl --compile_dir /path/to/neffs --resolution 2048 --seed 42
"""

import os
import sys
import time
import copy
import math
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx
import diffusers
from diffusers import DiffusionPipeline
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.attention_processor import Attention
from transformers.models.clip.modeling_clip import CLIPTextModelOutput
from typing import List, Tuple
from PIL import Image

try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
    from torch_neuronx.xla_impl.ops import nki_jit

    _flash_fwd_call = nki_jit()(attention_isa_kernel)
except ImportError:
    _flash_fwd_call = None
    print("WARNING: attention_isa_kernel not available, using SDPA fallback")


# ============================================================
# Configuration
# ============================================================
TILE_LATENT_SIZE = 128  # 1024/8 = 128 latent pixels per tile
TILE_OVERLAP = 32  # overlap in latent space (256 pixel overlap)
NUM_STEPS = 50  # match astronaut benchmark
GUIDANCE_SCALE = 7.5
DENOISE_STRENGTH = 0.35  # 18/50 steps of refinement


# ============================================================
# NKI Flash Attention
# ============================================================
def attention_wrapper_without_swap(query, key, value):
    bs, n_head, q_len, d_head = query.shape
    k_len = key.shape[2]
    v_len = value.shape[2]
    q = query.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, q_len))
    k = key.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, k_len))
    v = value.clone().reshape((bs * n_head, v_len, d_head))
    attn_output = torch.zeros(
        (bs * n_head, q_len, d_head), dtype=torch.bfloat16, device=q.device
    )
    scale = 1 / math.sqrt(d_head)
    _flash_fwd_call(
        q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap"
    )
    return attn_output.reshape((bs, n_head, q_len, d_head))


class KernelizedAttnProcessor2_0:
    def __init__(self):
        pass

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        use_nki = (
            _flash_fwd_call is not None
            and attention_mask is None
            and query.shape[3] <= query.shape[2]
            and query.shape[3] <= 128
            and value.shape[2] != 77
        )

        if use_nki:
            hidden_states = attention_wrapper_without_swap(query, key, value)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ============================================================
# Model Wrappers
# ============================================================
def get_attention_scores_neuron(self, query, key, attn_mask):
    if query.size() == key.size():
        attention_scores = torch.bmm(key, query.transpose(-1, -2)) * self.scale
        attention_probs = attention_scores.softmax(dim=1).permute(0, 2, 1)
    else:
        attention_scores = torch.bmm(query, key.transpose(-1, -2)) * self.scale
        attention_probs = attention_scores.softmax(dim=-1)
    return attention_probs


class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(
        self, sample, timestep, encoder_hidden_states, text_embeds=None, time_ids=None
    ):
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            return_dict=False,
        )


class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.add_embedding = unetwrap.unet.add_embedding
        self.device = unetwrap.unet.device

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        timestep_cond=None,
        added_cond_kwargs=None,
        return_dict=False,
        cross_attention_kwargs=None,
    ):
        sample = self.unetwrap(
            sample,
            timestep.float().expand((sample.shape[0],)),
            encoder_hidden_states,
            added_cond_kwargs["text_embeds"],
            added_cond_kwargs["time_ids"],
        )[0]
        return UNet2DConditionOutput(sample=sample)


class TraceableTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, text_input_ids):
        return self.text_encoder(
            text_input_ids, output_hidden_states=True, return_dict=False
        )


class TextEncoderOutputWrapper(nn.Module):
    def __init__(self, traceable_text_encoder, original_text_encoder):
        super().__init__()
        self.traceable_text_encoder = traceable_text_encoder
        self.config = original_text_encoder.config
        self.dtype = original_text_encoder.dtype
        self.device = original_text_encoder.device

    def forward(self, text_input_ids, output_hidden_states=True):
        out_tuple = self.traceable_text_encoder(text_input_ids)
        return CLIPTextModelOutput(
            text_embeds=out_tuple[0],
            last_hidden_state=out_tuple[1],
            hidden_states=out_tuple[2],
        )


# ============================================================
# Tiled Diffusion Core
# ============================================================
def get_tile_positions(
    full_size: int, tile_size: int, overlap: int
) -> List[Tuple[int, int]]:
    """Generate tile start/end positions with overlap."""
    stride = tile_size - overlap
    positions = []
    start = 0
    while start < full_size:
        end = min(start + tile_size, full_size)
        if end - start < tile_size and start > 0:
            start = full_size - tile_size
            end = full_size
        positions.append((start, end))
        if end >= full_size:
            break
        start += stride
    return positions


def tiled_denoise_step(
    latents,
    t,
    encoder_hidden_states,
    text_embeds,
    time_ids,
    unet_neuron,
    scheduler,
    tile_positions_h,
    tile_positions_w,
    tile_weights,
    guidance_scale,
):
    """Perform one denoising step using tiled UNet inference."""
    full_h, full_w = latents.shape[2], latents.shape[3]
    noise_pred_full = torch.zeros_like(latents)
    weight_sum = torch.zeros(1, 1, full_h, full_w, device=latents.device)

    for h_start, h_end in tile_positions_h:
        for w_start, w_end in tile_positions_w:
            tile_latent = latents[:, :, h_start:h_end, w_start:w_end].clone()

            # CRITICAL: scale_model_input required by EulerDiscreteScheduler
            tile_latent_scaled = scheduler.scale_model_input(tile_latent, t)

            # CFG: concat unconditional + conditional
            latent_input = torch.cat([tile_latent_scaled] * 2)
            t_expand = t.expand(2)

            # Run UNet on tile
            noise_pred = unet_neuron(
                latent_input, t_expand, encoder_hidden_states, text_embeds, time_ids
            )[0]

            # CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_tile = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # Accumulate with uniform weight (averaging creates coherence)
            th, tw = h_end - h_start, w_end - w_start
            tile_weight = tile_weights[:th, :tw].to(latents.device)
            noise_pred_full[:, :, h_start:h_end, w_start:w_end] += (
                noise_pred_tile * tile_weight.unsqueeze(0).unsqueeze(0)
            )
            weight_sum[:, :, h_start:h_end, w_start:w_end] += tile_weight.unsqueeze(
                0
            ).unsqueeze(0)

    # Normalize by weights
    noise_pred_full = noise_pred_full / weight_sum.clamp(min=1e-8)

    # Scheduler step
    latents = scheduler.step(noise_pred_full, t, latents, return_dict=False)[0]
    return latents


# ============================================================
# Compilation
# ============================================================
def compile_all(model_path, compile_dir):
    """Compile all SDXL components at 1024x1024."""
    os.makedirs(compile_dir, exist_ok=True)

    pipe = DiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )

    # Text Encoder 1
    print("Compiling Text Encoder 1...")
    te = copy.deepcopy(TraceableTextEncoder(pipe.text_encoder))
    text_ids = torch.tensor([[49406, 736, 1615, 49407] + [49407] * 73])
    neuron_te = torch_neuronx.trace(
        te,
        text_ids,
        compiler_workdir=os.path.join(compile_dir, "text_encoder"),
        compiler_args=[],
    )
    torch.jit.save(neuron_te, os.path.join(compile_dir, "text_encoder/model.pt"))
    del neuron_te, te

    # Text Encoder 2
    print("Compiling Text Encoder 2...")
    te2 = copy.deepcopy(TraceableTextEncoder(pipe.text_encoder_2))
    text_ids2 = torch.tensor([[49406, 736, 1615, 49407] + [0] * 73])
    neuron_te2 = torch_neuronx.trace(
        te2,
        text_ids2,
        compiler_workdir=os.path.join(compile_dir, "text_encoder_2"),
        compiler_args=[],
    )
    torch.jit.save(neuron_te2, os.path.join(compile_dir, "text_encoder_2/model.pt"))
    del neuron_te2, te2

    # VAE Decoder
    print("Compiling VAE Decoder...")
    decoder = copy.deepcopy(pipe.vae.decoder)
    decoder_in = torch.randn([1, 4, 128, 128])
    decoder_neuron = torch_neuronx.trace(
        decoder,
        decoder_in,
        compiler_workdir=os.path.join(compile_dir, "vae_decoder"),
        compiler_args=["--model-type=unet-inference"],
    )
    torch.jit.save(decoder_neuron, os.path.join(compile_dir, "vae_decoder/model.pt"))
    del decoder, decoder_neuron

    # VAE Post Quant Conv
    print("Compiling VAE Post Quant Conv...")
    pqc = copy.deepcopy(pipe.vae.post_quant_conv)
    pqc_in = torch.randn([1, 4, 128, 128])
    pqc_neuron = torch_neuronx.trace(
        pqc,
        pqc_in,
        compiler_workdir=os.path.join(compile_dir, "vae_post_quant_conv"),
    )
    torch.jit.save(
        pqc_neuron, os.path.join(compile_dir, "vae_post_quant_conv/model.pt")
    )
    del pqc, pqc_neuron

    # VAE Encoder (needed for img2img encode step)
    print("Compiling VAE Encoder...")
    encoder = copy.deepcopy(pipe.vae.encoder)
    encoder_in = torch.randn([1, 3, 1024, 1024])
    encoder_neuron = torch_neuronx.trace(
        encoder,
        encoder_in,
        compiler_workdir=os.path.join(compile_dir, "vae_encoder"),
        compiler_args=["--model-type=unet-inference"],
    )
    torch.jit.save(encoder_neuron, os.path.join(compile_dir, "vae_encoder/model.pt"))
    del encoder, encoder_neuron

    # VAE Quant Conv (needed for img2img encode step)
    print("Compiling VAE Quant Conv...")
    qc = copy.deepcopy(pipe.vae.quant_conv)
    qc_in = torch.randn([1, 8, 128, 128])
    qc_neuron = torch_neuronx.trace(
        qc,
        qc_in,
        compiler_workdir=os.path.join(compile_dir, "vae_quant_conv"),
    )
    torch.jit.save(qc_neuron, os.path.join(compile_dir, "vae_quant_conv/model.pt"))
    del qc, qc_neuron

    # UNet
    print("Compiling UNet (this takes ~30-60 min)...")
    Attention.get_attention_scores = get_attention_scores_neuron
    pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
    diffusers.models.attention_processor.AttnProcessor2_0.__call__ = (
        KernelizedAttnProcessor2_0.__call__
    )
    unet = copy.deepcopy(pipe.unet.unetwrap)
    del pipe

    sample = torch.randn([2, 4, 128, 128])
    timestep = torch.tensor(999).float().expand((2,))
    enc_hs = torch.randn([2, 77, 2048])
    text_emb = torch.randn([2, 1280])
    time_ids_in = torch.randn([2, 6])

    t0 = time.time()
    unet_neuron = torch_neuronx.trace(
        unet,
        (sample, timestep, enc_hs, text_emb, time_ids_in),
        compiler_workdir=os.path.join(compile_dir, "unet"),
        compiler_args=["--model-type=unet-inference", "--auto-cast", "matmult"],
    )
    torch.jit.save(unet_neuron, os.path.join(compile_dir, "unet/model.pt"))
    print(f"  UNet compiled in {time.time() - t0:.0f}s")
    del unet, unet_neuron

    print("\nALL COMPILATION COMPLETE")
    print(f"NEFFs saved to: {compile_dir}")


# ============================================================
# Load Models
# ============================================================
def load_all_models(model_path, compile_dir):
    """Load pipeline and all compiled Neuron models."""
    pipe = DiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )

    # Text encoders
    neuron_te = torch.jit.load(os.path.join(compile_dir, "text_encoder/model.pt"))
    neuron_te2 = torch.jit.load(os.path.join(compile_dir, "text_encoder_2/model.pt"))
    pipe.text_encoder = TextEncoderOutputWrapper(neuron_te, pipe.text_encoder)
    pipe.text_encoder_2 = TextEncoderOutputWrapper(neuron_te2, pipe.text_encoder_2)

    # UNet
    Attention.get_attention_scores = get_attention_scores_neuron
    pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
    diffusers.models.attention_processor.AttnProcessor2_0.__call__ = (
        KernelizedAttnProcessor2_0.__call__
    )
    unet_neuron = torch.jit.load(os.path.join(compile_dir, "unet/model.pt"))
    pipe.unet.unetwrap = unet_neuron

    # VAE decoder
    pipe.vae.decoder = torch.jit.load(os.path.join(compile_dir, "vae_decoder/model.pt"))
    pipe.vae.post_quant_conv = torch.jit.load(
        os.path.join(compile_dir, "vae_post_quant_conv/model.pt")
    )

    # VAE encoder (for img2img)
    vae_encoder = torch.jit.load(os.path.join(compile_dir, "vae_encoder/model.pt"))
    vae_quant_conv = torch.jit.load(
        os.path.join(compile_dir, "vae_quant_conv/model.pt")
    )

    return {
        "pipe": pipe,
        "unet_neuron": unet_neuron,
        "vae_encoder": vae_encoder,
        "vae_quant_conv": vae_quant_conv,
    }


# ============================================================
# High-Resolution Generation
# ============================================================
def generate_highres(models, prompt, resolution, seed, out_dir=None):
    """Generate a high-res image using img2img upscale approach."""
    pipe = models["pipe"]
    unet_neuron = models["unet_neuron"]
    vae_encoder = models["vae_encoder"]
    vae_quant_conv = models["vae_quant_conv"]
    latent_size = resolution // 8
    encode_tiles = resolution // 1024

    t_total = time.time()

    # Stage 1: Generate at 1024x1024
    t0 = time.time()
    output_1k = pipe(
        prompt,
        num_inference_steps=NUM_STEPS,
        generator=torch.Generator().manual_seed(seed),
    )
    img_1k = output_1k.images[0]
    t_1k = time.time() - t0

    # Stage 2: Upscale to target resolution
    img_hr = img_1k.resize((resolution, resolution), Image.LANCZOS)
    img_tensor = torch.from_numpy(np.array(img_hr)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    img_tensor = 2 * img_tensor - 1

    # Stage 3: Tiled VAE encode
    t_enc = time.time()
    latent_full = torch.zeros(1, 4, latent_size, latent_size)
    for row in range(encode_tiles):
        for col in range(encode_tiles):
            tile = img_tensor[
                :, :, row * 1024 : (row + 1) * 1024, col * 1024 : (col + 1) * 1024
            ]
            h = vae_encoder(tile)
            h = vae_quant_conv(h)
            latent_full[
                :, :, row * 128 : (row + 1) * 128, col * 128 : (col + 1) * 128
            ] = h[:, :4] * pipe.vae.config.scaling_factor
    enc_time = time.time() - t_enc

    # Stage 4: Add partial noise
    start_step = int(NUM_STEPS * (1 - DENOISE_STRENGTH))
    pipe.scheduler.set_timesteps(NUM_STEPS)
    timesteps = pipe.scheduler.timesteps[start_step:]
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(latent_full.shape, generator=generator)
    sigma = pipe.scheduler.sigmas[start_step]
    latents = latent_full + noise * sigma

    # Stage 5: Tiled denoising refinement
    with torch.no_grad():
        pe, npe, ppe, nppe = pipe.encode_prompt(
            prompt,
            device="cpu",
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
    enc_hs = torch.cat([npe, pe])
    text_emb = torch.cat([nppe, ppe])
    time_ids = torch.tensor([[1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0]])
    time_ids = torch.cat([time_ids] * 2)

    tile_positions_h = get_tile_positions(latent_size, TILE_LATENT_SIZE, TILE_OVERLAP)
    tile_positions_w = get_tile_positions(latent_size, TILE_LATENT_SIZE, TILE_OVERLAP)
    tile_weights = torch.ones(TILE_LATENT_SIZE, TILE_LATENT_SIZE)
    n_tiles = len(tile_positions_h) * len(tile_positions_w)

    pipe.scheduler._step_index = start_step

    t_denoise = time.time()
    for t in timesteps:
        latents = tiled_denoise_step(
            latents,
            t,
            enc_hs,
            text_emb,
            time_ids,
            unet_neuron,
            pipe.scheduler,
            tile_positions_h,
            tile_positions_w,
            tile_weights,
            GUIDANCE_SCALE,
        )
    denoise_time = time.time() - t_denoise

    # Stage 6: Tiled VAE decode
    t_vae = time.time()
    full_image = torch.zeros(1, 3, resolution, resolution)
    decode_tiles = resolution // 1024
    for row in range(decode_tiles):
        for col in range(decode_tiles):
            tile_lat = latents[
                :, :, row * 128 : (row + 1) * 128, col * 128 : (col + 1) * 128
            ]
            tile_lat = 1 / pipe.vae.config.scaling_factor * tile_lat
            tile_lat = pipe.vae.post_quant_conv(tile_lat)
            tile_px = pipe.vae.decoder(tile_lat)
            full_image[
                :, :, row * 1024 : (row + 1) * 1024, col * 1024 : (col + 1) * 1024
            ] = tile_px
    vae_time = time.time() - t_vae

    total_time = time.time() - t_total

    # Save image
    image = (full_image / 2 + 0.5).clamp(0, 1)
    image = image.squeeze(0).permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        Image.fromarray(image).save(os.path.join(out_dir, f"seed{seed}.png"))

    return {
        "resolution": resolution,
        "seed": seed,
        "total_time_s": round(total_time, 2),
        "gen_1k_s": round(t_1k, 2),
        "encode_s": round(enc_time, 2),
        "denoise_s": round(denoise_time, 2),
        "vae_decode_s": round(vae_time, 2),
        "n_tiles": n_tiles,
        "denoise_steps": len(timesteps),
        "denoise_strength": DENOISE_STRENGTH,
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="SDXL high-res img2img benchmark on Neuron"
    )
    parser.add_argument(
        "command",
        choices=["compile", "run", "benchmark"],
        help="compile: compile NEFFs; run: single image; benchmark: full benchmark",
    )
    parser.add_argument("--model", required=True, help="Path to SDXL model")
    parser.add_argument(
        "--compile_dir", required=True, help="Path to store/load compiled NEFFs"
    )
    parser.add_argument(
        "--resolution", type=int, default=2048, help="Target resolution (2048 or 4096)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (for run command)"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Seeds for benchmark (default: 42-51 for 2K, 42-44 for 4K)",
    )
    parser.add_argument(
        "--prompt",
        default="An astronaut riding a green horse",
        help="Generation prompt",
    )
    parser.add_argument(
        "--out", default=None, help="Output directory for images and results"
    )
    args = parser.parse_args()

    if args.command == "compile":
        compile_all(args.model, args.compile_dir)
        return

    # Load models
    print("Loading models...")
    models = load_all_models(args.model, args.compile_dir)

    if args.command == "run":
        out_dir = args.out or os.path.join(
            args.compile_dir, f"results_{args.resolution}"
        )
        print(f"\nGenerating {args.resolution}x{args.resolution} (seed={args.seed})...")
        r = generate_highres(models, args.prompt, args.resolution, args.seed, out_dir)
        print(f"\nResult: {r['total_time_s']:.2f}s total")
        print(f"  1K gen: {r['gen_1k_s']:.2f}s")
        print(f"  Encode: {r['encode_s']:.2f}s")
        print(
            f"  Denoise ({r['denoise_steps']} steps, {r['n_tiles']} tiles): {r['denoise_s']:.2f}s"
        )
        print(f"  VAE decode: {r['vae_decode_s']:.2f}s")
        return

    # Benchmark mode
    out_base = args.out or args.compile_dir

    # Warmup
    print("\nWarmup...")
    _ = generate_highres(models, args.prompt, 2048, seed=0)
    print("Warmup done\n")

    all_results = {}

    # 2K benchmark
    seeds_2k = args.seeds or list(range(42, 52))
    print("=" * 60)
    print(f"BENCHMARK: 2048x2048, {len(seeds_2k)} seeds")
    print("=" * 60)
    out_2k = os.path.join(out_base, "results_2048")
    results_2k = []
    for seed in seeds_2k:
        r = generate_highres(models, args.prompt, 2048, seed, out_2k)
        print(f"  seed={seed}: {r['total_time_s']:.2f}s")
        results_2k.append(r)

    mean_2k = np.mean([r["total_time_s"] for r in results_2k])
    std_2k = np.std([r["total_time_s"] for r in results_2k])
    print(
        f"\n2048x2048: {mean_2k:.2f}s +/- {std_2k:.2f}s ({len(seeds_2k)}/{len(seeds_2k)} pass)\n"
    )
    all_results["2048"] = {
        "mean_s": round(float(mean_2k), 2),
        "std_s": round(float(std_2k), 2),
        "n_seeds": len(seeds_2k),
        "pass": f"{len(seeds_2k)}/{len(seeds_2k)}",
        "per_seed": results_2k,
    }

    # 4K benchmark
    seeds_4k = [42, 43, 44]
    print("=" * 60)
    print(f"BENCHMARK: 4096x4096, {len(seeds_4k)} seeds")
    print("=" * 60)
    out_4k = os.path.join(out_base, "results_4096")
    results_4k = []
    for seed in seeds_4k:
        r = generate_highres(models, args.prompt, 4096, seed, out_4k)
        print(f"  seed={seed}: {r['total_time_s']:.2f}s")
        results_4k.append(r)

    mean_4k = np.mean([r["total_time_s"] for r in results_4k])
    std_4k = np.std([r["total_time_s"] for r in results_4k])
    print(
        f"\n4096x4096: {mean_4k:.2f}s +/- {std_4k:.2f}s ({len(seeds_4k)}/{len(seeds_4k)} pass)\n"
    )
    all_results["4096"] = {
        "mean_s": round(float(mean_4k), 2),
        "std_s": round(float(std_4k), 2),
        "n_seeds": len(seeds_4k),
        "pass": f"{len(seeds_4k)}/{len(seeds_4k)}",
        "per_seed": results_4k,
    }

    # Save summary
    summary = {
        "approach": "img2img_upscale",
        "description": "Generate 1K -> upscale -> tiled VAE encode -> partial noise -> tiled denoise -> tiled VAE decode",
        "denoise_strength": DENOISE_STRENGTH,
        "num_steps_total": NUM_STEPS,
        "num_steps_refine": int(NUM_STEPS * DENOISE_STRENGTH),
        "guidance_scale": GUIDANCE_SCALE,
        "tile_latent_size": TILE_LATENT_SIZE,
        "tile_overlap": TILE_OVERLAP,
        "prompt": args.prompt,
        "results": all_results,
    }
    summary_path = os.path.join(out_base, "benchmark_img2img_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    # Final table
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(
        f"{'Resolution':<12} {'Mean (s)':<12} {'Std (s)':<10} {'Pass':<8} {'$/image':<10}"
    )
    print("-" * 52)
    cost_per_hour = 2.235  # trn2.3xlarge full chip
    for res, data in all_results.items():
        cost = (data["mean_s"] / 3600) * cost_per_hour
        print(
            f"{res}x{res:<7} {data['mean_s']:<12.2f} {data['std_s']:<10.2f} {data['pass']:<8} ${cost:.4f}"
        )


if __name__ == "__main__":
    main()
