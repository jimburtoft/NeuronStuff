"""
Wan2.2 TI2V 720P Inference with Model Swapping.

Uses sequential model loading/unloading to fit 720P (1280x704) on trn2.3xlarge
with LNC=2 (4 logical cores, 24 GB HBM each).

Flow:
1. Load pipeline (CPU) for tokenizer + scheduler
2. Load text_encoder -> encode prompt -> unload text_encoder
3. Load transformer -> run denoising loop -> unload transformer
4. Load PQC + tiled decoder -> decode latents -> save video

This avoids the OOM that occurs when all models are loaded simultaneously.
"""
import os
os.environ["NEURON_RT_NUM_CORES"] = "4"
os.environ["LOCAL_WORLD_SIZE"] = "4"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

import argparse
import gc
import json
import numpy as np
from PIL import Image
import random
import time
import torch
import torch_neuronx

try:
    import torch_xla.core.xla_model as xm
    xm.mark_step = lambda *args, **kwargs: None
except ImportError:
    pass

from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video, load_image
from neuronx_distributed import NxDModel
from safetensors.torch import load_file

from neuron_wan2_2_ti2v.neuron_commons import (
    InferenceTextEncoderWrapperV2,
    DecoderWrapperV3Tiled,
    PostQuantConvWrapperV2,
)

SEED = 42
HUGGINGFACE_CACHE_DIR = "/opt/dlami/nvme/wan2.2_ti2v_hf_cache_dir"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model_config(model_path):
    with open(os.path.join(model_path, "config.json")) as f:
        return json.load(f)


def load_sharded_weights(model_path, tp_degree):
    weights_path = os.path.join(model_path, "weights")
    sharded = []
    for rank in range(tp_degree):
        raw = load_file(os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors"))
        ckpt = {k: v for k, v in raw.items() if "master_weight" not in k}
        sharded.append(ckpt)
    return sharded


def load_duplicated_weights(model_path, world_size):
    base = load_file(os.path.join(model_path, "weights", "tp0_sharded_checkpoint.safetensors"))
    return [{k: v.clone() for k, v in base.items()} for _ in range(world_size)]


def prepare_cp_checkpoints(tp_checkpoints, tp_degree, cp_degree):
    checkpoints = []
    for cp_rank in range(cp_degree):
        for tp_rank in range(tp_degree):
            world_rank = cp_rank * tp_degree + tp_rank
            ck = {k: v.clone() for k, v in tp_checkpoints[tp_rank].items()}
            rk = "transformer.global_rank.rank"
            if rk in ck:
                ck[rk] = torch.tensor([world_rank], dtype=torch.int32)
            checkpoints.append(ck)
    return checkpoints


def free_nxd_model(nxd_model, label="model"):
    """Delete NxDModel and force garbage collection to free HBM."""
    del nxd_model
    gc.collect()
    gc.collect()
    time.sleep(1)
    rss_gb = int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) / 1e6
    print(f"  [{label}] freed. RSS: {rss_gb:.1f} GB")


def main(args):
    set_seed(SEED)

    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    compiled_dir = args.compiled_models_dir
    height, width = args.height, args.width
    num_frames = args.num_frames
    num_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale
    seqlen = args.max_sequence_length

    # ================================================================
    # Phase 0: Load base pipeline (CPU only, for tokenizer + scheduler)
    # ================================================================
    print("=" * 60)
    print("Phase 0: Loading base pipeline (CPU)...")
    print("=" * 60)
    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32,
        cache_dir=HUGGINGFACE_CACHE_DIR
    )
    pipe = WanPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=torch.bfloat16,
        cache_dir=HUGGINGFACE_CACHE_DIR
    )
    print(f"  Pipeline loaded in {time.time()-t0:.1f}s")

    # Keep references we need, free the rest
    tokenizer = pipe.tokenizer
    scheduler = pipe.scheduler
    text_encoder_cpu = pipe.text_encoder
    transformer_config = pipe.transformer.config
    vae_config = pipe.vae.config
    vae_decoder_cpu = pipe.vae.decoder
    vae_pqc_cpu = pipe.vae.post_quant_conv

    # ================================================================
    # Phase 0b: Encode input image for I2V (CPU, before text encoding)
    # ================================================================
    image_condition = None
    if args.image:
        print()
        print("Encoding input image for I2V (CPU VAE encoder)...")
        t_i2v = time.time()
        from diffusers.utils import load_image as _load_image
        input_image = _load_image(args.image)
        input_image = input_image.resize((width, height), Image.LANCZOS)
        img_tensor = torch.from_numpy(np.array(input_image)).float() / 127.5 - 1.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # [1, C, 1, H, W]
        img_tensor = img_tensor.to(torch.float32)

        with torch.no_grad():
            image_latents = pipe.vae.encode(img_tensor).latent_dist.mode()
            latents_mean_i2v = torch.tensor(vae_config.latents_mean).view(1, -1, 1, 1, 1)
            latents_std_i2v = torch.tensor(vae_config.latents_std).view(1, -1, 1, 1, 1)
            image_latents = (image_latents - latents_mean_i2v) / latents_std_i2v

        image_condition = image_latents.to(torch.float32)
        print(f"  Image encoded in {time.time()-t_i2v:.1f}s, latents: {image_condition.shape}")
        del img_tensor

    # ================================================================
    # Phase 1: Encode text prompt
    # ================================================================
    print()
    print("=" * 60)
    print("Phase 1: Encoding text prompt...")
    print("=" * 60)
    t0 = time.time()

    # Load text_encoder NxD model
    te_dir = f"{compiled_dir}/text_encoder"
    te_config = load_model_config(te_dir)
    te_tp = te_config["tp_degree"]
    te_ws = te_config.get("world_size", te_tp)

    te_wrapper = InferenceTextEncoderWrapperV2(torch.bfloat16, text_encoder_cpu, seqlen)
    te_nxd = NxDModel.load(os.path.join(te_dir, "nxd_model.pt"))
    te_weights = load_sharded_weights(te_dir, te_tp)
    if te_ws > te_tp:
        te_weights = prepare_cp_checkpoints(te_weights, te_tp, te_ws // te_tp)
    te_nxd.set_weights(te_weights)
    te_nxd.to_neuron()
    te_wrapper.t = te_nxd

    # Replace pipeline text_encoder temporarily for encode_prompt
    pipe.text_encoder = te_wrapper

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        max_sequence_length=seqlen,
        device=torch.device("cpu"),
    )
    prompt_embeds = prompt_embeds.to(torch.bfloat16)
    negative_prompt_embeds = negative_prompt_embeds.to(torch.bfloat16)
    print(f"  prompt_embeds: {prompt_embeds.shape}")
    print(f"  negative_prompt_embeds: {negative_prompt_embeds.shape}")
    print(f"  Text encoding done in {time.time()-t0:.1f}s")

    # Free text_encoder
    del te_weights
    pipe.text_encoder = text_encoder_cpu  # Restore CPU version
    free_nxd_model(te_nxd, "text_encoder")
    del te_wrapper

    # ================================================================
    # Phase 2: Denoising loop with transformer
    # ================================================================
    print()
    print("=" * 60)
    print("Phase 2: Loading transformer and running denoising...")
    print("=" * 60)
    t0 = time.time()

    # Check for CFG parallel first
    transformer_cfg_path = f"{compiled_dir}/transformer_cfg"
    transformer_cp_path = f"{compiled_dir}/transformer"
    transformer_path = transformer_cfg_path if os.path.exists(transformer_cfg_path) else transformer_cp_path

    tf_config = load_model_config(transformer_path)
    tp_degree = tf_config["tp_degree"]
    cp_degree = tf_config["cp_degree"]
    world_size = tf_config["world_size"]
    cfg_parallel = tf_config.get("cfg_parallel", False)

    print(f"  Loading transformer (TP={tp_degree}, CP={cp_degree}, WS={world_size}, CFG={cfg_parallel})...")

    tf_nxd = NxDModel.load(os.path.join(transformer_path, "nxd_model.pt"))
    tp_ckpts = load_sharded_weights(transformer_path, tp_degree)
    cp_ckpts = prepare_cp_checkpoints(tp_ckpts, tp_degree, cp_degree)
    del tp_ckpts
    tf_nxd.set_weights(cp_ckpts)
    del cp_ckpts
    tf_nxd.to_neuron()

    # Load RoPE
    rope_cache = torch.load(os.path.join(transformer_path, "rope_cache.pt"))
    rope_cos = rope_cache["rotary_emb_cos"].to(torch.bfloat16)
    rope_sin = rope_cache["rotary_emb_sin"].to(torch.bfloat16)
    print(f"  Transformer loaded in {time.time()-t0:.1f}s")
    print(f"  RoPE: cos={rope_cos.shape}, sin={rope_sin.shape}")

    # Prepare latents
    generator = torch.Generator().manual_seed(SEED)
    num_latent_frames = (num_frames - 1) // 4 + 1  # vae_scale_factor_temporal = 4
    latent_h = height // 16  # vae_scale_factor_spatial = 16
    latent_w = width // 16
    num_channels = transformer_config.in_channels  # 48
    latents = torch.randn(
        (1, num_channels, num_latent_frames, latent_h, latent_w),
        generator=generator, dtype=torch.float32
    )
    # I2V: prepare first_frame_mask and condition for per-position timestep approach
    first_frame_mask = None
    i2v_condition = None
    if image_condition is not None:
        # first_frame_mask: 1 for noisy positions (frames 1+), 0 for clean (frame 0)
        first_frame_mask = torch.ones(
            1, 1, num_latent_frames, latent_h, latent_w, dtype=torch.float32
        )
        first_frame_mask[:, :, 0] = 0
        i2v_condition = image_condition  # [1, C, 1, H, W] normalized latents
        print(f"  Latents: {latents.shape} (I2V: per-position timestep approach)")
    else:
        print(f"  Latents: {latents.shape} (T2V: pure noise)")

    # Set up scheduler
    scheduler.set_timesteps(num_steps, device=torch.device("cpu"))
    timesteps = scheduler.timesteps

    # Patch size and sequence length for timestep mask
    p_t, p_h, p_w = 1, 2, 2  # patch_size from model config
    ts_seq_len = num_latent_frames // p_t * (latent_h // p_h) * (latent_w // p_w)

    # Compute timestep_mask for I2V (post-patch positions)
    if first_frame_mask is not None:
        # Downsample first_frame_mask to post-patch seq positions
        # first_frame_mask: [1, 1, F, H, W] -> post-patch: [F//p_t, H//p_h, W//p_w]
        ts_mask = first_frame_mask[0, 0, ::p_t, ::p_h, ::p_w].flatten()  # [seq_len]
        ts_mask = ts_mask.unsqueeze(0).to(torch.float32)  # [1, seq_len]
    else:
        ts_mask = torch.ones(1, ts_seq_len, dtype=torch.float32)

    # Helper to run transformer forward
    def run_transformer(hidden_states, timestep, encoder_hidden_states):
        output = tf_nxd(
            hidden_states,
            timestep,
            encoder_hidden_states,
            rope_cos,
            rope_sin,
            ts_mask,
        )
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output

    # CFG Parallel: batch both calls
    if cfg_parallel:
        def denoise_step(latent_input_bf16, timestep_expanded):
            hs_batched = torch.cat([latent_input_bf16, latent_input_bf16], dim=0)
            enc_batched = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            ts_batched = torch.cat([timestep_expanded, timestep_expanded], dim=0) if timestep_expanded is not None else None
            output = run_transformer(hs_batched, ts_batched, enc_batched)
            noise_uncond = output[0:1]
            noise_cond = output[1:2]
            return noise_uncond + guidance_scale * (noise_cond - noise_uncond)
    else:
        def denoise_step(latent_input_bf16, timestep_expanded):
            noise_cond = run_transformer(latent_input_bf16, timestep_expanded, prompt_embeds)
            noise_uncond = run_transformer(latent_input_bf16, timestep_expanded, negative_prompt_embeds)
            return noise_uncond + guidance_scale * (noise_cond - noise_uncond)

    # Denoising loop
    print(f"  Starting {num_steps}-step denoising at {height}x{width}...")
    t_denoise_start = time.time()

    print(f"  Timestep sequence length: {ts_seq_len}")

    for i, t in enumerate(timesteps):
        step_start = time.time()

        if first_frame_mask is not None:
            # I2V: construct model input using first_frame_mask
            # latent_model_input = (1 - mask) * condition + mask * latents
            latent_model_input = (
                (1 - first_frame_mask) * i2v_condition + first_frame_mask * latents
            ).to(torch.bfloat16)
        else:
            # T2V: pure noise
            latent_model_input = latents.to(torch.bfloat16)

        # Scalar timestep [batch] - the compiled model handles per-position
        # conditioning internally using timestep_mask
        timestep_scalar = t.unsqueeze(0).to(torch.float32)  # [1]

        noise_pred = denoise_step(latent_model_input, timestep_scalar)

        # Cast to float32 for scheduler precision
        noise_pred = noise_pred.float()

        # Scheduler step
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        step_time = time.time() - step_start
        if i < 3 or (i + 1) % 10 == 0 or i == num_steps - 1:
            print(f"    Step {i+1}/{num_steps}: {step_time:.3f}s")

    # I2V: restore condition into final latents for decode
    if first_frame_mask is not None:
        latents = (1 - first_frame_mask) * i2v_condition + first_frame_mask * latents

    denoise_time = time.time() - t_denoise_start
    per_step = denoise_time / num_steps
    print(f"  Denoising complete: {denoise_time:.1f}s ({per_step:.3f}s/step)")

    # Free transformer
    del tf_nxd, rope_cos, rope_sin, rope_cache
    gc.collect()
    gc.collect()
    time.sleep(2)
    rss_gb = int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) / 1e6
    print(f"  Transformer freed. RSS: {rss_gb:.1f} GB")

    # ================================================================
    # Phase 3: Decode latents with PQC + tiled decoder
    # ================================================================
    print()
    print("=" * 60)
    print("Phase 3: Loading decoder and decoding latents...")
    print("=" * 60)
    t0 = time.time()

    # Denormalize latents (reverse the pipeline normalization)
    latents = latents.to(torch.float32)
    latents_mean = torch.tensor(vae_config.latents_mean).view(1, vae_config.z_dim, 1, 1, 1)
    latents_std_inv = 1.0 / torch.tensor(vae_config.latents_std).view(1, vae_config.z_dim, 1, 1, 1)
    latents = latents / latents_std_inv + latents_mean

    # Load PQC
    pqc_dir = f"{compiled_dir}/post_quant_conv"
    pqc_config = load_model_config(pqc_dir)
    pqc_ws = pqc_config.get("world_size", 4)
    pqc_nxd = NxDModel.load(os.path.join(pqc_dir, "nxd_model.pt"))
    pqc_weights = load_duplicated_weights(pqc_dir, pqc_ws)
    pqc_nxd.set_weights(pqc_weights)
    del pqc_weights
    pqc_nxd.to_neuron()
    pqc_wrapper = PostQuantConvWrapperV2(vae_pqc_cpu)
    pqc_wrapper.nxd_model = pqc_nxd
    print(f"  PQC loaded in {time.time()-t0:.1f}s")

    # Load tiled decoder
    t1 = time.time()
    decoder_dir = f"{compiled_dir}/decoder_tiled"
    dec_config = load_model_config(decoder_dir)
    dec_frames = dec_config.get("decoder_frames", 2)
    tile_h = dec_config["height"] // 16
    tile_w = dec_config["width"] // 16
    overlap = dec_config.get("overlap_latent", 4)
    dec_ws = dec_config.get("world_size", 4)

    dec_wrapper = DecoderWrapperV3Tiled(
        vae_decoder_cpu, decoder_frames=dec_frames,
        tile_h_latent=tile_h, tile_w_latent=tile_w, overlap_latent=overlap
    )
    dec_nxd = NxDModel.load(os.path.join(decoder_dir, "nxd_model.pt"))
    dec_weights = load_duplicated_weights(decoder_dir, dec_ws)
    dec_nxd.set_weights(dec_weights)
    del dec_weights
    dec_nxd.to_neuron()
    dec_wrapper.nxd_model = dec_nxd
    print(f"  Tiled decoder loaded in {time.time()-t1:.1f}s")
    print(f"    tile={tile_h}x{tile_w} latent, overlap={overlap}, frames={dec_frames}")

    # Run PQC on latents
    print("  Running post_quant_conv...")
    t2 = time.time()
    z = pqc_wrapper(latents.to(torch.float32))
    print(f"    PQC done in {time.time()-t2:.1f}s, z={z.shape}")

    # Free PQC (decoder needs all the memory it can get)
    del pqc_nxd, pqc_wrapper
    gc.collect()

    # Run tiled decoder
    print("  Running tiled decoder...")
    t3 = time.time()
    video_tensor = dec_wrapper.decode_latents(z)
    print(f"    Decode done in {time.time()-t3:.1f}s, shape={video_tensor.shape}")

    # Unpatchify (same as diffusers does)
    from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify
    if vae_config.patch_size is not None:
        video_tensor = unpatchify(video_tensor, patch_size=vae_config.patch_size)
    video_tensor = torch.clamp(video_tensor, min=-1.0, max=1.0)

    # Post-process to numpy frames
    # video_tensor: [B, C, T, H, W] in [-1, 1]
    video = (video_tensor[0].permute(1, 2, 3, 0).float().numpy() + 1.0) / 2.0
    video = (video * 255).clip(0, 255).astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in video]
    print(f"  Output: {len(frames)} frames at {frames[0].size}")

    # Save video
    output_path = args.output
    export_to_video(frames, output_path, fps=args.fps)
    total_time = time.time() - t_denoise_start  # From start of denoising
    print(f"\nVideo saved to: {output_path}")
    mode = "I2V" if args.image else "T2V"
    print(f"Mode: {mode}")
    print(f"Denoising: {denoise_time:.1f}s ({per_step:.3f}s/step)")
    print(f"Decode: {time.time()-t0:.1f}s")
    print(f"Total (denoise+decode): {total_time:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wan2.2 720P with model swapping")
    parser.add_argument("--compiled_models_dir", type=str, required=True)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--prompt", type=str, default="A cat walks on the grass, realistic")
    parser.add_argument("--negative_prompt", type=str,
                        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards")
    parser.add_argument("--image", type=str, default=None, help="Input image for I2V (omit for T2V)")
    parser.add_argument("--output", type=str, default="output_720p.mp4")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    args = parser.parse_args()
    main(args)
