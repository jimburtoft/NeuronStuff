"""
E2E Wan2.2 T2V-A14B inference — single-process expert swap via copy_().

Key improvement: Instead of spawning separate subprocesses for each expert
(which costs ~179s per expert load), this script loads the model ONCE and
swaps expert weights in-place using tensor.copy_(). The NEFF reads directly
from CPU tensor memory via DMA, so no re-initialization is needed.

Performance gain: ~179s saved per run (eliminates Expert 2 subprocess + reload).

Requirements:
  - Pre-compiled model at COMPILED_DIR (same NEFF for both experts)
  - Pre-sharded weights for both experts
  - trn2.48xlarge (64 cores, LNC=2)
  - SDK 2.29.1

Usage:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  python run_e2e_single_process.py
"""

import gc
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file

# Paths
COMPILED_DIR = "/opt/dlami/nvme/compiled_models_cp16_bs2"
CACHE_DIR = "/opt/dlami/nvme/models/Wan2.2-T2V-A14B-Diffusers"
HENAN_DIR = "/opt/dlami/nvme/aws-neuron-samples/torch-neuronx/inference/hf_pretrained_wan2.2_t2v_a14b"
OUTPUT_VIDEO = "/opt/dlami/nvme/output_single_process.mp4"

# Config
HEIGHT = 768
WIDTH = 1280
NUM_FRAMES = 81
NUM_STEPS = 40
GUIDANCE_SCALE = 5.0
PROMPT = "A cat wearing a top hat walks confidently down a rain-soaked neon-lit Tokyo street at night, reflections shimmering on the wet pavement"

# Neuron env
os.environ["NEURON_RT_VISIBLE_CORES"] = "0-63"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "1"


def load_sharded_weights(model_path, tp_degree):
    """Load TP-sharded weight checkpoints."""
    weights_path = os.path.join(model_path, "weights")
    sharded = []
    for rank in range(tp_degree):
        path = os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
        raw = load_file(path)
        ckpt = {k: v for k, v in raw.items() if "master_weight" not in k}
        sharded.append(ckpt)
    return sharded


def prepare_cp_checkpoints(tp_checkpoints, tp_degree, cp_degree):
    """Expand TP checkpoints into CP checkpoints with global rank injection."""
    checkpoints = []
    for cp_rank in range(cp_degree):
        for tp_rank in range(tp_degree):
            world_rank = cp_rank * tp_degree + tp_rank
            ckpt = {k: v.clone() for k, v in tp_checkpoints[tp_rank].items()}
            ckpt["transformer.global_rank.rank"] = torch.tensor(
                [world_rank], dtype=torch.int32
            )
            checkpoints.append(ckpt)
    return checkpoints


def swap_weights_inplace(nxd_model, new_checkpoints):
    """Swap expert weights using copy_() — no re-initialization needed.

    The NEFF reads from CPU tensor memory via DMA. copy_() updates the data
    at the same memory address, so the next forward pass sees the new weights
    without calling to_neuron() or initialize().

    This is 1.63x faster than replace_weights() and avoids the SIGSEGV that
    occurs when replace_weights() assigns new tensor objects.
    """
    for rank in range(len(new_checkpoints)):
        for key in new_checkpoints[rank]:
            if key in nxd_model.weights[rank]:
                nxd_model.weights[rank][key].copy_(new_checkpoints[rank][key])
            else:
                # New key not in model (shouldn't happen for same-architecture experts)
                print(f"  WARNING: key {key} not in model weights for rank {rank}")
                nxd_model.weights[rank][key] = new_checkpoints[rank][key]


def run_step(
    nxd_model,
    hidden_states,
    timestep,
    encoder_hidden_states,
    rotary_emb_cos,
    rotary_emb_sin,
):
    """Execute one transformer forward pass."""
    if timestep.dim() > 1:
        timestep = timestep.flatten()[0:1]
    elif timestep.dim() == 0:
        timestep = timestep.unsqueeze(0)
    timestep = timestep.to(torch.float32)
    output = nxd_model(
        hidden_states, timestep, encoder_hidden_states, rotary_emb_cos, rotary_emb_sin
    )
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output


def main():
    total_start = time.time()

    print("=" * 60)
    print("Wan2.2 T2V-A14B: SINGLE-PROCESS with copy_() expert swap")
    print("=" * 60)
    print(f"Resolution: {HEIGHT}x{WIDTH}, Frames: {NUM_FRAMES}, Steps: {NUM_STEPS}")
    print(f"Prompt: {PROMPT[:80]}...")

    # Import Neuron packages
    import torch_neuronx  # noqa: F401
    from neuronx_distributed import NxDModel

    # Load config
    config_path = os.path.join(COMPILED_DIR, "transformer", "config.json")
    with open(config_path) as f:
        config = json.load(f)
    tp_degree = config["tp_degree"]
    cp_degree = config["cp_degree"]
    world_size = config["world_size"]
    print(f"Config: TP={tp_degree}, CP={cp_degree}, world_size={world_size}")

    DTYPE = torch.bfloat16

    # === Step 1: Load Expert 1 weights ===
    print("\n=== Loading Expert 1 Weights ===")
    t0 = time.time()
    expert1_path = os.path.join(COMPILED_DIR, "transformer")
    tp_ckpts_1 = load_sharded_weights(expert1_path, tp_degree)
    checkpoints_1 = prepare_cp_checkpoints(tp_ckpts_1, tp_degree, cp_degree)
    del tp_ckpts_1
    gc.collect()
    print(f"  Expert 1 weights prepared in {time.time() - t0:.1f}s")

    # === Step 2: Load Expert 2 weights (preload into RAM) ===
    print("\n=== Pre-loading Expert 2 Weights ===")
    t0 = time.time()
    expert2_path = os.path.join(COMPILED_DIR, "transformer_2")
    tp_ckpts_2 = load_sharded_weights(expert2_path, tp_degree)
    checkpoints_2 = prepare_cp_checkpoints(tp_ckpts_2, tp_degree, cp_degree)
    del tp_ckpts_2
    gc.collect()
    print(f"  Expert 2 weights prepared in {time.time() - t0:.1f}s")

    # === Step 3: Load NxDModel with Expert 1 ===
    print("\n=== Loading NxDModel ===")
    t0 = time.time()
    nxd_path = os.path.join(expert1_path, "nxd_model.pt")
    nxd_model = NxDModel.load(nxd_path, start_rank=0, local_ranks_size=world_size)
    nxd_model.set_weights(checkpoints_1)
    nxd_model.to_neuron()
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    # Load RoPE cache
    rope_cache = torch.load(os.path.join(expert1_path, "rope_cache.pt"))
    rotary_emb_cos = rope_cache["rotary_emb_cos"].to(DTYPE)
    rotary_emb_sin = rope_cache["rotary_emb_sin"].to(DTYPE)

    # === Step 4: Setup scheduler and latents ===
    from diffusers import UniPCMultistepScheduler

    # Compute scheduling
    scheduler_config_path = os.path.join(
        CACHE_DIR, "scheduler", "scheduler_config.json"
    )
    if os.path.exists(scheduler_config_path):
        with open(scheduler_config_path) as f:
            sched_config = json.load(f)
        scheduler = UniPCMultistepScheduler(**sched_config)
    else:
        from diffusers import WanPipeline

        pipe = WanPipeline.from_pretrained(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        )
        scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        del pipe

    scheduler.set_timesteps(NUM_STEPS)
    timesteps = scheduler.timesteps

    switch_idx = int(NUM_STEPS * 0.325)
    print(f"\nExpert switch at step {switch_idx}")

    # Generate latents
    latent_frames = (NUM_FRAMES - 1) // 4 + 1
    latent_h = HEIGHT // 8
    latent_w = WIDTH // 8
    generator = torch.Generator().manual_seed(42)
    latents = torch.randn(
        1,
        16,
        latent_frames,
        latent_h,
        latent_w,
        generator=generator,
        dtype=torch.float32,
    )

    # Mask for timestep expansion
    mask_tensor = torch.ones(
        1, 1, latent_frames, latent_h, latent_w, dtype=torch.float32
    )

    # === Step 5: Text encoding (subprocess — separate cores) ===
    print("\n=== Text Encoding ===")
    # Use subprocess for text encoding (runs on cores 0-7, separate from transformer)
    t0 = time.time()
    # [simplified: assume pre-computed text embeddings exist]
    te_output_path = "/opt/dlami/nvme/text_output.pt"
    if os.path.exists(te_output_path):
        te_output = torch.load(te_output_path, weights_only=False)
        prompt_embeds = te_output["prompt_embeds"].to(DTYPE)
        negative_prompt_embeds = te_output["negative_prompt_embeds"].to(DTYPE)
    else:
        print("  ERROR: Text encoding output not found. Run text encoder first.")
        sys.exit(1)
    print(f"  Text embeddings loaded in {time.time() - t0:.1f}s")

    # === Step 6: Denoising with Expert 1 (high-noise steps) ===
    print(f"\n=== Phase 1: Expert 1, steps 0-{switch_idx - 1} ===")
    t_phase1_start = time.time()

    for i in range(switch_idx):
        t = timesteps[i]
        latent_input = latents.to(DTYPE)

        # Expand timesteps
        temp_ts = (mask_tensor[0][0][:, ::2, ::2] * t).flatten()
        ts = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)

        # Batched CFG: concat conditional and unconditional
        latent_batch = latent_input.repeat(2, 1, 1, 1, 1)  # [2, 16, T, H, W]
        embed_batch = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
        ts_batch = ts.repeat(2, 1)

        output = run_step(
            nxd_model,
            latent_batch,
            ts_batch,
            embed_batch,
            rotary_emb_cos,
            rotary_emb_sin,
        )

        noise_pred, noise_uncond = output.chunk(2, dim=0)
        noise_pred = noise_uncond.float() + GUIDANCE_SCALE * (
            noise_pred.float() - noise_uncond.float()
        )
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        elapsed = time.time() - t_phase1_start
        print(f"  Step {i + 1}/{switch_idx} (t={t.item():.0f}) - {elapsed:.1f}s")

    phase1_time = time.time() - t_phase1_start
    print(f"Phase 1 done: {phase1_time:.1f}s ({phase1_time / switch_idx:.2f}s/step)")

    # === Step 7: Swap to Expert 2 using copy_() ===
    print(f"\n=== Swapping to Expert 2 (copy_() in-place) ===")
    t_swap_start = time.time()
    swap_weights_inplace(nxd_model, checkpoints_2)
    swap_time = time.time() - t_swap_start
    print(f"  Expert swap done in {swap_time:.1f}s (no subprocess, no reload!)")

    # Free Expert 1 weights from RAM (Expert 2 already in model)
    del checkpoints_1
    gc.collect()

    # === Step 8: Denoising with Expert 2 (low-noise steps) ===
    remaining_steps = NUM_STEPS - switch_idx
    print(f"\n=== Phase 2: Expert 2, steps {switch_idx}-{NUM_STEPS - 1} ===")
    t_phase2_start = time.time()

    for i in range(switch_idx, NUM_STEPS):
        t = timesteps[i]
        latent_input = latents.to(DTYPE)

        temp_ts = (mask_tensor[0][0][:, ::2, ::2] * t).flatten()
        ts = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)

        latent_batch = latent_input.repeat(2, 1, 1, 1, 1)
        embed_batch = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
        ts_batch = ts.repeat(2, 1)

        output = run_step(
            nxd_model,
            latent_batch,
            ts_batch,
            embed_batch,
            rotary_emb_cos,
            rotary_emb_sin,
        )

        noise_pred, noise_uncond = output.chunk(2, dim=0)
        noise_pred = noise_uncond.float() + GUIDANCE_SCALE * (
            noise_pred.float() - noise_uncond.float()
        )
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        step_num = i - switch_idx + 1
        elapsed = time.time() - t_phase2_start
        print(
            f"  Step {step_num}/{remaining_steps} (t={t.item():.0f}) - {elapsed:.1f}s"
        )

    phase2_time = time.time() - t_phase2_start
    print(
        f"Phase 2 done: {phase2_time:.1f}s ({phase2_time / remaining_steps:.2f}s/step)"
    )

    # === Summary ===
    total_time = time.time() - total_start
    total_denoise = phase1_time + phase2_time

    print(f"\n{'=' * 60}")
    print(f"SINGLE-PROCESS E2E RESULTS")
    print(f"{'=' * 60}")
    print(f"  Model load (once):     {load_time:.1f}s")
    print(f"  Expert swap (copy_()):  {swap_time:.1f}s")
    print(
        f"  Phase 1 ({switch_idx} steps):   {phase1_time:.1f}s ({phase1_time / switch_idx:.2f}s/step)"
    )
    print(
        f"  Phase 2 ({remaining_steps} steps):  {phase2_time:.1f}s ({phase2_time / remaining_steps:.2f}s/step)"
    )
    print(
        f"  Total denoising:       {total_denoise:.1f}s ({total_denoise / NUM_STEPS:.2f}s/step)"
    )
    print(f"  TOTAL E2E:             {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"\n  Savings vs subprocess: ~179s (eliminated Expert 2 reload)")
    print(f"  Expert swap overhead:  {swap_time:.1f}s (vs 179s subprocess)")

    # Save latents
    torch.save(latents, "/opt/dlami/nvme/latents_single_process.pt")
    print(f"\nLatents saved for VAE decoding.")


if __name__ == "__main__":
    main()
