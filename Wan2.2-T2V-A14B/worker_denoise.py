"""
Wan 2.2 T2V-A14B Subprocess Worker
====================================
Runs denoising steps for ONE expert in a clean subprocess.
Called by run_a14b_tp4.py.

Neuron RT environment variables (core assignment, etc.) are passed
via the neuron_env dict from the orchestrator.

Usage:
    python worker_denoise.py <input_path> <output_path>
"""

import gc
import json
import os
import sys
import time

import torch
from safetensors.torch import load_file


def setup_neuron_env(env_dict):
    """Set Neuron environment variables from dict."""
    for k, v in env_dict.items():
        os.environ[k] = str(v)


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


def fix_norm_weights(sharded_weights, pipe_state_dict):
    """Fix norm_q/norm_k sharding mismatch.

    The compiler shards norm weights differently from attention weights.
    This re-slices them from the unsharded pipeline state dict.
    """
    hidden_size = 5120
    tp_degree = len(sharded_weights)
    ideal_norm_size = hidden_size // tp_degree
    expected_norm_size = None
    for key in sharded_weights[0]:
        if "norm_q.weight" in key:
            actual_size = sharded_weights[0][key].shape[0]
            if actual_size != ideal_norm_size and actual_size != hidden_size:
                expected_norm_size = actual_size
            break
    if expected_norm_size is None:
        return

    unsharded_norms = {}
    for key, value in pipe_state_dict.items():
        if "norm_k.weight" in key or "norm_q.weight" in key:
            unsharded_norms[f"transformer.{key}"] = value.clone()

    norm_tp = (
        unsharded_norms[list(unsharded_norms.keys())[0]].shape[0] // expected_norm_size
    )
    fixed = 0
    for rank in range(tp_degree):
        norm_rank = rank % norm_tp
        for norm_key, full_weight in unsharded_norms.items():
            if (
                norm_key in sharded_weights[rank]
                and sharded_weights[rank][norm_key].shape[0] != expected_norm_size
            ):
                start = norm_rank * expected_norm_size
                sharded_weights[rank][norm_key] = (
                    full_weight[start : start + expected_norm_size]
                    .to(sharded_weights[rank][norm_key].dtype)
                    .clone()
                )
                fixed += 1
    if fixed > 0:
        print(f"  Fixed {fixed} norm weights to size {expected_norm_size}")


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
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"[Worker] Loading input from {input_path}")
    data = torch.load(input_path, weights_only=False)

    # Setup Neuron env BEFORE importing torch_neuronx
    setup_neuron_env(data["neuron_env"])

    import torch_neuronx  # noqa: F401

    try:
        import torch_xla.core.xla_model as xm

        xm.mark_step = lambda *a, **kw: None
    except ImportError:
        pass
    from neuronx_distributed import NxDModel

    compiled_path = data["compiled_path"]
    config_path = os.path.join(compiled_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    tp_degree = config["tp_degree"]
    cp_degree = config["cp_degree"]
    world_size = config["world_size"]

    DTYPE = torch.bfloat16

    # Load weights
    print(f"[Worker] Loading weights (TP={tp_degree}, CP={cp_degree})...")
    t_load_start = time.time()
    tp_ckpts = load_sharded_weights(compiled_path, tp_degree)
    if data.get("pipe_transformer_state") is not None:
        fix_norm_weights(tp_ckpts, data["pipe_transformer_state"])
    checkpoints = prepare_cp_checkpoints(tp_ckpts, tp_degree, cp_degree)
    del tp_ckpts
    gc.collect()

    # Load NxDModel
    nxd_path = os.path.join(compiled_path, "nxd_model.pt")
    print(f"[Worker] Loading NxDModel...")
    nxd_model = NxDModel.load(nxd_path, start_rank=0, local_ranks_size=world_size)
    nxd_model.set_weights(checkpoints)
    del checkpoints
    gc.collect()
    nxd_model.to_neuron()
    load_time = time.time() - t_load_start
    print(f"[Worker] Model loaded in {load_time:.1f}s")

    # Load RoPE
    rope_cache = torch.load(os.path.join(compiled_path, "rope_cache.pt"))
    rotary_emb_cos = rope_cache["rotary_emb_cos"].to(DTYPE)
    rotary_emb_sin = rope_cache["rotary_emb_sin"].to(DTYPE)

    # Reconstruct scheduler
    from diffusers import UniPCMultistepScheduler

    scheduler = UniPCMultistepScheduler(**data["scheduler_config"])
    scheduler.set_timesteps(data["num_inference_steps"], device=torch.device("cpu"))

    if data.get("scheduler_state") is not None:
        state = data["scheduler_state"]
        for key, value in state.items():
            if hasattr(scheduler, key) and value is not None:
                setattr(scheduler, key, value)

    timesteps = scheduler.timesteps
    latents = data["latents"]
    prompt_embeds = data["prompt_embeds"].to(DTYPE)
    negative_prompt_embeds = data["negative_prompt_embeds"].to(DTYPE)
    step_start = data["step_start"]
    step_end = data["step_end"]
    guidance_scale = data["guidance_scale"]
    expand_timesteps = data["expand_timesteps"]
    mask = data["mask"]

    # Run denoising
    num_steps = step_end - step_start
    print(
        f"[Worker] Running {num_steps} denoising steps "
        f"(steps {step_start}-{step_end - 1})..."
    )
    t_denoise_start = time.time()

    for i in range(step_start, step_end):
        t = timesteps[i]
        latent_input = latents.to(DTYPE)

        if expand_timesteps:
            temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
            ts = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
        else:
            ts = t.expand(latents.shape[0])

        noise_pred = run_step(
            nxd_model, latent_input, ts, prompt_embeds, rotary_emb_cos, rotary_emb_sin
        )
        noise_uncond = run_step(
            nxd_model,
            latent_input,
            ts,
            negative_prompt_embeds,
            rotary_emb_cos,
            rotary_emb_sin,
        )
        noise_pred = noise_uncond.float() + guidance_scale * (
            noise_pred.float() - noise_uncond.float()
        )
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        step_num = i - step_start + 1
        elapsed = time.time() - t_denoise_start
        print(f"    Step {step_num}/{num_steps} (t={t.item():.0f}) - {elapsed:.1f}s")

    denoise_time = time.time() - t_denoise_start
    print(
        f"[Worker] Done: {num_steps} steps in {denoise_time:.1f}s "
        f"({denoise_time / num_steps:.1f}s/step)"
    )

    # Save scheduler state for next phase
    scheduler_state = {}
    for attr in [
        "order_list",
        "model_outputs",
        "timestep_list",
        "lower_order_nums",
        "sample",
    ]:
        scheduler_state[attr] = getattr(scheduler, attr, None)

    output = {
        "latents": latents.cpu(),
        "load_time": load_time,
        "denoise_time": denoise_time,
        "scheduler_state": scheduler_state,
    }
    torch.save(output, output_path)
    print(f"[Worker] Saved output to {output_path}")


if __name__ == "__main__":
    main()
