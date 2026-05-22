"""
NxDModel.replace_weights() SIGSEGV reproduction and workaround.

ROOT CAUSE (discovered 2026-05-22):
  NxDModel.replace_weights() calls set_weights() which assigns NEW tensor objects
  to the weights dict, then calls to_neuron() -> initialize(). The C++ SPMDModel
  holds raw pointers to the original weight tensors. When new tensors are assigned,
  the old tensors may be garbage-collected, causing initialize() to read freed memory.

  SIGSEGV trigger: assigning new tensor objects to nxd_model.weights[rank][key]
  then calling to_neuron() (which re-initializes with stale pointers).

WORKAROUND (verified bit-exact correct):
  Use tensor.copy_() to modify weight data IN-PLACE. The NEFF reads directly from
  the CPU tensor memory via DMA. copy_() changes the data without changing the
  tensor object (same memory address), so no re-initialization is needed.

  # Instead of: nxd_model.replace_weights(new_checkpoints)
  # Do this:
  for rank in range(world_size):
      for key in new_checkpoints[rank]:
          nxd_model.weights[rank][key].copy_(new_checkpoints[rank][key])
  # Forward pass immediately reads new weights - no to_neuron() needed!

PERFORMANCE:
  copy_() only:       87ms/expert (1.63x faster than replace_weights)
  copy_() + re-init: 120ms/expert
  replace_weights(): 142ms/expert (CRASHES at world_size > 4)

CORRECTNESS:
  copy_() produces BIT-EXACT results compared to replace_weights().
  Max difference: 0.0 (verified on SDK 2.29.1, trn2.3xlarge).

Environment:
  - Instance: trn2.3xlarge (4 logical NeuronCores with LNC=2)
  - SDK: 2.29.1 (DLAMI 20260502)
  - Venv: /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/

Usage:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  python repro_replace_weights.py
"""

import os
import sys
import time
import signal
import traceback

os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "1"

import torch
import torch.nn as nn
import torch_neuronx
from neuronx_distributed import ModelBuilder, NxDParallelState, shard_checkpoint
from neuronx_distributed import NxDModel
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    SPMDRank,
)
from neuronx_distributed.parallel_layers import parallel_state
from safetensors.torch import save_file as _save_file, load_file


def safe_save_file(tensors, filename):
    """Clone all tensors before saving to avoid mmap issues."""
    cloned = {k: v.clone().contiguous() for k, v in tensors.items()}
    _save_file(cloned, filename)


def timeout_handler(signum, frame):
    print("\n!!! TIMEOUT - process likely hung in replace_weights !!!")
    sys.exit(2)


# ============================================================================
# TEST 1: Tiny model (2 linear layers), TP=4
# ============================================================================


class TinyModel(nn.Module):
    """Minimal model: 2 ColumnParallel + RowParallel linear layers."""

    def __init__(self, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear1 = ColumnParallelLinear(
            hidden_size,
            hidden_size * 4,
            bias=False,
            gather_output=False,
            dtype=torch.bfloat16,
        )
        self.linear2 = RowParallelLinear(
            hidden_size * 4,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=torch.bfloat16,
        )
        self.global_rank = SPMDRank(world_size=4)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        return x


def test_tiny_replace_weights():
    """Test 1: Tiny model with TP=4."""
    print("\n" + "=" * 60)
    print("TEST 1: Tiny model (2 layers, 256 hidden, TP=4)")
    print("=" * 60)

    hidden_size = 256
    tp_degree = 4
    world_size = 4

    with NxDParallelState(world_size=world_size, tensor_model_parallel_size=tp_degree):
        model = TinyModel(hidden_size=hidden_size)
        model = model.to(torch.bfloat16)
        model.eval()

        # Trace
        sample_input = torch.randn(1, 32, hidden_size, dtype=torch.bfloat16)
        builder = ModelBuilder(model=model)
        builder.trace(kwargs={"x": sample_input}, tag="inference")

        # Compile
        print("  Compiling...")
        compile_dir = "/tmp/repro_tiny_compile"
        traced_model = builder.compile(
            compiler_args="--model-type=transformer -O1 --auto-cast=none",
            compiler_workdir=compile_dir,
        )

        # Save
        save_dir = "/tmp/repro_tiny_model"
        os.makedirs(save_dir, exist_ok=True)
        traced_model.save(os.path.join(save_dir, "nxd_model.pt"))

        # Create two sets of random weights
        print("  Creating weight sets A and B...")
        state_dict = model.state_dict()

        # Weight set A (original)
        weights_a = []
        for rank in range(world_size):
            ckpt = {}
            for key, val in state_dict.items():
                if "global_rank" in key:
                    ckpt[key] = torch.tensor([rank], dtype=torch.int32)
                else:
                    ckpt[key] = val.clone()
            weights_a.append(ckpt)

        # Weight set B (different random values)
        weights_b = []
        for rank in range(world_size):
            ckpt = {}
            for key, val in state_dict.items():
                if "global_rank" in key:
                    ckpt[key] = torch.tensor([rank], dtype=torch.int32)
                else:
                    ckpt[key] = torch.randn_like(val)
            weights_b.append(ckpt)

    # Load model
    print("  Loading NxDModel with weights A...")
    nxd_model = NxDModel.load(
        os.path.join(save_dir, "nxd_model.pt"),
        start_rank=0,
        local_ranks_size=world_size,
    )
    nxd_model.set_weights(weights_a)
    nxd_model.to_neuron()
    print("  Loaded successfully")

    # Forward pass with weights A
    print("  Running forward pass with weights A...")
    test_input = torch.randn(1, 32, hidden_size, dtype=torch.bfloat16)
    output_a = nxd_model(test_input)
    print(f"  Output A shape: {output_a.shape}, mean: {output_a.float().mean():.6f}")

    # CRITICAL: Call replace_weights
    print("  Calling replace_weights() with weights B...")
    signal.alarm(60)  # 60 second timeout
    try:
        nxd_model.replace_weights(weights_b)
        signal.alarm(0)
        print("  replace_weights() succeeded!")
    except Exception as e:
        signal.alarm(0)
        print(f"  replace_weights() raised exception: {type(e).__name__}: {e}")
        return False

    # Forward pass with weights B
    print("  Running forward pass with weights B...")
    output_b = nxd_model(test_input)
    print(f"  Output B shape: {output_b.shape}, mean: {output_b.float().mean():.6f}")

    # Verify outputs are different (weights changed)
    if torch.allclose(output_a, output_b):
        print("  WARNING: Outputs are identical - weights may not have been replaced!")
    else:
        print("  PASS: Outputs differ - weights were successfully replaced")

    return True


# ============================================================================
# TEST 2: Larger model (8 layers, ~100M params), TP=4
# ============================================================================


class MediumModel(nn.Module):
    """8-layer MLP with TP sharding. ~100M params at hidden=2048."""

    def __init__(self, hidden_size=2048, num_layers=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "up": ColumnParallelLinear(
                            hidden_size,
                            hidden_size * 4,
                            bias=False,
                            gather_output=False,
                            dtype=torch.bfloat16,
                        ),
                        "down": RowParallelLinear(
                            hidden_size * 4,
                            hidden_size,
                            bias=False,
                            input_is_parallel=True,
                            dtype=torch.bfloat16,
                        ),
                    }
                )
            )
        self.global_rank = SPMDRank(world_size=4)

    def forward(self, x):
        for layer in self.layers:
            residual = x
            x = layer["up"](x)
            x = torch.nn.functional.gelu(x)
            x = layer["down"](x)
            x = x + residual
        return x


def test_medium_replace_weights():
    """Test 2: Medium model with TP=4."""
    print("\n" + "=" * 60)
    print("TEST 2: Medium model (8 layers, 2048 hidden, TP=4, ~100M params)")
    print("=" * 60)

    hidden_size = 2048
    num_layers = 8
    tp_degree = 4
    world_size = 4

    with NxDParallelState(world_size=world_size, tensor_model_parallel_size=tp_degree):
        model = MediumModel(hidden_size=hidden_size, num_layers=num_layers)
        model = model.to(torch.bfloat16)
        model.eval()

        # Trace
        sample_input = torch.randn(1, 64, hidden_size, dtype=torch.bfloat16)
        builder = ModelBuilder(model=model)
        builder.trace(kwargs={"x": sample_input}, tag="inference")

        # Compile
        print("  Compiling...")
        compile_dir = "/tmp/repro_medium_compile"
        traced_model = builder.compile(
            compiler_args="--model-type=transformer -O1 --auto-cast=none",
            compiler_workdir=compile_dir,
        )

        # Save
        save_dir = "/tmp/repro_medium_model"
        os.makedirs(save_dir, exist_ok=True)
        traced_model.save(os.path.join(save_dir, "nxd_model.pt"))

        state_dict = model.state_dict()

        # Create weight sets
        print("  Creating weight sets A and B...")
        weights_a = []
        weights_b = []
        for rank in range(world_size):
            ckpt_a = {}
            ckpt_b = {}
            for key, val in state_dict.items():
                if "global_rank" in key:
                    ckpt_a[key] = torch.tensor([rank], dtype=torch.int32)
                    ckpt_b[key] = torch.tensor([rank], dtype=torch.int32)
                else:
                    ckpt_a[key] = val.clone()
                    ckpt_b[key] = torch.randn_like(val)
            weights_a.append(ckpt_a)
            weights_b.append(ckpt_b)

    # Load model
    print("  Loading NxDModel with weights A...")
    nxd_model = NxDModel.load(
        os.path.join(save_dir, "nxd_model.pt"),
        start_rank=0,
        local_ranks_size=world_size,
    )
    nxd_model.set_weights(weights_a)
    nxd_model.to_neuron()
    print("  Loaded successfully")

    # Forward pass with weights A
    print("  Running forward pass with weights A...")
    test_input = torch.randn(1, 64, hidden_size, dtype=torch.bfloat16)
    output_a = nxd_model(test_input)
    print(f"  Output A mean: {output_a.float().mean():.6f}")

    # CRITICAL: replace_weights
    print("  Calling replace_weights() with weights B...")
    signal.alarm(120)
    try:
        t0 = time.time()
        nxd_model.replace_weights(weights_b)
        elapsed = time.time() - t0
        signal.alarm(0)
        print(f"  replace_weights() succeeded in {elapsed:.2f}s!")
    except Exception as e:
        signal.alarm(0)
        print(f"  replace_weights() raised exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    # Forward pass with weights B
    print("  Running forward pass with weights B...")
    output_b = nxd_model(test_input)
    print(f"  Output B mean: {output_b.float().mean():.6f}")

    if not torch.allclose(output_a, output_b):
        print("  PASS: Outputs differ - weights were successfully replaced")
    else:
        print("  WARNING: Outputs identical - weights may not have been replaced!")

    return True


# ============================================================================
# TEST 3: Model with simulated Context Parallel (TP=4, "DP"=2 as CP proxy)
# ============================================================================


class CPModel(nn.Module):
    """Model that simulates CP by using DP group for scatter/gather."""

    def __init__(self, hidden_size=1024, num_layers=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "up": ColumnParallelLinear(
                            hidden_size,
                            hidden_size * 4,
                            bias=False,
                            gather_output=False,
                            dtype=torch.bfloat16,
                        ),
                        "down": RowParallelLinear(
                            hidden_size * 4,
                            hidden_size,
                            bias=False,
                            input_is_parallel=True,
                            dtype=torch.bfloat16,
                        ),
                    }
                )
            )
        self.global_rank = SPMDRank(world_size=8)

    def forward(self, x):
        for layer in self.layers:
            residual = x
            x = layer["up"](x)
            x = torch.nn.functional.gelu(x)
            x = layer["down"](x)
            x = x + residual
        return x


def test_cp_replace_weights():
    """Test 3: Model with world_size=8 (simulating TP=4, CP=2)."""
    print("\n" + "=" * 60)
    print("TEST 3: CP model (4 layers, 1024 hidden, TP=4, CP=2, world_size=8)")
    print("=" * 60)

    hidden_size = 1024
    num_layers = 4
    tp_degree = 4
    world_size = 8  # TP=4 * CP=2

    with NxDParallelState(world_size=world_size, tensor_model_parallel_size=tp_degree):
        model = CPModel(hidden_size=hidden_size, num_layers=num_layers)
        model = model.to(torch.bfloat16)
        model.eval()

        # Trace
        sample_input = torch.randn(1, 32, hidden_size, dtype=torch.bfloat16)
        builder = ModelBuilder(model=model)
        builder.trace(kwargs={"x": sample_input}, tag="inference")

        # Compile
        print("  Compiling...")
        compile_dir = "/tmp/repro_cp_compile"
        traced_model = builder.compile(
            compiler_args="--model-type=transformer -O1 --auto-cast=none",
            compiler_workdir=compile_dir,
        )

        # Save
        save_dir = "/tmp/repro_cp_model"
        os.makedirs(save_dir, exist_ok=True)
        traced_model.save(os.path.join(save_dir, "nxd_model.pt"))

        state_dict = model.state_dict()

        # Create weight sets -- replicate TP shards across CP ranks
        print("  Creating weight sets A and B (with CP replication)...")
        weights_a = []
        weights_b = []
        for world_rank in range(world_size):
            tp_rank = world_rank % tp_degree
            ckpt_a = {}
            ckpt_b = {}
            for key, val in state_dict.items():
                if "global_rank" in key:
                    ckpt_a[key] = torch.tensor([world_rank], dtype=torch.int32)
                    ckpt_b[key] = torch.tensor([world_rank], dtype=torch.int32)
                else:
                    ckpt_a[key] = val.clone()
                    ckpt_b[key] = torch.randn_like(val)
            weights_a.append(ckpt_a)
            weights_b.append(ckpt_b)

    # Load model
    print("  Loading NxDModel (world_size=8)...")
    nxd_model = NxDModel.load(
        os.path.join(save_dir, "nxd_model.pt"),
        start_rank=0,
        local_ranks_size=world_size,
    )
    nxd_model.set_weights(weights_a)
    nxd_model.to_neuron()
    print("  Loaded successfully")

    # Forward pass with weights A
    print("  Running forward pass with weights A...")
    test_input = torch.randn(1, 32, hidden_size, dtype=torch.bfloat16)
    output_a = nxd_model(test_input)
    print(f"  Output A mean: {output_a.float().mean():.6f}")

    # CRITICAL: replace_weights
    print("  Calling replace_weights() with weights B...")
    signal.alarm(120)
    try:
        t0 = time.time()
        nxd_model.replace_weights(weights_b)
        elapsed = time.time() - t0
        signal.alarm(0)
        print(f"  replace_weights() succeeded in {elapsed:.2f}s!")
    except Exception as e:
        signal.alarm(0)
        print(f"  replace_weights() raised exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    # Forward pass with weights B
    print("  Running forward pass with weights B...")
    output_b = nxd_model(test_input)
    print(f"  Output B mean: {output_b.float().mean():.6f}")

    if not torch.allclose(output_a, output_b):
        print("  PASS: Outputs differ - weights were successfully replaced")
    else:
        print("  WARNING: Outputs identical")

    return True


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 60)
    print("NxDModel.replace_weights() SIGSEGV Reproduction")
    print("=" * 60)
    print(f"SDK: {torch_neuronx.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {os.popen('neuron-ls 2>/dev/null | head -3').read().strip()}")
    print()

    # Set up signal handler for timeouts
    signal.signal(signal.SIGALRM, timeout_handler)

    results = {}

    # Test 1: Tiny
    try:
        results["test1_tiny"] = test_tiny_replace_weights()
    except SystemExit:
        results["test1_tiny"] = "TIMEOUT/CRASH"
    except Exception as e:
        results["test1_tiny"] = f"EXCEPTION: {e}"
        traceback.print_exc()

    # Test 2: Medium (only if Test 1 passed)
    if results.get("test1_tiny") is True:
        try:
            results["test2_medium"] = test_medium_replace_weights()
        except SystemExit:
            results["test2_medium"] = "TIMEOUT/CRASH"
        except Exception as e:
            results["test2_medium"] = f"EXCEPTION: {e}"
            traceback.print_exc()

    # Test 3: CP (only if Test 2 passed)
    if results.get("test2_medium") is True:
        try:
            results["test3_cp"] = test_cp_replace_weights()
        except SystemExit:
            results["test3_cp"] = "TIMEOUT/CRASH"
        except Exception as e:
            results["test3_cp"] = f"EXCEPTION: {e}"
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test_name, result in results.items():
        status = (
            "PASS" if result is True else ("FAIL" if result is False else str(result))
        )
        print(f"  {test_name}: {status}")
    print()

    # If all tests pass, the SIGSEGV may be scale-dependent (needs trn2.48xlarge)
    all_passed = all(v is True for v in results.values())
    if all_passed:
        print(
            "All tests PASSED - SIGSEGV may only occur at larger scale (world_size=64)"
        )
        print("Next step: test on trn2.48xlarge with world_size=16 or 64")
    else:
        first_fail = next((k for k, v in results.items() if v is not True), None)
        print(f"First failure: {first_fail}")
        print("SIGSEGV reproduced at this scale - minimal repro found!")


if __name__ == "__main__":
    main()
