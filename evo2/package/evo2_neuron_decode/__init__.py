"""Autoregressive decode for Evo 2 / StripedHyena 2 on AWS Trainium.

WHAT THIS IS
    A set of fused NKI kernels plus the wiring that routes Arc Institute's `vortex`
    decode path through them. It makes `evo2_7b_base` generate on Trainium 2 at
    ~30 tok/s per logical core (~120 tok/s per trn2.3xlarge at DP=4).

WHAT THIS IS NOT
    Not a model re-implementation. `enable()` MONKEY-PATCHES vortex classes in place
    (`HyenaCascade.sequential_forward`, `ParallelGatedMLP.forward`, `RMSNorm.forward`,
    `MHA.forward`). That is a real coupling: an upstream vortex change can break it
    silently. It is pinned and verified against the commit in VENDORED.md, and
    `verify()` exists so you can check rather than trust.

    For TRAINING or forward-only embeddings, use canallee/evo2-neuron instead -- a clean
    re-implementation with a training stack. This package does decode, which that one
    explicitly does not.

QUICK START
    import evo2_neuron_decode as evo2d

    evo2d.enable(max_seqlen=512)            # patch vortex (defaults = measured best config)
    cfg = evo2d.prepare_config(cfg, 512)    # required config keys + Dynamo-traceable config
    model = StripedHyena(cfg)               # build + load Arc's checkpoint as usual
    model = evo2d.to_device(model, dtype=torch.bfloat16)   # NOT bare .to() -- see to_device
    evo2d.compile_blocks(model)             # NOT optional: 1.43x decode, 6.1x prefill

MEASURED (trn2.3xlarge, SDK 2.32 PyTorch Native, evo2_7b_base, bf16)
    decode  32.9 ms/step at BS=1 -> 30.4 tok/s;  BS=32 97.5 ms -> 328 tok/s
    DP=4    119.5 tok/s (BS=1) .. 1318 tok/s (BS=32) per instance
    prefill 0.610 ms/token at L=64, 0.218 at L=1024 (sub-linear)
    TTFT    71.9 ms at L=64
    parity  logits cos_sim 0.99999930 vs an independent fp32 CPU run; batch-invariant
            (BS=1/8/48/56 tokens identical); per-row identical at BS=4/32.
            NOT token-identical to the eager reference, and that is not achievable for
            this model: top1-top2 logit margins are 0.013-0.057 while bf16-vs-fp32 logit
            noise is 0.6367, so argmax flips are expected. Use logit parity, not tokens.

Single-request latency
    One logical core decodes at ~30 ms/token at BS=1. Data parallelism raises throughput but not
    per-request latency; this package targets batched throughput (TP=4 or DP=4 on one trn2 device).

REQUIRED, and not optional
    torch.compile ALL 32 blocks. Worth 1.43x on decode and 6.1x on prefill. See
    `compile_blocks()`. Partial compilation is slower than none.

CONSTRAINT
    max_seqlen must be divisible by 256 (attention_tkg shards s_prior by 2 and the
    shard must be a multiple of p_max=128). `enable()` validates this and raises with
    the next valid value rather than letting neuronx-cc emit NCC_INKI016.
"""

from .api import (enable, disable, is_enabled, compile_blocks, verify, config_summary,
                  to_device, prepare_config, make_config, MAX_SEQLEN_MULTIPLE)

__version__ = "0.1.0"
from .prefill import prefill, pick_bucket
__all__ = ["prefill", "pick_bucket", "enable", "disable", "is_enabled", "compile_blocks", "verify",
           "config_summary", "to_device", "prepare_config", "make_config",
           "MAX_SEQLEN_MULTIPLE", "__version__"]
