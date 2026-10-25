"""The public API. One `enable()` replaces eleven EVO2_* environment flags.

Each optimization below is gated on a measurement, and the default is the configuration
that was actually verified end to end. Where an option is OFF by default it is because it
was measured to not help -- the reason is recorded inline so nobody re-litigates it.
"""

from __future__ import annotations

import os
import warnings

# attention_tkg shards s_prior by 2; the SHARD must be a multiple of p_max=128.
MAX_SEQLEN_MULTIPLE = 256

_STATE: dict = {"enabled": False, "installed": [], "opts": {}}

# name -> (default, one-line rationale tied to a measurement)
_OPTIONS = {
    "casts":        (True,  "cast_reduce: 1.05x, removes redundant dtype churn"),
    "state_concat": (True,  "1.036x at BS>1 only; no-op at BS=1"),
    "cascade":      (True,  "HCL cascade kernel, ~19 ops -> 1 launch, 5.09x isolated"),
    "hcm_cascade":  (True,  "HCM/HCS cascade, 18 of 32 layers, 12.90x isolated"),
    "fused_norm":   (True,  "fused RMSNorm, 1.59x at BS=1 (was 38.7% of the step)"),
    "mlp_cat":      (True,  "l1+l2 into one GEMV, ~1% (7/7 pairs, real but small)"),
    "attention_tkg":(True,  "nkilib attention_tkg + BH tiling, 1.72x, no batch cap"),
    "traceable_cfg":(True,  "TraceableConfig: Dynamo graph breaks 139 -> 23 at BS=32"),
    # --- measured NOT to help; left as knobs only so the negative result is discoverable ---
    "fixed_kv":     (False, "REJECTED: no measurable gain"),
    "state_tensor": (False, "REJECTED: 4.4x SLOWER"),
    "fir_weight_cache": (False, "REJECTED: within noise, and a sign-flip risk"),
    "offset_tensor":(False, "REJECTED: no gain"),
}


def config_summary() -> str:
    """Human-readable summary of what is enabled and why. Print this in issue reports."""
    lines = ["evo2_neuron_decode configuration:"]
    opts = _STATE["opts"] or {k: v[0] for k, v in _OPTIONS.items()}
    for name, (default, why) in _OPTIONS.items():
        on = opts.get(name, default)
        lines.append(f"  {'ON ' if on else 'off'}  {name:<16} {why}")
    lines.append(f"  enabled={_STATE['enabled']}  installed={len(_STATE['installed'])} patches")
    return "\n".join(lines)


def _check_max_seqlen(max_seqlen: int | None) -> None:
    if max_seqlen is None:
        return
    if max_seqlen % MAX_SEQLEN_MULTIPLE != 0:
        nxt = ((max_seqlen + MAX_SEQLEN_MULTIPLE - 1) // MAX_SEQLEN_MULTIPLE) * MAX_SEQLEN_MULTIPLE
        raise ValueError(
            f"max_seqlen must be divisible by {MAX_SEQLEN_MULTIPLE} (got {max_seqlen}). "
            f"attention_tkg shards s_prior by 2 and the shard must be a multiple of "
            f"p_max=128, so a bad value fails at COMPILE time with NCC_INKI016 "
            f"'Sharded s_prior must be divisible by p_max' (yours would shard to "
            f"{max_seqlen // 2}). Use max_seqlen={nxt}, or enable(attention_tkg=False) "
            f"to fall back to eager MHA with no such constraint."
        )


def enable(*, max_seqlen: int | None = None, **overrides) -> dict:
    """Patch vortex for Trainium decode. Returns the effective option dict.

    Defaults are the measured best configuration. Pass keyword overrides to change
    one, e.g. ``enable(attention_tkg=False)``. Call ``config_summary()`` to see them.

    Args:
        max_seqlen: if given, validated against the divisible-by-256 constraint now
            rather than at kernel-compile time. Strongly recommended.
        **overrides: any key in ``_OPTIONS``.
    """
    unknown = set(overrides) - set(_OPTIONS)
    if unknown:
        raise TypeError(
            f"unknown option(s) {sorted(unknown)}; valid: {sorted(_OPTIONS)}")

    _check_max_seqlen(max_seqlen)

    if _STATE["enabled"]:
        warnings.warn("evo2_neuron_decode.enable() called twice; ignoring the second call. "
                      "Call disable() first to change options.", RuntimeWarning)
        return dict(_STATE["opts"])

    opts = {k: v[0] for k, v in _OPTIONS.items()}
    opts.update(overrides)

    from . import patches as P

    P.neuron_shim.install()
    # The attention region (EVO2_ATTN_OP) is torch.compile'd per layer and per batch shape
    # even when compile_blocks() is not called; with Dynamo's default limit of 8 the 5 attention
    # layers x a second batch size exceed it and Dynamo falls back to eager, which then fails on the
    # nki_op (found on SDK 2.33 by test_batch_size_sweep_in_one_process). Raise the limits here too.
    import torch._dynamo as _dyn
    _dyn.config.cache_size_limit = max(getattr(_dyn.config, "cache_size_limit", 8), 65536)
    _dyn.config.accumulated_cache_size_limit = max(
        getattr(_dyn.config, "accumulated_cache_size_limit", 256), 1048576)

    installed = []
    # Always-on correctness/lowering patches: these are what make the model run at all
    # on this backend, not optimizations.
    for mod in (P.fir_conv_patch, P.iir_recurrence_patch, P.rmsnorm_patch, P.rotary_patch):
        mod.install(); installed.append(mod.__name__.rsplit(".", 1)[-1])

    order = [
        ("casts",         P.cast_reduce),
        ("state_concat",  P.state_concat),
        ("cascade",       P.cascade_patch),
        ("hcm_cascade",   P.hcm_patch),
        ("fused_norm",    P.norm_patch),
        ("mlp_cat",       P.mlp_cat_patch),
        ("attention_tkg", P.tkg_mha2),
    ]
    for key, mod in order:
        if opts.get(key):
            mod.install(); installed.append(key)

    _STATE.update(enabled=True, installed=installed, opts=opts)
    return dict(opts)


def make_config(cfg):
    """Wrap a vortex `dotdict` config so Dynamo can trace attribute reads.

    vortex's config is a ``dotdict`` whose ``__getattr__`` IS ``dict.get`` -- a C builtin
    Dynamo cannot trace, so **every** ``config.x`` read inside a compiled region breaks the
    graph. Measured: 89 of 111 graph breaks. This wrapper drops it to 23.

    Preserves ``dotdict`` semantics exactly, including returning None for missing keys --
    a plain object would raise AttributeError and change behaviour, not just tracing.
    """
    if not _STATE["opts"].get("traceable_cfg", True):
        return cfg
    from . import patches as P
    return P.dotdict_patch.make(cfg)


def to_device(model, device="neuron", dtype=None):
    """Move a vortex model to Neuron **completely**. Use this instead of bare ``.to()``.

    ``nn.Module.to()`` moves parameters and registered buffers. vortex additionally stores
    plain tensor ATTRIBUTES on its modules (e.g. filter/pole state set in ``__init__``) which
    ``.to()`` does NOT touch, and it keeps a ``block_idx_to_device`` map that must be updated.
    Miss either and the first forward dies with::

        RuntimeError: input tensor is on cpu device, expected neuron

    That is a genuinely non-obvious setup step, so it lives here rather than in every caller's
    boilerplate.
    """
    import torch

    dev = torch.device(device)
    if dtype is not None:
        model = model.to(dtype)
    model = model.to(dev)

    if hasattr(model, "block_idx_to_device"):
        model.block_idx_to_device = {k: dev for k in model.block_idx_to_device}

    moved = 0
    for mod in model.modules():
        for name, val in list(vars(mod).items()):
            if torch.is_tensor(val) and val.device.type != dev.type:
                setattr(mod, name, val.to(dev)); moved += 1
        for name, val in list(mod._buffers.items()):
            if torch.is_tensor(val) and val.device.type != dev.type:
                mod._buffers[name] = val.to(dev); moved += 1
    _STATE["moved_tensors"] = moved

    # De-duplicate the MLP up/gate weights ON THE DEVICE, before compile_blocks()
    # traces them. mlp_cat_patch concatenates l1/l2 into _w12; doing that lazily inside the
    # first forward left ~33 of 64 standalone l1/l2 device tensors resident alongside every
    # _w12 (per-allocation log: 65 x 86 MiB single MLP weights where 32 belong).
    # This must run AFTER .to(dev): .to() copies parameters, which breaks storage sharing.
    _dedup_mlp(model)
    _pretranspose(model)
    return model


_PRET_CLS = None


def _pret_cls():
    """Built lazily so importing evo2_neuron_decode (e.g. for its NKI kernels in a torch-free
    simulator venv) does not require torch."""
    global _PRET_CLS
    if _PRET_CLS is None:
        import torch
        class _PreTLinear(torch.nn.Module):
            """y = x @ wt (+ b), wt = W.t() stored contiguous. With W in (out, in) layout the
            compiler emits ~80% of a decode GEMV's TensorEngine FLOPs as weight TRANSPOSE matmuls
            (skill profile); with (in, out) it emits 4-12%. Output bit-identical (probe, 4 shapes)."""

            def __init__(self, weight, bias, ret_tuple=False, te_return_bias=False):
                import torch
                super().__init__()
                self.wt = torch.nn.Parameter(weight.detach().t().contiguous(), requires_grad=False)
                self.bias = None if bias is None else torch.nn.Parameter(bias.detach(), requires_grad=False)
                self.in_features, self.out_features = self.wt.shape
                self._ret_tuple = ret_tuple
                self.te_return_bias = te_return_bias

            def forward(self, x):
                y = x.to(self.wt.dtype) @ self.wt
                if self.bias is not None and not self.te_return_bias:
                    y = y + self.bias
                if self._ret_tuple:
                    return y, (self.bias if self.te_return_bias else None)
                return y
        _PRET_CLS = _PreTLinear
    return _PRET_CLS


def _pretranspose(model):
    """EVO2_WT=1: store every decode GEMV weight transposed (in, out) and replace the module.
    The original (out, in) weight is released, so device memory does not grow."""
    import os
    import torch
    # DEFAULT ON. TP=4 @ 8K, gated, alternating: B=48 43.98 -> 40.85 ms (3 pairs,
    # 1.077x, 1091 -> 1175 tok/s); B=64 49.16 -> 47.03; B=32 36.68 -> 34.94 / 36.37 -> 34.30.
    # Skill profile: TRANSPOSE share of cascade-graph TE FLOPs 80% -> 8-28%, TE active
    # 0.26-0.31 -> 0.12-0.18 ms/graph. 48-step greedy tokens identical on both prompts and at
    # B=32/48/64; step-1 logits differ at ~1e-5 relative (different matmul lowering). EVO2_WT=0 off.
    if os.environ.get("EVO2_WT", "1") != "1":
        return 0
    from .tp import RowParallelLinear
    _PreTLinear = _pret_cls()

    def _t(w):
        # Transpose via the host so the device never holds both layouts of one weight, and drop
        # the device original before allocating the new one (single core at 8K is ~full; the
        # first version kept every original alive until the loop ended and failed to load NEFFs).
        dev = w.device
        h = w.detach().to("cpu").t().contiguous()
        return h, dev

    n = 0
    targets = []
    for parent in model.modules():
        for name, child in parent.named_children():
            if isinstance(child, RowParallelLinear) or (type(child).__name__ == "TELinear"
                    and getattr(child, "weight", None) is not None) or (
                    type(child) is torch.nn.Linear and name in ("Wqkv", "out_proj", "out_filter_dense", "l3")):
                targets.append((parent, name))
    for parent, name in targets:
        child = getattr(parent, name)
        h, dev = _t(child.weight)
        bias = getattr(child, "bias", None)
        if isinstance(child, RowParallelLinear):
            child.weight = None
            child.wt = torch.nn.Parameter(h.to(dev), requires_grad=False)
        else:
            te = type(child).__name__ == "TELinear"
            trb = bool(getattr(child, "te_return_bias", False))
            child.weight = None
            new = _PreTLinear.__new__(_PreTLinear)
            torch.nn.Module.__init__(new)
            new.wt = torch.nn.Parameter(h.to(dev), requires_grad=False)
            new.bias = None if bias is None else torch.nn.Parameter(bias.detach(), requires_grad=False)
            new.in_features, new.out_features = new.wt.shape
            new._ret_tuple = te
            new.te_return_bias = trb
            setattr(parent, name, new)
        del child, h
        n += 1
    # The fused-MLP path (mlp_cat_patch) is the only reader of _w12t. If mlp_cat is off, stock
    # ParallelGatedMLP.forward calls l1/l2 directly, so their weights must stay (found on SDK 2.33:
    # test_greedy_parity_vs_reference's reference arm failed "vec (0)" after the pre-transpose change).
    _mlp_cat_on = bool(_STATE.get("opts", {}).get("mlp_cat", True)) if _STATE.get("enabled") else True
    for mod in (model.modules() if _mlp_cat_on else ()):
        w12 = getattr(mod, "_w12", None)
        if isinstance(w12, torch.Tensor):
            dev, dt = w12.device, w12.dtype
            h = w12.to("cpu").t().contiguous()
            for ln in ("l1", "l2"):
                lin = getattr(mod, ln, None)
                if lin is not None:
                    lin.weight = torch.nn.Parameter(torch.empty(0, device=dev, dtype=dt), requires_grad=False)
            mod._w12 = None
            del w12
            mod._w12t = h.to(dev)
            del h
            n += 1
    import gc
    gc.collect()
    ec = getattr(getattr(torch, "neuron", None), "empty_cache", None)
    if callable(ec):
        ec()
    _STATE["pretransposed"] = n
    return n


def _dedup_mlp(model):
    """Build each ParallelGatedMLP's _w12 now and point l1/l2 at slices of it, so the
    up/gate weights occupy ONE device allocation per block. On by default; EVO2_MLP_ALIAS=0 disables."""
    import os
    import torch
    # DEFAULT ON: load-time de-dup measured on hardware -- frees 5.55 GB of
    # duplicated MLP weight at the B=8 @ 8K failure point (live tensors 23.06 -> ~17.7 GB in
    # the runtime table), leaves exactly 32 standalone l3 weights, and step-1 decode logits are
    # BIT-IDENTICAL to the un-aliased model. EVO2_MLP_ALIAS=0 restores the old layout.
    if os.environ.get("EVO2_MLP_ALIAS", "1") != "1":
        return 0
    n = 0
    for mod in model.modules():
        l1, l2 = getattr(mod, "l1", None), getattr(mod, "l2", None)
        if not (isinstance(l1, torch.nn.Linear) and isinstance(l2, torch.nn.Linear)
                and hasattr(mod, "l3")):
            continue
        if l1.bias is not None or l2.bias is not None:
            continue
        w12 = torch.cat([l1.weight.detach(), l2.weight.detach()], 0).contiguous()
        I_ = l1.out_features
        l1.weight = torch.nn.Parameter(w12[:I_], requires_grad=False)
        l2.weight = torch.nn.Parameter(w12[I_:], requires_grad=False)
        mod._w12 = w12          # mlp_cat_patch then finds it already built
        n += 1
    _STATE["mlp_dedup_blocks"] = n
    return n


def prepare_config(cfg, max_seqlen: int):
    """Set the config keys this backend REQUIRES, and validate max_seqlen.

    ``use_flash_attn=False`` is not a preference: vortex's ``MHA.__init__`` hard-asserts
    *"FlashAttention is not installed"* on the flash branch (attention.py:70) and flash-attn is
    CUDA-only, so the flash path cannot be constructed here at all.
    """
    _check_max_seqlen(max_seqlen)
    cfg.max_seqlen = max_seqlen
    cfg.use_flash_attn = False          # MHA.__init__ asserts otherwise; CUDA-only
    cfg.use_flashfft = False
    cfg.use_flash_depthwise = False
    cfg.use_flash_rmsnorm = False
    cfg.prefill_style = "recurrence"
    cfg.inference_mode = True
    return make_config(cfg)


def compile_blocks(model, n: int | None = None):
    """``torch.compile`` the transformer blocks. NOT OPTIONAL -- see the module docstring.

    Worth 1.43x on decode and 6.1x on prefill. Compile ALL blocks or none: partial
    compilation measured SLOWER than eager.

    Uses ``block.compile(...)`` rather than ``blocks[i] = torch.compile(blocks[i])`` --
    the latter inserts ``_orig_mod.`` into parameter names and breaks checkpoint loading.
    """
    import torch
    import torch._dynamo

    # Default 8 is far below what 32 blocks need. Exceeding it is loudly warned, not
    # silent, on torch 2.11-2.13 -- but raise it anyway. NOTE cache_size_limit and
    # recompile_limit are a bidirectional alias for one value; either name works.
    torch._dynamo.config.cache_size_limit = max(
        getattr(torch._dynamo.config, "cache_size_limit", 8), 65536)
    torch._dynamo.config.accumulated_cache_size_limit = max(
        getattr(torch._dynamo.config, "accumulated_cache_size_limit", 256), 1048576)

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise AttributeError("model has no .blocks; pass the StripedHyena inner model")
    total = len(blocks)
    n = total if n is None else min(n, total)
    if 0 < n < total:
        warnings.warn(
            f"compiling {n} of {total} blocks; PARTIAL COMPILATION MEASURED SLOWER THAN "
            f"EAGER. Compile all or none.", RuntimeWarning)
    for i in range(n):
        blocks[i].compile(backend="neuron", dynamic=False)
    return model


def verify(model, *, steps: int = 6, bs: int = 1) -> dict:
    """Greedy-token parity of the patched model against an unpatched reference.

    This is the gate that matters. Note deliberately that end-to-end **logits cos_sim** is
    NOT used as the verdict: the residual stream grows ~2e14x across 32 blocks, so an early
    block's contribution to the final logits is ~1e-14 relative and a broken block can read
    cos_sim 1.000000. Discrete token equality does not dilute. Per-kernel parity (the
    simulator tests) is the other half.

    Returns a dict with 'tokens' and 'ok'.
    """
    raise NotImplementedError(
        "verify() needs a reference run and is provided by tests/test_hardware.py::"
        "test_greedy_parity_vs_unpatched, which has the harness to build both models. "
        "Kept in the API surface as the documented gate.")


# Reverse install order. dotdict_patch is a FACTORY (make() returns a new object; it mutates
# nothing) and neuron_shim only stubs absent CUDA modules, so neither has anything to undo --
# that is by design, not an omission.
_UNINSTALL_ORDER = ("tkg_mha2", "mlp_cat_patch", "norm_patch", "hcm_patch", "cascade_patch",
                    "state_concat", "cast_reduce", "rotary_patch", "rmsnorm_patch",
                    "iir_recurrence_patch", "fir_conv_patch")
_NO_UNINSTALL = ("dotdict_patch", "neuron_shim")


def disable(*, strict: bool = False) -> list[str]:
    """Undo `enable()`. Returns the names of patches that were reverted.

    Every patch that mutates vortex provides ``uninstall()``; ``dotdict_patch`` and
    ``neuron_shim`` do not because they mutate nothing (see ``_NO_UNINSTALL``).

    Args:
        strict: raise if any ``uninstall()`` fails. Default warns and continues, so one bad
            revert cannot leave the rest of the patches installed.
    """
    if not _STATE["enabled"]:
        return []
    from . import patches as P

    reverted, failed = [], []
    for name in _UNINSTALL_ORDER:
        mod = getattr(P, name)
        fn = getattr(mod, "uninstall", None)
        if fn is None:
            failed.append((name, "no uninstall()")); continue
        try:
            fn(); reverted.append(name)
        except Exception as e:
            failed.append((name, repr(e)))
    if failed:
        msg = "disable() could not revert: " + ", ".join(f"{n} ({w})" for n, w in failed)
        if strict:
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning)
    _STATE.update(enabled=False, installed=[], opts={})
    return reverted


def is_enabled() -> bool:
    """True if `enable()` is currently in effect."""
    return bool(_STATE["enabled"])
