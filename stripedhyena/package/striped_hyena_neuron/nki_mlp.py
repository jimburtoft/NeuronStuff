"""Lever ``nki_mlp``: nkilib ``core.mlp.mlp`` (shipped with the Neuron SDK) for the SH1 SiLU-gated MLP.

  mlp(z) = l3(silu(l1 z) * l2 z),  l1/l2 (I, D), l3 (D, I), bf16, no bias; I = 14336 / TP world.

Weights are stored ONCE, transposed (in, out): gate = l1^T (D, I), up = l2^T (D, I), down = l3^T (I, D)
(the nkilib layout; the originals are freed). Per call site the row count R (decode: B; prefill: B*L) is a
static shape, so the path is chosen at trace time:

  R <= TKG_MAX        nkilib TKG (mode=DECODE; at I = 14336 DECODE fails validation from 128 rows:
                      "Not enough memory for Gate/Up projection weights"); at R <= FUSE_MAX decode call sites fuse the post-norm RMSNorm
                      into the kernel (NormType.RMS_NORM = x * rsqrt(mean(x^2) + eps) * w, the flash formula)
  R >= CTE_MIN        nkilib CTE, norm in torch (prefill keeps its fp32 residual -> bf16 norm output)
  otherwise           torch on the transposed weights (x @ Wt), norm in torch.rms_norm form

Thresholds: ``thresholds()`` (per-MLP microbench, then end-to-end decode A/B on trn2);
env-overridable for A/B (SHN_NKI_MLP_{FUSE_MAX,TKG_MAX,CTE_MIN}).
Under TP the down projection is row-parallel: one all_reduce after the kernel (as RowParallelLinear).

nki_op registration happens in ``apply`` (before any compiled call). ``kernels/nkilib_mlp_compat`` is
installed first: under the earlier NKI 0.7 frontend (torch-neuronx 2.13) nkilib failed to specialize mlp_cte_basic_transpose's list
comprehensions; on NKI 0.7.0 GA (SDK 2.33) the patch is a no-op.
"""
import os

import torch
import torch.nn.functional as F

LNC = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
# Fused post-norm (TKG NormType.RMS_NORM; one-core decode at <= 32 rows; never under TP, never with the cascade kernel):
# default ON since SDK 2.33. Measured on SDK 2.33 (3 processes each, sequential): 1 core B=8 @ 8K 43.98 vs 46.77 ms
# (1.063x), B=32 @ 2K 49.06 vs 49.69 ms (1.013x); with cascade at B=1 33.89 vs 33.26 ms (0.98x) -> skipped there.
# Passes every reference gate on SDK 2.33 (short vs ref top-1 0.984); on torch-neuronx 2.13 it missed one by one token (0.977).
FUSE_NORM = os.environ.get("SHN_NKI_MLP_FUSE_NORM", "1") == "1"
_ENV_FUSE = os.environ.get("SHN_NKI_MLP_FUSE_MAX")
_ENV_TKG = os.environ.get("SHN_NKI_MLP_TKG_MAX")
_ENV_CTE = os.environ.get("SHN_NKI_MLP_CTE_MIN")
_NO_COLLECTIVE = os.environ.get("SHN_TP_DIAG_NO_ALLREDUCE", "0") == "1"

_OPS = {}


def thresholds(I_):
    """(fuse_max, tkg_max, cte_min) rows. Microbench (per MLP, vs compiled torch): I=14336 fused TKG
    1.08-1.18x at R<=32, unfused TKG ~1.0x at 64, CTE 1.13-1.22x at 512-2048, x@Wt 0.83-0.86x at 32-64;
    I=3584 (TP=4 shard) fused/unfused TKG 1.14-1.6x up to 128 (mode=DECODE), x@Wt 1.2-1.6x at 128-512.
    End to end (decides): 1 core B=48 2K x@Wt at 48 rows beat unfused TKG (49.0-49.4 vs 51.8-52.1 ms) -> tkg_max 32.
    Under TP=4 the TKG kernel LOSES in the real step (B=1 step graph 18.1-18.3 vs 15.0-15.2 ms; B=8 24.7 vs 19.1)
    while x@Wt wins (13.2 / 17.2 ms) -> TP shard (I <= 8192): no TKG, x@Wt for decode, CTE for prefill."""
    full = I_ > 8192
    fuse = int(_ENV_FUSE) if _ENV_FUSE else (32 if full else 0)
    tkg = int(_ENV_TKG) if _ENV_TKG else (32 if full else 0)
    cte = int(_ENV_CTE) if _ENV_CTE else (512 if full else 2048)
    return fuse, tkg, cte


def tiling(I_):
    # measured: at I = 14336 the default gate/up column tiling fails kernel validation (NCC_INKI016 "Stack out
    # of memory", surfaced by Dynamo as "RuntimeError when making fake tensor call"); down tiling is fine
    return (False, True) if I_ > 8192 else (True, True)


def _register(norm, gu, dn, mode_name):
    key = (norm, gu, dn, mode_name)
    if key in _OPS:
        return _OPS[key]
    from torch_neuronx import wrap_nki
    from torch_neuronx.nki_hop import nki_op
    try:
        from .kernels.nkilib_mlp_compat import install
    except ImportError:
        from kernels.nkilib_mlp_compat import install
    install()
    from nkilib.core.mlp.mlp import mlp as _mlp
    from nkilib.core.utils.common_types import ActFnType, NormType
    try:
        from nkilib.core.mlp.mlp_parameters import ComputationMode
    except Exception:  # noqa: BLE001
        from nkilib.core.utils.common_types import ComputationMode
    K = wrap_nki(_mlp)[LNC]
    mode = getattr(ComputationMode, mode_name)
    name = "shn::mlp_%s_%d%d_%s" % ("rms" if norm else "plain", gu, dn, mode_name.lower())
    if norm:
        @nki_op(name, mutates_args={})
        def op(x: torch.Tensor, g: torch.Tensor, u: torch.Tensor, d: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            r = K(hidden_tensor=x, gate_proj_weights_tensor=g, up_proj_weights_tensor=u,
                  down_proj_weights_tensor=d, normalization_weights_tensor=w,
                  activation_fn=ActFnType.SiLU, normalization_type=NormType.RMS_NORM, eps=1e-5,
                  use_tkg_gate_up_proj_column_tiling=gu, use_tkg_down_proj_column_tiling=dn, mode=mode)
            return r[0] if isinstance(r, (list, tuple)) else r
    else:
        @nki_op(name, mutates_args={})
        def op(x: torch.Tensor, g: torch.Tensor, u: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
            r = K(hidden_tensor=x, gate_proj_weights_tensor=g, up_proj_weights_tensor=u,
                  down_proj_weights_tensor=d, activation_fn=ActFnType.SiLU,
                  use_tkg_gate_up_proj_column_tiling=gu, use_tkg_down_proj_column_tiling=dn, mode=mode)
            return r[0] if isinstance(r, (list, tuple)) else r
    _OPS[key] = op
    return op


def _reduce(y, tp):
    if tp and not _NO_COLLECTIVE:
        torch.distributed.all_reduce(y)
    return y


def _run(m, z2, w=None):
    """z2: (R, D) bf16 (normed unless w is given). Returns (R, D)."""
    R, D = z2.shape
    I_ = m._I
    fuse, tkg, cte = thresholds(I_)
    if w is not None and R > fuse:
        return None, False
    if R <= tkg:
        gu, dn = tiling(I_)
        if w is not None:
            y = _register(True, gu, dn, "DECODE")(z2.reshape(1, R, D), m._g, m._u, m._d, w)
        else:
            y = _register(False, gu, dn, "DECODE")(z2.reshape(1, R, D), m._g, m._u, m._d)
        return _reduce(y.reshape(R, D), m._tp), True
    if w is not None:
        return None, False
    if R >= cte:
        y = _register(False, True, True, "PREFILL")(z2.reshape(1, R, D), m._g, m._u, m._d)
        return _reduce(y.reshape(R, D), m._tp), True
    y = (F.silu(z2 @ m._g) * (z2 @ m._u)) @ m._d
    return _reduce(y, m._tp), True


def forward(self, z):
    """ParallelGatedMLP.forward replacement (z already normed by the caller; any leading dims)."""
    shp = z.shape
    y, _ = _run(self, z.reshape(-1, shp[-1]))
    return y.reshape(shp)


def _cascade_on():
    try:
        from . import opts as O
    except ImportError:
        from port import opts as O
    return "cascade" in O.ON


def norm_mlp(mlp, u, norm_fn, scale, eps):
    """mlp(norm_fn(u)) for decode call sites; fuses the norm into the TKG kernel when it applies."""
    if getattr(mlp, "_nki", False) and FUSE_NORM and not _cascade_on() and u.dim() == 2 and u.dtype == torch.bfloat16:
        y, ok = _run(mlp, u, mlp._normw)
        if ok:
            return y
    return mlp(norm_fn(u, scale, eps))


def apply(model):
    """Convert every ParallelGatedMLP in place (after TP sharding). Returns the number converted."""
    import gc
    from stripedhyena.layers import ParallelGatedMLP
    n = 0
    for b in model.blocks:
        m = b.mlp
        if not isinstance(m, ParallelGatedMLP):
            continue
        dev = m.l1.weight.device
        tp = hasattr(m.l3, "in_features") and type(m.l3).__name__ == "RowParallelLinear"
        g = m.l1.weight.detach().to("cpu").t().contiguous()
        u = m.l2.weight.detach().to("cpu").t().contiguous()
        d = m.l3.weight.detach().to("cpu").t().contiguous()
        m.l1 = m.l2 = m.l3 = None
        gc.collect()
        m._g, m._u, m._d = g.to(dev), u.to(dev), d.to(dev)
        m._I = g.shape[1]
        m._tp = tp
        m._normw = b.post_norm.scale.detach().reshape(1, -1).contiguous()   # checkpoint dtype, as microbenched
        m._nki = True
        m.forward = forward.__get__(m)
        n += 1
    I_ = model.blocks[0].mlp._I
    gu, dn = tiling(I_)
    for norm in (True, False):
        _register(norm, gu, dn, "DECODE")
    _register(False, True, True, "PREFILL")
    from .decode import empty_cache
    empty_cache()
    return n
