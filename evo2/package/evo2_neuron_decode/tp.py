"""Tensor parallelism for Evo 2 (StripedHyena-2) on PyTorch Native, Trainium.

``shard_model(model, rank, world)`` splits every block of a loaded vortex ``StripedHyena`` across
``world`` ranks, Megatron style, IN PLACE:

  ============================== ============ ==========================================
  module                          split         notes
  ============================== ============ ==========================================
  blocks.*.projections            Colwise       contiguous rows; correct ONLY because
                                                ``interleave: True`` (stride-3 channel order)
  blocks.*.filter.{short_filter_*, h, log_poles, residues, D}
                                  per channel   sliced by hand to match the projection split
  blocks.*.out_filter_dense       Rowwise       + all_reduce
  blocks.*.inner_mha_cls.Wqkv     Colwise       by head, (three h d) layout
  blocks.*.inner_mha_cls.out_proj Rowwise       + all_reduce; num_heads set to H/world
  blocks.*.mlp.l1, l2             Colwise
  blocks.*.mlp.l3                 Rowwise       + all_reduce
  embedding / unembed / norm      replicated    0.065% of the weights
  ============================== ============ ==========================================

Two all-reduces per block, 64 per decode step for the 32-block model.

ORDERING: call this on the CPU model BEFORE ``to_device()``. ``to_device()`` builds the fused
MLP weight ``_w12`` from ``l1``/``l2`` (load-time de-dup); sharding afterwards would leave a
full-width ``_w12``. Then ``to_device()`` and ``compile_blocks()`` as usual, per rank.

The slicing here is the same as ``tp/shard_math_cpu.py`` / ``tp/shard_attn_cpu.py``, which
proved it exact against the unsharded blocks (rel <= 1.7e-5, fp32).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# DIAGNOSTIC ONLY: skip the all-reduce to measure what the collectives cost.
# Produces WRONG outputs. Never set outside a timing experiment.
_NO_COLLECTIVE = os.environ.get("EVO2_TP_DIAG_NO_ALLREDUCE", "0") == "1"


class RowParallelLinear(nn.Module):
    """``in_features`` sharded. Each rank computes a partial product; one all_reduce sums them.

    The bias is stored as ``bias / world`` on EVERY rank (not on rank 0 only), so the ranks stay
    structurally identical. qwen_edit_lora found that rank-divergent parameters make the
    Neuron backend schedule mismatched graphs around the collective.
    """

    def __init__(self, weight, bias, world):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias / world, requires_grad=False)
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.wt = None   # set by api._pretranspose (EVO2_WT=1): (in, out) layout, weight released

    def forward(self, x):
        if self.wt is not None:
            y = x @ self.wt
            if self.bias is not None:
                y = y + self.bias
        else:
            y = F.linear(x, self.weight, self.bias)
        if not _NO_COLLECTIVE:
            torch.distributed.all_reduce(y)
        return y


def _p(t):
    return nn.Parameter(t.contiguous().clone(), requires_grad=False)


def _row(lin, sl, world):
    return RowParallelLinear(lin.weight.detach()[:, sl], None if lin.bias is None
                             else lin.bias.detach(), world)


def _col(lin, sl):
    lin.weight = _p(lin.weight.detach()[sl])
    if lin.bias is not None:
        lin.bias = _p(lin.bias.detach()[sl])
    lin.out_features = lin.weight.shape[0]


def _shard_mlp(mlp, r, w):
    I = mlp.l1.out_features
    if I % w:
        raise ValueError(f"inner_mlp_size {I} not divisible by world {w}")
    c = I // w
    sl = slice(r * c, (r + 1) * c)
    _col(mlp.l1, sl)
    _col(mlp.l2, sl)
    mlp.l3 = _row(mlp.l3, sl, w)


def _shard_cascade(block, r, w):
    f = block.filter
    D = f.hidden_size
    if D % w:
        raise ValueError(f"hidden_size {D} not divisible by world {w}")
    C = D // w
    groups = f.hyena_filter_groups
    ch_per_group = D // groups
    if groups % w or C % ch_per_group:
        raise ValueError(f"{groups} filter groups ({ch_per_group} ch each) do not split evenly "
                         f"over world {w}")
    sl3 = slice(3 * r * C, 3 * (r + 1) * C)        # contiguous because interleave=True
    slc = slice(r * C, (r + 1) * C)
    gsl = slice(r * groups // w, (r + 1) * groups // w)
    _col(block.projections, sl3)
    f.short_filter_weight = _p(f.short_filter_weight.detach()[sl3])
    if getattr(f, "short_filter_bias", None) is not None:
        f.short_filter_bias = _p(f.short_filter_bias.detach()[sl3])
    for name in ("h", "log_poles", "residues"):
        t = getattr(f, name, None)
        if isinstance(t, torch.Tensor) and t.shape[0] == groups:
            setattr(f, name, _p(t.detach()[gsl]))
    Dt = getattr(f, "D", None)
    if isinstance(Dt, torch.Tensor) and Dt.shape[0] == D:
        f.D = _p(Dt.detach()[slc])
    f.hidden_size = C
    f.hyena_filter_groups = groups // w
    f.num_systems = groups // w
    f.channels_per_group = C // (groups // w)
    block.out_filter_dense = _row(block.out_filter_dense, slc, w)


def _shard_attention(block, r, w):
    a = block.inner_mha_cls
    H = a.num_heads
    if H % w:
        raise ValueError(f"{H} heads not divisible by world {w}")
    Hr = H // w
    Dh = a.head_dim
    D = a.Wqkv.weight.shape[1]
    W3 = a.Wqkv.weight.detach().view(3, H, Dh, D)          # (three h d) -- vortex attention.py:653
    a.Wqkv.weight = _p(W3[:, r * Hr:(r + 1) * Hr].reshape(3 * Hr * Dh, D))
    if a.Wqkv.bias is not None:
        a.Wqkv.bias = _p(a.Wqkv.bias.detach().view(3, H, Dh)[:, r * Hr:(r + 1) * Hr].reshape(-1))
    a.Wqkv.out_features = a.Wqkv.weight.shape[0]
    a.out_proj = _row(a.out_proj, slice(r * Hr * Dh, (r + 1) * Hr * Dh), w)
    a.num_heads = Hr
    for attr in ("num_heads_kv", "num_heads_per_rank"):
        if hasattr(a, attr):
            setattr(a, attr, Hr)


def shard_model(model, rank: int, world: int):
    """Shard every block of a vortex StripedHyena in place. Returns a per-type count."""
    cfg = model.config
    attn = set(cfg.attn_layer_idxs)
    n = {"cascade": 0, "attention": 0, "mlp": 0}
    for i, block in enumerate(model.blocks):
        if i in attn:
            _shard_attention(block, rank, world); n["attention"] += 1
        else:
            _shard_cascade(block, rank, world); n["cascade"] += 1
        _shard_mlp(block.mlp, rank, world); n["mlp"] += 1
    model._tp = (rank, world)
    return n
