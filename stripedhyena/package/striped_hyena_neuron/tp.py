"""Tensor parallelism for StripedHyena 1 decode/prefill on PyTorch Native.

Adapted from the Evo 2 decode package ``evo2_neuron_decode/tp.py`` (commit 7c93113; see PROVENANCE.md): Megatron
column/row split, one ``all_reduce`` per Row-parallel linear, two per block (mixer output, MLP), and
row-parallel bias stored as bias/world on every rank (rank-divergent params made the backend
schedule mismatched graphs). SH1 differences:

  module                              split
  blocks.*.projections (Hyena)        Colwise BY HEAD: the column split groups rows per head as
                                      [x2_h | x1_h | v_h] (3 x 128), so contiguous row shards of whole
                                      heads keep each rank's split local (32 heads / world)
  blocks.*.filter.{short_filter_*}    per row, same slice as projections
  blocks.*.filter.{poles, residues, D} per channel (heads h -> channels h*128 .. h*128+127)
  blocks.*.out_filter_dense           Rowwise + all_reduce
  blocks.*.inner_mha_cls.Wqkv         Colwise: Q heads [r*H/w ...) and the matching KV heads
                                      [r*Hkv/w ...) (GQA groups stay on one rank; Hkv=8 -> world <= 8)
  blocks.*.inner_mha_cls.out_proj     Rowwise + all_reduce
  blocks.*.mlp.l1, l2 / l3            Colwise / Rowwise + all_reduce
  embedding, unembed, norms           replicated

Call ``shard_model`` on the loaded model BEFORE ``port.opts.apply`` / Decoder / Prefiller construction
(the tables, planes and fused weights are built from the sharded parameters).
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_NO_COLLECTIVE = os.environ.get("SHN_TP_DIAG_NO_ALLREDUCE", "0") == "1"   # DIAGNOSTIC ONLY (wrong outputs)


class RowParallelLinear(nn.Module):
    def __init__(self, weight, bias, world):
        super().__init__()
        self.weight = nn.Parameter(weight.contiguous(), requires_grad=False)
        self.bias = None if bias is None else nn.Parameter((bias / world).contiguous(), requires_grad=False)
        self.in_features, self.out_features = weight.shape[1], weight.shape[0]

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        if not _NO_COLLECTIVE:
            torch.distributed.all_reduce(y)
        return y


def _p(t):
    return nn.Parameter(t.detach().contiguous().clone(), requires_grad=False)


def _col(lin, idx):
    lin.weight = _p(lin.weight[idx])
    if lin.bias is not None:
        lin.bias = _p(lin.bias[idx])
    lin.out_features = lin.weight.shape[0]


def _row(lin, sl, world):
    return RowParallelLinear(lin.weight.detach()[:, sl], None if lin.bias is None else lin.bias.detach(), world)


def _shard_mlp(mlp, r, w):
    I_ = mlp.l1.out_features
    assert I_ % w == 0
    c = I_ // w
    sl = slice(r * c, (r + 1) * c)
    _col(mlp.l1, sl)
    _col(mlp.l2, sl)
    mlp.l3 = _row(mlp.l3, sl, w)


def _shard_hyena(block, r, w):
    f = block.filter
    H, dh = f.num_attention_heads, f.hidden_size_per_attention_head
    assert H % w == 0
    Hr = H // w
    rows = slice(r * Hr * 3 * dh, (r + 1) * Hr * 3 * dh)          # whole heads of [x2|x1|v]
    chans = slice(r * Hr * dh, (r + 1) * Hr * dh)
    _col(block.projections, rows)
    f.short_filter_weight = _p(f.short_filter_weight[rows])
    f.short_filter_bias = _p(f.short_filter_bias[rows])
    f.poles = _p(f.poles[chans])
    f.residues = _p(f.residues[chans])
    f.D = _p(f.D[chans])
    f.num_attention_heads = Hr
    f.hidden_size = Hr * dh
    f._modal_tab = None
    f._planes = None
    block.out_filter_dense = _row(block.out_filter_dense, chans, w)


def _shard_attn(block, r, w):
    a = block.inner_mha_cls
    H, Hkv, d = a.num_heads, a.num_heads_kv, a.head_dim
    assert H % w == 0 and Hkv % w == 0, (H, Hkv, w)
    Hr, Kr = H // w, Hkv // w
    q = torch.arange(r * Hr * d, (r + 1) * Hr * d)
    k = H * d + torch.arange(r * Kr * d, (r + 1) * Kr * d)
    v = H * d + Hkv * d + torch.arange(r * Kr * d, (r + 1) * Kr * d)
    _col(a.Wqkv, torch.cat([q, k, v]))
    a.out_proj = _row(a.out_proj, slice(r * Hr * d, (r + 1) * Hr * d), w)
    a.num_heads, a.num_heads_kv = Hr, Kr


def shard_model(model, rank, world):
    n = {"hyena": 0, "attn": 0}
    for b in model.blocks:
        if hasattr(b, "filter"):
            _shard_hyena(b, rank, world)
            n["hyena"] += 1
        else:
            _shard_attn(b, rank, world)
            n["attn"] += 1
        _shard_mlp(b.mlp, rank, world)
    model._tp = (rank, world)
    return n
