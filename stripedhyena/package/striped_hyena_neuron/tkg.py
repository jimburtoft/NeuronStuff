"""Decode lever: nkilib ``attention_tkg`` for StripedHyena 1 GQA decode (32 Q / 8 KV heads, d=128).

Adapted from the Evo 2 decode package ``evo2_neuron_decode/patches/tkg_mha2.py`` (commit 7c93113; see PROVENANCE.md),
keeping its measured design decisions, restated for GQA:
- GQA FOLD (measured probe): one kernel row per (sequence, KV head): ``bs = B*Hkv``, ``q_head = H/Hkv
  = 4``, ``s_active = 1``. NCC_INKI016 ``bs*q_head*s_active <= 128`` -> at most 32 rows (B <= 4)
  per call; larger batches are tiled into independent row chunks (measured BH tiling).
- KV-NATIVE, PRE-TRANSPOSED K: the KV cache lives in the kernel layout,
  one contiguous tensor PER CHUNK (measured: a sliced view passed into the compiled op gave
  wrong output): K ``(c, 1, d, W)`` with ``tp_k_prior=False``, V ``(c, 1, W, d)``.
- nki_op INSIDE torch.compile: the kernel is registered as an ``nki_op`` (cfg and
  SbufManager closed over), so the whole attention block compiles to one graph with 0 breaks.
- POSITIONS IN GRAPH: ``rope_pos_ids`` are built from the position tensor inside
  the graph; no eager per-step fill.
- ``fuse_rope=True`` + ``use_pos_id=True``: the kernel applies rotate-half RoPE to q and k_active
  (base from ``inv_freqs``, fp32), scales q by 1/sqrt(d), builds the causal mask from positions,
  and returns ``k_out`` = RoPE'd k_active, which we write into the K cache (each key rotated once).
- Unwritten slots are ZERO (measured: garbage in the masked tail poisons the softmax).
- FILL BUCKETING is optional here: ``curr_sprior`` = a bucket >= fill reads only the
  filled prefix; one compiled graph per bucket.
Constraint: W divisible by 256 (measured: the kernel shards s_prior by 2, shard % 128 == 0).
"""
import torch

SBUF_BYTES = 24 * 1024 * 192
_OPS = {}
_KERN = None
_SBM = None


def _kern():
    global _KERN, _SBM
    if _KERN is None:
        import os
        from nkilib.core.attention.attention_tkg import attention_tkg
        from nkilib.core.utils.allocator import SbufManager
        from torch_neuronx import wrap_nki
        lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
        _KERN = wrap_nki(attention_tkg)[(lnc,)]
        _SBM = SbufManager(0, SBUF_BYTES)
        _SBM.open_scope()
    return _KERN, _SBM


def get_op(c, q_head, d, curr, full):
    """One registered nki_op per (rows c, curr_sprior, full_sprior)."""
    key = (c, q_head, d, curr, full)
    if key in _OPS:
        return _OPS[key]
    from nkilib.core.attention.attention_tkg_utils import AttnTKGConfig
    from torch_neuronx.nki_hop import nki_op
    KERN, SBM = _kern()
    cfg = AttnTKGConfig(bs=c, q_head=q_head, s_active=1, curr_sprior=curr, full_sprior=full, d_head=d,
                        block_len=0, tp_k_prior=False, use_pos_id=True, fuse_rope=True)

    @nki_op("shn::attn_tkg_c%d_q%d_s%d_f%d" % (c, q_head, curr, full), mutates_args={})
    def op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kp: torch.Tensor, vp: torch.Tensor,
           m: torch.Tensor, inv: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        o = torch.empty(q.shape[0], q.shape[1], d, 1, dtype=q.dtype, device=q.device)
        ko = torch.empty(q.shape[0], 1, d, 1, dtype=q.dtype, device=q.device)
        r = KERN(q, k, v, kp, vp, m, o, cfg, SBM, inv, pos, None, None, None, ko)
        r = r[0] if isinstance(r, (tuple, list)) else r
        return r, ko
    op._cfg = cfg
    _OPS[key] = op
    return op


def chunk_sizes(B, Hkv, q_head):
    rows = B * Hkv
    maxc = (128 // q_head) // Hkv * Hkv          # whole sequences per chunk (32 rows = 4 seqs)
    out = []
    while rows > 0:
        c = min(maxc, rows)
        out.append(c)
        rows -= c
    return out


def to_kernel_layout(k, v, sizes):
    """k, v: (B, Hkv, W, d) -> per-chunk lists: K (c, 1, d, W), V (c, 1, W, d), contiguous."""
    B, Hkv, W, d = k.shape
    kf = k.reshape(B * Hkv, 1, W, d)
    vf = v.reshape(B * Hkv, 1, W, d)
    ks, vs, r0 = [], [], 0
    for c in sizes:
        ks.append(kf[r0:r0 + c].transpose(2, 3).contiguous())
        vs.append(vf[r0:r0 + c].contiguous())
        r0 += c
    return ks, vs


def tkg_attention(q, k, v, kTs, vTs, masks, inv, pos, H, Hkv, d, curr=None):
    """Decode attention for one layer. q (B, H, d), k/v (B, Hkv, d) un-rotated; pos (1,) long.
    kTs/vTs: per-chunk kernel-layout caches (written in place at pos). Returns ctx (B, H*d)."""
    B = q.shape[0]
    G = H // Hkv
    W = vTs[0].shape[2]
    curr = W if curr is None else curr
    qf = q.reshape(B * Hkv, G, 1, d)
    kf = k.reshape(B * Hkv, 1, 1, d)
    vf = v.reshape(B * Hkv, 1, 1, d)
    pf = pos.to(torch.float32).view(1, 1)
    outs, r0 = [], 0
    for j, (kT, vT) in enumerate(zip(kTs, vTs)):
        c = kT.shape[0]
        op = get_op(c, G, d, curr, W)
        o, ko = op(qf[r0:r0 + c], kf[r0:r0 + c], vf[r0:r0 + c], kT, vT, masks[j], inv,
                   pf.expand(c, 1).contiguous())
        kT.index_copy_(3, pos, ko)
        vT.index_copy_(2, pos, vf[r0:r0 + c])
        outs.append(o)
        r0 += c
    o = outs[0] if len(outs) == 1 else torch.cat(outs, 0)          # (B*Hkv, G, d, 1)
    return o.reshape(B, H * d)
