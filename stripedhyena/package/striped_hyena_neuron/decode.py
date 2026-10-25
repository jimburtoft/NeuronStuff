"""Functional decode step for StripedHyena 1 on Neuron.

The upstream path threads ``inference_params`` dataclasses + dicts through every block, and keys
work on python ints (``seqlen_offset``). Here the decode state is explicit tensors, and each block's
step is a method of a small ``nn.Module`` wrapper whose ``forward`` takes only tensors, so
``torch.compile(backend="neuron", dynamic=False)`` sees a fixed graph per block with no value guards
(a python-int offset recompiles every step; the position lives in the graph).

State (``DecodeState``), B = batch, W = KV window (slots), positions are a 1-element long tensor:
  hyena layer i:  fir[i] (2, B, 3D) bf16 PLANAR (slot 0 = input at t-2, slot 1 = t-1),
                  s_re[i], s_im[i] (B, D, 2) fp32
  (measured: the upstream interleaved (B, 3D, 2) FIR state, updated by cat/stack on the last axis,
  made the compiled Hyena step 27.7 ms instead of 1.3 ms at B=8 -- planar state fixes it.)
  attn layer j:   k[j], v[j] (B, Hkv, W, d) bf16, zero beyond the fill
Math is identical to ``port/hyena.sequential_forward`` and ``port/attention.NeuronMHA`` decode.

Modes (``Decoder(mode=...)``):
  eager     -- every op dispatched eagerly
  blocks:K  -- the first K of the 32 block wrappers compiled, each its own graph (+ head if K == 32)
  step      -- the whole step (embed, 32 blocks, norm, LM head, argmax) in ONE compiled graph
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# DIAGNOSTIC ONLY (measured ablation bounds): skip a component of the decode step to measure what it
# costs inside the real compiled step. Produces WRONG outputs. "cascade" = Hyena FIR + IIR + gate
# (y = x2), "attn_core" = RoPE + KV write + SDPA (ctx = q), "mlp" = both MLPs.
DIAG_SKIP = set(filter(None, os.environ.get("SHN_DIAG_SKIP", "").split(",")))

from . import modal
from .rmsnorm import rmsnorm_flash_formula
from . import nki_mlp as _nkimlp


class DecodeState:
    def __init__(self, fir, s_re, s_im, k, v, pos, W):
        self.fir, self.s_re, self.s_im, self.k, self.v = fir, s_re, s_im, k, v
        self.pos = pos          # (1,) long device tensor: the slot written by the NEXT step
        self.W = W

    @classmethod
    def from_inference_params(cls, model, ip, B, W, offset, repeat=1):
        """Convert the upstream ``inference_params`` (after a prefill through port.attention /
        port.hyena) into explicit tensors. ``repeat`` tiles batch row(s) (bench: one real prefill
        replicated to B rows -- real KV contents, real fill)."""
        fir, s_re, s_im, k, v = [], [], [], [], []
        hy, mh = ip["hyena"], ip["mha"]
        rep = (lambda t: t.repeat_interleave(repeat, 0).contiguous()) if repeat > 1 else (lambda t: t)
        for i, b in enumerate(model.blocks):
            # pop as we go: the source (B=1) copies are released layer by layer (peak HBM)
            if hasattr(b, "filter"):
                fir.append(rep(hy.fir_state_dict.pop(i)).permute(2, 0, 1).contiguous())
                sr, si = hy.state_dict.pop(i)
                s_re.append(rep(sr)); s_im.append(rep(si))
            else:
                c = mh.key_value_memory_dict.pop(i)                 # (2, b, Hkv, Wc, d)
                assert c.shape[3] == W, (c.shape, W)
                k.append(rep(c[0]).contiguous()); v.append(rep(c[1]).contiguous())
                del c
        pos = torch.tensor([offset], dtype=torch.long, device=fir[0].device)
        return cls(fir, s_re, s_im, k, v, pos, W)


_CASC_OP = None


def _cascade_op():
    """kernels/sh1_cascade.py as an nki_op (one registration per process)."""
    global _CASC_OP
    if _CASC_OP is None:
        from torch_neuronx import wrap_nki
        from torch_neuronx.nki_hop import nki_op
        from .kernels.sh1_cascade import sh1_cascade
        K = wrap_nki(sh1_cascade)

        @nki_op("shn::sh1_cascade", mutates_args={})
        def op(z: torch.Tensor, fir: torch.Tensor, s_re: torch.Tensor, s_im: torch.Tensor, wb: torch.Tensor,
               pl: torch.Tensor, dsk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            return K(z, fir, s_re, s_im, wb, pl, dsk)
        _CASC_OP = op
    return _CASC_OP



def _mlp_n(b, u, eps):
    """b.mlp(post_norm(u)); with the nki_mlp lever the norm can be fused into the nkilib TKG kernel."""
    if getattr(b.mlp, "_nki", False):
        return _nkimlp.norm_mlp(b.mlp, u, rmsnorm_flash_formula, b.post_norm.scale, eps)
    return b.mlp(rmsnorm_flash_formula(u, b.post_norm.scale, eps))

class HyenaStep(nn.Module):
    """One ParallelGatedConvBlock decode step (pre_norm, projections, FIR k=3, column split, real-pair
    2-state IIR, out_filter_dense, residual, post_norm, MLP, residual)."""

    def __init__(self, blk):
        super().__init__()
        self.blk = blk
        f = blk.filter
        p_re, p_im, r_re, r_im = modal.decode_planes(f.poles, f.residues, f.poles.device)
        self.register_buffer("p_re", p_re, persistent=False)
        self.register_buffer("p_im", p_im, persistent=False)
        self.register_buffer("r_re", r_re, persistent=False)
        self.register_buffer("r_im", r_im, persistent=False)
        self.H = f.num_attention_heads
        self.dh = f.hidden_size_per_attention_head
        self.eps = blk.pre_norm.eps
        self.casc = {}          # B -> (wb, pl, dsk) kernel weights (opts "cascade")

    def cascade_weights(self, B):
        if B not in self.casc:
            f = self.blk.filter
            D = f.hidden_size
            P, G = 128, D // 128
            w = f.short_filter_weight[:, 0].float()                                   # (3D, 3)
            wb = torch.stack([w[:, 0], w[:, 1], w[:, 2], f.short_filter_bias.float()])  # (4, 3D)
            wb = wb.reshape(4, 3, P, G).permute(2, 0, 1, 3)[:, :, None].expand(P, 4, B, 3, G).contiguous()
            K = self.p_re.shape[-1]
            pl = torch.stack([self.p_re[0], self.p_im[0], self.r_re[0], self.r_im[0]])   # (4, D, K)
            pl = pl.reshape(4, P, G, K).permute(1, 0, 2, 3)[:, :, None].expand(P, 4, B, G, K).contiguous()
            dsk = f.D.float().reshape(P, 1, G).expand(P, B, G).contiguous()
            self.casc[B] = (wb, pl, dsk)
            _cascade_op()       # register before any compiled call (graph-break otherwise)
        return self.casc[B]

    def forward(self, u, fir, s_re, s_im):
        # u: (B, D) residual stream
        b, f = self.blk, self.blk.filter
        z = b.projections(rmsnorm_flash_formula(u, b.pre_norm.scale, self.eps))      # (B, 3D)
        if self.casc:
            Bz = z.shape[0]
            D = f.hidden_size
            P, G = 128, D // 128
            K = s_re.shape[-1]
            wb, pl, dsk = self.casc[Bz]
            y, fir_n, s_re_n, s_im_n = _cascade_op()(
                z.reshape(Bz, 3, P, G), fir.reshape(2, Bz, 3, P, G), s_re.reshape(Bz, P, G, K),
                s_im.reshape(Bz, P, G, K), wb, pl, dsk)
            u = b.out_filter_dense(y.reshape(Bz, D).to(u.dtype)) + u
            if "mlp" not in DIAG_SKIP:
                u = _mlp_n(b, u, self.eps) + u
            return u, fir_n.reshape(2, Bz, 3 * D), s_re_n.reshape(Bz, D, K), s_im_n.reshape(Bz, D, K)
        if "cascade" in DIAG_SKIP:
            u = b.out_filter_dense(z[:, :u.shape[1]]) + u
            if "mlp" not in DIAG_SKIP:
                u = _mlp_n(b, u, self.eps) + u
            return u, fir, s_re, s_im
        w = f.short_filter_weight
        z_f = w[:, 0, 2][None] * z + fir[0] * w[:, 0, 0][None] + fir[1] * w[:, 0, 1][None] + f.short_filter_bias
        fir_n = torch.stack([fir[1], z], 0)
        from .hyena import split3
        x2, x1, v = split3(z_f, self.H, self.dh, 1)
        x1v = x1 * v
        res, s_re_n, s_im_n = modal.step_iir_real(x1v, s_re, s_im, self.p_re, self.p_im, self.r_re, self.r_im)
        y = (x2 * (res + f.D * x1v)).to(u.dtype)
        u = b.out_filter_dense(y) + u
        if "mlp" not in DIAG_SKIP:
            u = _mlp_n(b, u, self.eps) + u
        return u, fir_n, s_re_n, s_im_n


def _rope(x, cos, sin):
    c = torch.cat([cos, cos], -1)
    s = torch.cat([sin, sin], -1)
    d2 = x.shape[-1] // 2
    return x * c + torch.cat([-x[..., d2:], x[..., :d2]], -1) * s


class AttnStep(nn.Module):
    """One AttentionBlock decode step over a fixed W-slot KV window (masked beyond pos)."""

    def __init__(self, blk):
        super().__init__()
        self.blk = blk
        m = blk.inner_mha_cls
        self.H, self.Hkv, self.d = m.num_heads, m.num_heads_kv, m.head_dim
        dev = b_device(blk)
        self.register_buffer("inv_freq", m.rotary_emb._compute_inv_freq(device=dev), persistent=False)
        # attention_tkg inputs: inv_freqs (d/2, 1) fp32; masks set by Decoder
        self.register_buffer("inv_col", self.inv_freq.reshape(-1, 1).contiguous(), persistent=False)
        self.masks = None
        self.curr = None
        self.eps = blk.pre_norm.eps
        self.scale = m.softmax_scale if m.softmax_scale is not None else self.d ** -0.5

    def forward(self, u, kc, vc, pos):
        b, m = self.blk, self.blk.inner_mha_cls
        B = u.shape[0]
        H, Hkv, d = self.H, self.Hkv, self.d
        qkv = m.Wqkv(rmsnorm_flash_formula(u, b.pre_norm.scale, self.eps))           # (B, (H+2Hkv) d)
        if isinstance(kc, (list, tuple)):
            from . import tkg
            ctx = tkg.tkg_attention(qkv[:, : H * d], qkv[:, H * d:(H + Hkv) * d], qkv[:, (H + Hkv) * d:],
                                    kc, vc, self.masks, self.inv_col, pos, H, Hkv, d, self.curr)
            u = m.out_proj(ctx) + u
            return _mlp_n(b, u, self.eps) + u
        q = qkv[:, : H * d].reshape(B, H, 1, d)
        if "attn_core" in DIAG_SKIP:
            u = m.out_proj(q.reshape(B, H * d)) + u
            if "mlp" not in DIAG_SKIP:
                u = _mlp_n(b, u, self.eps) + u
            return u
        kv = qkv[:, H * d:].reshape(B, 2, Hkv, 1, d)
        k, v = kv[:, 0], kv[:, 1]
        freqs = pos.to(torch.float32)[:, None] * self.inv_freq[None, :]
        cos, sin = torch.cos(freqs).to(q.dtype), torch.sin(freqs).to(q.dtype)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        kc.index_copy_(2, pos, k)
        vc.index_copy_(2, pos, v)
        W = kc.shape[2] if self.curr is None else self.curr
        kr, vr = (kc, vc) if W == kc.shape[2] else (kc[:, :, :W], vc[:, :, :W])
        mask = torch.where(torch.arange(W, device=u.device) <= pos, 0.0, float("-inf")).to(q.dtype)
        ctx = F.scaled_dot_product_attention(q, kr, vr, attn_mask=mask[None, None, None, :],
                                             scale=self.scale, enable_gqa=True)
        u = m.out_proj(ctx.reshape(B, H * d)) + u
        if "mlp" not in DIAG_SKIP:
            u = _mlp_n(b, u, self.eps) + u
        return u


def free_prefill_tables(model):
    """Drop the per-layer modal prefill tables (~67 MB/layer, 1.07 GB total at chunk 64); the
    decode step uses its own pole/residue planes. Rebuilt lazily by the next prefill."""
    import gc
    n = 0
    for b in model.blocks:
        if hasattr(b, "filter") and getattr(b.filter, "_modal_tab", None) is not None:
            b.filter._modal_tab = None
            n += 1
    gc.collect()
    empty_cache()
    return n


def empty_cache():
    try:
        import torch_neuronx
        torch_neuronx.empty_cache()
    except Exception:  # noqa: BLE001  (CPU tests)
        pass


def b_device(blk):
    return next(blk.parameters()).device


def to_tkg_layout(dec, S):
    """Convert DecodeState KV (B, Hkv, W, d) into attention_tkg per-chunk layout, in place."""
    from . import tkg
    a0 = next(st for st, h in zip(dec.steps, dec.is_hy) if not h)
    B = S.k[0].shape[0]
    G = a0.H // a0.Hkv
    sizes = tkg.chunk_sizes(B, a0.Hkv, G)
    dt = S.k[0].dtype
    masks = [torch.ones(1, c, G, 1, dtype=dt, device=S.k[0].device) for c in sizes]
    for st, h in zip(dec.steps, dec.is_hy):
        if not h:
            st.masks = masks
    for j in range(len(S.k)):
        S.k[j], S.v[j] = tkg.to_kernel_layout(S.k[j], S.v[j], sizes)
    # register the nki_ops BEFORE any compiled call: registering inside the compiled region
    # graph-breaks on torch.library.infer_schema (2 breaks per attention block, measured)
    for c in set(sizes):
        tkg.get_op(c, G, a0.d, S.W, S.W)
    return sizes


def enable_cascade(dec, B):
    """opts "cascade": build the per-B kernel weights on every Hyena step module."""
    n = 0
    for st, h in zip(dec.steps, dec.is_hy):
        if h:
            st.cascade_weights(B)
            n += 1
    return n


def set_bucket(dec, S, max_fill, bmin=512):
    """attention_tkg reads only curr_sprior slots: next power of two >= max_fill (min bmin), <= W.
    ``max_fill`` is a host int (the highest position the next steps will reach + 1)."""
    fill = max_fill
    b = bmin
    while b < fill:
        b *= 2
    curr = min(b, S.W)
    for st, h in zip(dec.steps, dec.is_hy):
        if not h:
            st.curr = curr
            if st.masks is not None:
                from . import tkg
                for m in st.masks:
                    tkg.get_op(m.shape[1], st.H // st.Hkv, st.d, curr, S.W)
    return curr


class Head(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.emb = model.embedding_layer
        self.norm = model.norm
        self.unembed = model.unembed

    def embed(self, ids):
        return self.emb.embed(ids)

    def forward(self, u):
        x = rmsnorm_flash_formula(u, self.norm.scale, self.norm.eps)
        logits = self.unembed.unembed(x)
        # greedy token in the same graph: eager aten::argmax.out falls back to the host on torch-neuronx 2.13
        return logits, logits.argmax(-1)


class Decoder(nn.Module):
    def __init__(self, model, mode="eager"):
        super().__init__()
        self.steps = nn.ModuleList(HyenaStep(b) if hasattr(b, "filter") else AttnStep(b) for b in model.blocks)
        self.is_hy = [hasattr(b, "filter") for b in model.blocks]
        self.head = Head(model)
        self.mode = mode
        self._fns = [s for s in self.steps]
        self._head = self.head
        self._step_fn = None
        if mode.startswith("blocks:"):
            K = int(mode.split(":")[1])
            _raise_limits()
            for i in range(K):
                self.steps[i].compile(backend="neuron", dynamic=False)
            if K == len(self.steps):
                self.head.compile(backend="neuron", dynamic=False)
        elif mode == "step":
            _raise_limits()
            self._step_fn = torch.compile(self._step_body, backend="neuron", dynamic=False)

    def _step_body(self, ids, fir, s_re, s_im, k, v, pos):
        u = self.head.embed(ids)
        fir, s_re, s_im = list(fir), list(s_re), list(s_im)
        hi = ai = 0
        for i, st in enumerate(self.steps):
            if self.is_hy[i]:
                u, fir[hi], s_re[hi], s_im[hi] = st(u, fir[hi], s_re[hi], s_im[hi])
                hi += 1
            else:
                u = st(u, k[ai], v[ai], pos)
                ai += 1
        logits, tok = self.head(u)
        return logits, tok, fir, s_re, s_im

    def forward(self, ids, S: DecodeState):
        """ids: (B,) long device tensor. Advances S in place. Returns (logits (B, V), greedy tokens (B,))."""
        fn = self._step_fn if self._step_fn is not None else self._step_body
        logits, tok, S.fir, S.s_re, S.s_im = fn(ids, S.fir, S.s_re, S.s_im, S.k, S.v, S.pos)
        S.pos.add_(1)
        return logits, tok


def _raise_limits():
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 65536)
    torch._dynamo.config.accumulated_cache_size_limit = max(
        torch._dynamo.config.accumulated_cache_size_limit, 1048576)
