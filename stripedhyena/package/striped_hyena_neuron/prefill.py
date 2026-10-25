"""Compiled on-device prefill for StripedHyena 1.

Same math as the eager port (``port/hyena.parallel_forward`` + ``port/attention.NeuronMHA`` prefill),
restructured so each block is a tensors-only ``nn.Module`` compiled with
``torch.compile(backend="neuron", dynamic=False)`` -- one graph per (block type, B, L), shared by
all 16 blocks of a type. Produces the ``DecodeState`` for ``port/decode.Decoder`` directly
(planar FIR state, real-pair IIR state, fixed-window KV cache).

Hyena long conv: ``port/modal.modal_conv`` (chunked real matmuls + log-depth scan). Its tables are
per-layer buffers passed as inputs (so one graph serves all 16 layers).
Attention: causal GQA SDPA (``enable_gqa=True``), query-chunked for long L.

    pf = Prefiller(model, mode="blocks")       # or "eager"
    logits_last, S = pf(ids (B, L) long, W)    # W = KV window (>= L + decode steps)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import hyena as port_hyena
from . import modal
from .decode import DecodeState, _raise_limits
from .rmsnorm import rmsnorm_flash_formula

import os as _os
# Prefill residual stream in fp32 (default ON, measured): with a bf16 residual stream the compiled
# prefill scored WikiText window 0 at PPL 4.94 (eager 4.66, CUDA reference 4.66, fp32 4.66) -- deterministic,
# from position 2 onward (a "\n\n" token whose residual norm is ~7 while neighbours are ~0.3). Only
# block 0 compiled already gave 6.51; an fp32 block 0 gave 5.52; isotropic 0.1-0.3% noise in every
# block did not move it. With an fp32 residual stream (block math still bf16) all 10 windows match.
RESID_FP32 = _os.environ.get("SHN_RESID_FP32", "1") == "1"
QCHUNK_ELEMS = 1 << int(_os.environ.get("SHN_QCHUNK_LOG2", "27"))   # bound H * qc * L score elements per SDPA call (A/B knob)
# Continuation segments that read a KV bucket > 8192 slots use smaller query chunks. Measured on SDK 2.33, TP=4, seg 8192:
# 32K TTFT 9.55 s at 2^27 -> 7.57 s at 2^25 (8.15 s at 2^26); 16K 2.28 -> 2.09 s; 8K buckets keep 2^27 (2^25 was 2 % slower).
QCHUNK_ELEMS_LONG = 1 << int(_os.environ.get("SHN_QCHUNK_LONG_LOG2", "25"))
WB_MIN = 1024            # smallest KV bucket read by a continuation segment


def _norm_in(u, scale, eps, dt):
    """RMSNorm of the residual stream, output in the weight dtype (u may be fp32 if RESID_FP32)."""
    return rmsnorm_flash_formula(u, scale, eps).to(dt)


def modal_conv_t(x, H, PE, AT, q_re, q_im, s0_re=None, s0_im=None):
    """``modal.modal_conv`` with the tables as explicit tensors (graph-shareable across layers).
    Optional initial state s0 (B, D, K) (state just before x[..., 0]): then L must be a multiple of
    C (left zero-padding is only exact for a zero initial state)."""
    B, D, L = x.shape
    C = H.shape[-1]
    K = PE.shape[1] // 2
    N = -(-L // C)
    Lp = N * C
    pad = Lp - L
    has_s0 = s0_re is not None
    assert not (has_s0 and pad), "segment length must be a multiple of the modal chunk"
    if pad:
        # slice-assign into zeros: F.pad and zeros+cat both hit neuronx-cc NCC_ISFV902 (SFKVectorizer
        # "index 512 is out of bounds for axis 1 with size 512") on this fp32 graph at L % 64 != 0
        # (L=1982; measured). L % 64 == 0 never pads.
        xp = x.new_zeros(B, D, Lp)
        xp[..., pad:] = x
        x = xp
    X = x.reshape(B, D, N, C).permute(1, 3, 0, 2).reshape(D, C, B * N)
    y = torch.bmm(H, X)
    E = torch.bmm(PE, X).reshape(D, 2, K, B, N)
    S_re, S_im = E[:, 0], E[:, 1]
    if has_s0:
        z_re, z_im = s0_re.permute(1, 2, 0), s0_im.permute(1, 2, 0)          # (D, K, B)
        qr, qi = q_re[0].reshape(D, K, 1), q_im[0].reshape(D, K, 1)          # p^C (D, K, 1)
        S_re = torch.cat([(S_re[..., 0] + qr * z_re - qi * z_im)[..., None], S_re[..., 1:]], -1)
        S_im = torch.cat([(S_im[..., 0] + qr * z_im + qi * z_re)[..., None], S_im[..., 1:]], -1)
    s, i = 1, 0
    while s < N:
        qr, qi = q_re[i], q_im[i]
        a_re, a_im = S_re[..., :-s], S_im[..., :-s]
        n_re = S_re[..., s:] + qr * a_re - qi * a_im
        n_im = S_im[..., s:] + qr * a_im + qi * a_re
        S_re = torch.cat([S_re[..., :s], n_re], -1)
        S_im = torch.cat([S_im[..., :s], n_im], -1)
        s *= 2
        i += 1
    if N > 1 or has_s0:
        if has_s0:
            P = torch.cat([torch.cat([z_re[..., None], S_re[..., :-1]], -1),
                           torch.cat([z_im[..., None], S_im[..., :-1]], -1)], 1)
        else:
            P = torch.cat([F.pad(S_re[..., :-1], (1, 0)), F.pad(S_im[..., :-1], (1, 0))], 1)
        y = y + torch.bmm(AT, P.reshape(D, 2 * K, B * N))
    y = y.reshape(D, C, B, N).permute(2, 0, 3, 1).reshape(B, D, Lp)
    if pad:
        y = y[..., pad:]
    return y, S_re[..., N - 1].permute(2, 0, 1), S_im[..., N - 1].permute(2, 0, 1)


MODAL_IMPL = _os.environ.get("SHN_MODAL_IMPL", "bmm")   # "bmm" (per-channel Toeplitz) | "tri"
CAUSAL_SKIP = _os.environ.get("SHN_CAUSAL_SKIP", "1") == "1"   # prefill SDPA: query chunk i reads keys [0, end_i)


def modal_conv_tri(x, W_re, W_im, G_re, G_im, Pc1_re, Pc1_im, Utri, AT, q_re, q_im, s0_re=None, s0_im=None):
    """Same result as ``modal_conv_t``, semiseparable form: within a chunk,
        y_t = Re sum_k r_k p_k^t * sum_{j<=t} p_k^-j x_j ,   E = p^(C-1) * sum_j p^-j x_j,
    so the per-channel (C x C) Toeplitz bmm (4096 tiny matmuls) becomes elementwise work plus ONE
    shared lower-triangular (C x C) matmul. Stable: rounding is relative to sum|x| (|p^t p^-j| <= 1)."""
    B, D, L = x.shape
    C = Utri.shape[0]
    K = W_re.shape[1]
    N = -(-L // C)
    Lp = N * C
    pad = Lp - L
    has_s0 = s0_re is not None
    assert not (has_s0 and pad)
    if pad:
        xp = x.new_zeros(B, D, Lp)
        xp[..., pad:] = x
        x = xp
    X = x.reshape(B, D, 1, N, C)
    a_re = X * W_re[None, :, :, None, :]                                     # (B, D, K, N, C)
    a_im = X * W_im[None, :, :, None, :]
    c = torch.stack([a_re, a_im], 0).reshape(-1, C) @ Utri                   # prefix sums over j <= t
    c = c.reshape(2, B, D, K, N, C)
    c_re, c_im = c[0], c[1]
    g_re, g_im = G_re[None, :, :, None, :], G_im[None, :, :, None, :]
    y = (g_re * c_re - g_im * c_im).sum(2)                                   # (B, D, N, C)
    e_re = Pc1_re[None, :, :, :] * c_re[..., C - 1] - Pc1_im[None, :, :, :] * c_im[..., C - 1]   # (B, D, K, N)
    e_im = Pc1_re[None, :, :, :] * c_im[..., C - 1] + Pc1_im[None, :, :, :] * c_re[..., C - 1]
    S_re, S_im = e_re.permute(1, 2, 0, 3), e_im.permute(1, 2, 0, 3)          # (D, K, B, N)
    if has_s0:
        z_re, z_im = s0_re.permute(1, 2, 0), s0_im.permute(1, 2, 0)
        qr, qi = q_re[0].reshape(D, K, 1), q_im[0].reshape(D, K, 1)
        S_re = torch.cat([(S_re[..., 0] + qr * z_re - qi * z_im)[..., None], S_re[..., 1:]], -1)
        S_im = torch.cat([(S_im[..., 0] + qr * z_im + qi * z_re)[..., None], S_im[..., 1:]], -1)
    s, i = 1, 0
    while s < N:
        qr, qi = q_re[i], q_im[i]
        a_re2, a_im2 = S_re[..., :-s], S_im[..., :-s]
        n_re = S_re[..., s:] + qr * a_re2 - qi * a_im2
        n_im = S_im[..., s:] + qr * a_im2 + qi * a_re2
        S_re = torch.cat([S_re[..., :s], n_re], -1)
        S_im = torch.cat([S_im[..., :s], n_im], -1)
        s *= 2
        i += 1
    y = y.permute(1, 3, 0, 2).reshape(D, C, B * N)                            # (D, C, BN) like modal_conv_t
    if N > 1 or has_s0:
        if has_s0:
            P = torch.cat([torch.cat([z_re[..., None], S_re[..., :-1]], -1),
                           torch.cat([z_im[..., None], S_im[..., :-1]], -1)], 1)
        else:
            P = torch.cat([F.pad(S_re[..., :-1], (1, 0)), F.pad(S_im[..., :-1], (1, 0))], 1)
        y = y + torch.bmm(AT, P.reshape(D, 2 * K, B * N))
    y = y.reshape(D, C, B, N).permute(2, 0, 3, 1).reshape(B, D, Lp)
    if pad:
        y = y[..., pad:]
    return y, S_re[..., N - 1].permute(2, 0, 1), S_im[..., N - 1].permute(2, 0, 1)


class HyenaPrefill(nn.Module):
    def __init__(self, blk, tab):
        super().__init__()
        self.blk = blk
        for n in ("H", "PE", "AT", "q_re", "q_im", "W_re", "W_im", "G_re", "G_im", "Pc1_re", "Pc1_im", "Utri"):
            self.register_buffer(n, getattr(tab, n), persistent=False)
        self.tri_ok = tab.tri_ok
        f = blk.filter
        self.nh, self.dh = f.num_attention_heads, f.hidden_size_per_attention_head
        self.eps = blk.pre_norm.eps
        self.mixer = _HyMixer(self)
        self.tail = _HyTail(self)

    def forward(self, u, fir_prev=None, s0_re=None, s0_im=None):
        """u (B, L, D). Continuation segment: fir_prev (2, B, 3D) = projections at the 2 previous
        positions, s0_* (B, D, K) = modal state before this segment.
        Two graphs when compiled (``mixer`` then ``tail``): with the fp32 modal-conv matmuls and the
        bf16 MLP in ONE compiled graph, WikiText window 0 scored PPL 4.94 / 8.26 instead of 4.66
        (measured, deterministic; each part compiled alone, or [mixer] + [tail], is exact)."""
        y, x1v, x2, fir, s_re, s_im = self.mixer(u, fir_prev, s0_re, s0_im)
        return self.tail(u, y, x1v, x2), fir, s_re, s_im


class _HyMixer(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = [hp]          # not a submodule (avoid double registration)

    def forward(self, u, fir_prev=None, s0_re=None, s0_im=None):
        hp = self.hp[0]
        b, f = hp.blk, hp.blk.filter
        Bz, L, _ = u.shape
        z = b.projections(_norm_in(u, b.pre_norm.scale, hp.eps, f.short_filter_weight.dtype))  # (B, L, 3D)
        if fir_prev is not None:
            zc = torch.cat([fir_prev.permute(1, 0, 2), z], 1)                           # (B, L+2, 3D)
            w = f.short_filter_weight[:, 0].float()
            zf = (zc[:, 2:].float() * w[:, 2] + zc[:, 1:L + 1].float() * w[:, 1] + zc[:, :L].float() * w[:, 0]
                  + f.short_filter_bias.float()).to(z.dtype).permute(0, 2, 1)
            uc = zc.permute(0, 2, 1)
        else:
            zf = port_hyena.fir3(z, f.short_filter_weight, f.short_filter_bias).permute(0, 2, 1)
            uc = z.permute(0, 2, 1)                                                     # (B, 3D, L)
        x2, x1, v = port_hyena.split3(zf, hp.nh, hp.dh, 1)
        x1v = x1 * v
        if MODAL_IMPL == "tri" and hp.tri_ok:
            y, s_re, s_im = modal_conv_tri(x1v.to(torch.float32), hp.W_re, hp.W_im, hp.G_re, hp.G_im, hp.Pc1_re,
                                           hp.Pc1_im, hp.Utri, hp.AT, hp.q_re, hp.q_im, s0_re, s0_im)
        else:
            y, s_re, s_im = modal_conv_t(x1v.to(torch.float32), hp.H, hp.PE, hp.AT, hp.q_re, hp.q_im,
                                         s0_re, s0_im)
        Lu = uc.shape[-1]
        if Lu >= 2:
            fir = uc[..., Lu - 2:].permute(2, 0, 1).contiguous()                        # (2, B, 3D)
        else:
            fir = torch.cat([torch.zeros_like(uc), uc], -1).permute(2, 0, 1)
        return y, x1v, x2, fir, s_re, s_im


class _HyTail(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = [hp]

    def forward(self, u, y, x1v, x2):
        hp = self.hp[0]
        b, f = hp.blk, hp.blk.filter
        y = (y.to(x1v.dtype) + x1v * f.D.unsqueeze(-1)) * x2
        u = b.out_filter_dense(y.permute(0, 2, 1)).to(u.dtype) + u
        return b.mlp(_norm_in(u, b.post_norm.scale, hp.eps, x1v.dtype)).to(u.dtype) + u


class AttnPrefill(nn.Module):
    def __init__(self, blk):
        super().__init__()
        self.blk = blk
        m = blk.inner_mha_cls
        self.H, self.Hkv, self.d = m.num_heads, m.num_heads_kv, m.head_dim
        self.register_buffer("inv_freq", m.rotary_emb._compute_inv_freq(device=next(blk.parameters()).device),
                             persistent=False)
        self.eps = blk.pre_norm.eps
        self.scale = m.softmax_scale if m.softmax_scale is not None else self.d ** -0.5
        self.wb = None

    def forward(self, u, kc, vc, off=None):
        """kc, vc: (B, Hkv, W, d) zero KV windows; slots [0, L) (or [off, off+L) for a continuation
        segment, off a (1,) long tensor) are written in place."""
        b, m = self.blk, self.blk.inner_mha_cls
        B, L, _ = u.shape
        H, Hkv, d = self.H, self.Hkv, self.d
        qkv = m.Wqkv(_norm_in(u, b.pre_norm.scale, self.eps, kc.dtype))
        q = qkv[..., : H * d].reshape(B, L, H, d).transpose(1, 2)
        kv = qkv[..., H * d:].reshape(B, L, 2, Hkv, d).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        ar = torch.arange(L, device=u.device)
        pos_i = ar if off is None else ar + off
        pos = pos_i.to(torch.float32)
        fr = pos[:, None] * self.inv_freq[None, :]
        cos, sin = torch.cos(fr).to(q.dtype), torch.sin(fr).to(q.dtype)
        c2, s2 = torch.cat([cos, cos], -1), torch.cat([sin, sin], -1)
        rot = lambda x: x * c2 + torch.cat([-x[..., d // 2:], x[..., :d // 2]], -1) * s2  # noqa: E731
        q, k = rot(q), rot(k)
        if off is not None:
            # continuation: write [off, off+L), attend over the first `self.wb` window slots (a static
            # power-of-two bucket >= off+L set by the driver: one graph per bucket) with a mask
            kc.index_copy_(2, pos_i, k)
            vc.index_copy_(2, pos_i, v)
            Wn = self.wb if self.wb else kc.shape[2]
            kc, vc = kc[:, :, :Wn], vc[:, :, :Wn]
            qc = max(128, min(L, (QCHUNK_ELEMS_LONG if Wn > 8192 else QCHUNK_ELEMS) // (H * Wn)))
            n = -(-L // qc)
            qp = F.pad(q, (0, 0, 0, n * qc - L)) if n * qc != L else q
            pp = F.pad(pos_i, (0, n * qc - L)) if n * qc != L else pos_i
            kpos = torch.arange(Wn, device=u.device)[None, :]
            outs = []
            for i in range(n):
                outs.append(F.scaled_dot_product_attention(qp[:, :, i * qc:(i + 1) * qc], kc, vc,
                                                           attn_mask=kpos <= pp[i * qc:(i + 1) * qc, None],
                                                           scale=self.scale, enable_gqa=True))
            ctx = torch.cat(outs, 2)[:, :, :L]
            u = m.out_proj(ctx.transpose(1, 2).reshape(B, L, H * d)).to(u.dtype) + u
            return b.mlp(_norm_in(u, b.post_norm.scale, self.eps, kc.dtype)).to(u.dtype) + u
        kc[:, :, :L] = k
        vc[:, :, :L] = v
        qc = max(128, min(L, QCHUNK_ELEMS // (H * L)))
        if qc >= L:
            ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale, enable_gqa=True)
        else:
            n = -(-L // qc)
            qp = F.pad(q, (0, 0, 0, n * qc - L)) if n * qc != L else q
            kpos = torch.arange(L, device=u.device)[None, :]
            outs = []
            for i in range(n):
                qpos = torch.arange(i * qc, (i + 1) * qc, device=u.device)[:, None]
                if CAUSAL_SKIP:
                    e = min(L, (i + 1) * qc)            # keys beyond the chunk's last query are all masked
                    outs.append(F.scaled_dot_product_attention(qp[:, :, i * qc:(i + 1) * qc], k[:, :, :e], v[:, :, :e],
                                                               attn_mask=kpos[:, :e] <= qpos, scale=self.scale,
                                                               enable_gqa=True))
                else:
                    outs.append(F.scaled_dot_product_attention(qp[:, :, i * qc:(i + 1) * qc], k, v,
                                                               attn_mask=kpos <= qpos, scale=self.scale,
                                                               enable_gqa=True))
            ctx = torch.cat(outs, 2)[:, :, :L]
        u = m.out_proj(ctx.transpose(1, 2).reshape(B, L, H * d)).to(u.dtype) + u
        u = b.mlp(_norm_in(u, b.post_norm.scale, self.eps, kc.dtype)).to(u.dtype) + u
        return u


class PrefillHead(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.norm, self.unembed = model.norm, model.unembed

    def forward(self, u_last):
        lg = self.unembed.unembed(_norm_in(u_last, self.norm.scale, self.norm.eps, self.norm.scale.dtype))
        return lg, lg.argmax(-1)


class ScoreHead(nn.Module):
    """Final norm + LM head + log-softmax, gathered at the target ids: per-token log p (fp32) and
    whether the target is the argmax (for scoring / perplexity without moving (B, L, V) logits)."""

    def __init__(self, model):
        super().__init__()
        self.norm, self.unembed = model.norm, model.unembed

    def forward(self, u, tgt):
        lg = self.unembed.unembed(_norm_in(u, self.norm.scale, self.norm.eps, self.norm.scale.dtype)).float()
        lp = torch.log_softmax(lg, -1)
        return lp.gather(-1, tgt[..., None])[..., 0], (lg.argmax(-1) == tgt)


class Prefiller(nn.Module):
    def __init__(self, model, mode="blocks"):
        super().__init__()
        self.emb = model.embedding_layer
        dev = model.embedding_layer.weight.device
        mods = []
        for b in model.blocks:
            if hasattr(b, "filter"):
                tab = port_hyena._tables(b.filter, dev)
                mods.append(HyenaPrefill(b, tab))
            else:
                mods.append(AttnPrefill(b))
        self.blocks = nn.ModuleList(mods)
        self.is_hy = [isinstance(m, HyenaPrefill) for m in mods]
        self.head = PrefillHead(model)
        self.score_head = ScoreHead(model)
        self.Hkv = model.blocks[1].inner_mha_cls.num_heads_kv
        self.d = model.blocks[1].inner_mha_cls.head_dim
        if mode.startswith("blocks"):
            # "blocks" = all; "blocks:hy|hymix|hytail|attn|head" = only that part (debug)
            which = mode.split(":")[1] if ":" in mode else "all"
            _raise_limits()
            for m, h in zip(self.blocks, self.is_hy):
                if which in ("all", "hy", "hymix", "hytail") and h:
                    if which != "hytail":
                        m.mixer.compile(backend="neuron", dynamic=False)
                    if which != "hymix":
                        m.tail.compile(backend="neuron", dynamic=False)
                elif which in ("all", "attn") and not h:
                    m.compile(backend="neuron", dynamic=False)
            if which in ("all", "head"):
                self.head.compile(backend="neuron", dynamic=False)
                self.score_head.compile(backend="neuron", dynamic=False)

    @torch.inference_mode()
    def forward(self, ids, W, kv_dtype=None, seg=None, tgt=None):
        """Prefill ids (B, L) into a fresh DecodeState with a W-slot KV window.
        seg: segment length (multiple of the modal chunk) for a SEGMENTED prefill: the first segment
        is L - (n-1)*seg tokens, every later one exactly seg (one compiled graph per block type for
        all continuation segments). Bounds activation memory at long L."""
        B, L = ids.shape
        if kv_dtype is None:
            kv_dtype = self.emb.weight.dtype
        assert L <= W
        dev = ids.device
        C = port_hyena.CHUNK
        r = L % C
        # A length that is not a multiple of the modal chunk C is left-padded inside the modal conv.
        # For long padded graphs neuronx-cc failed with NCC_ISFV902 (SFKVectorizer index out of
        # bounds; L=1982, measured), short ones compile (L=37). So the remainder r = L % C goes first
        # as its own short segment and everything after it is C-aligned continuation segments.
        bounds = [(0, r)] if r else []
        a0 = r
        rest = L - r
        if rest:
            if seg is None or seg >= rest:
                bounds.append((a0, L))
            else:
                assert seg % C == 0, (seg, C)
                n = -(-rest // seg)
                first = rest - (n - 1) * seg
                bounds.append((a0, a0 + first))
                bounds += [(a0 + first + i * seg, a0 + first + (i + 1) * seg) for i in range(n - 1)]
        fir, s_re, s_im, ks, vs = [None] * 16, [None] * 16, [None] * 16, [], []
        scores = []
        for _ in range(sum(1 for h in self.is_hy if not h)):
            ks.append(torch.zeros(B, self.Hkv, W, self.d, dtype=kv_dtype, device=dev))
            vs.append(torch.zeros(B, self.Hkv, W, self.d, dtype=kv_dtype, device=dev))
        for si, (a, e) in enumerate(bounds):
            u = self.emb.embed(ids[:, a:e])
            if RESID_FP32:
                u = u.float()
            off = None if si == 0 else torch.tensor([a], dtype=torch.long, device=dev)
            wb = 1
            while wb < e:
                wb *= 2
            wb = min(max(wb, WB_MIN), W)
            hi = ai = 0
            for i, m in enumerate(self.blocks):
                if self.is_hy[i]:
                    if si == 0:
                        u, fir[hi], s_re[hi], s_im[hi] = m(u)
                    else:
                        u, fir[hi], s_re[hi], s_im[hi] = m(u, fir[hi], s_re[hi], s_im[hi])
                    hi += 1
                else:
                    m.wb = wb
                    u = m(u, ks[ai], vs[ai], off)
                    ai += 1
            if tgt is not None:
                scores.append(self.score_head(u, tgt[:, a:e]))
        if tgt is not None:
            return torch.cat([x[0] for x in scores], 1), torch.cat([x[1] for x in scores], 1)
        fir, s_re, s_im = fir[:hi], s_re[:hi], s_im[:hi]
        lg, tok = self.head(u[:, -1])
        S = DecodeState(fir, s_re, s_im, ks, vs, torch.tensor([L], dtype=torch.long, device=dev), W)
        return lg, tok, S
