"""Pure-torch drop-in for ``flash_attn.modules.mha.MHA`` as used by StripedHyena 1.

StripedHyena's ``AttentionBlock`` constructs ``flash_attn.modules.mha.MHA`` unconditionally (there is
no fallback branch in ``stripedhyena/model.py``). flash-attn is CUDA-only, so on CPU / Neuron the
module must be replaced before ``stripedhyena.model`` is imported.

Semantics reproduced (flash-attn 2.x, GQA path: ``num_heads_kv != num_heads``, no cross-attn,
no dwconv, no return_residual):

- parameters ``Wqkv`` (no bias here), ``out_proj``, ``rotary_emb.inv_freq`` -- same names, so the
  HF checkpoint loads unchanged;
- ``Wqkv(x)`` -> ``q = [..., :H*d]`` as ``(h d)``, ``kv = [..., H*d:]`` as ``(two hkv d)``;
- rotary: non-interleaved (rotate-half), ``rotary_emb_dim = head_dim``, applied to q and k only.
  flash-attn's ``RotaryEmbedding`` uses fp32 positions and **recomputes inv_freq in fp32 if the
  buffer is not fp32** (the checkpoint stores it as bf16), then casts cos/sin to the activation
  dtype. Reproduced exactly;
- causal softmax attention, scale ``1/sqrt(d)``, GQA by repeating KV heads;
- KV cache in ``inference_params.key_value_memory_dict[layer_idx]`` shaped
  ``(B, max_seqlen, 2, Hkv, d)``, written at ``seqlen_offset`` (flash-attn's layout).

``base ** t`` is written as ``exp(t * ln base)`` (scalar-base pow has no Neuron lowering;
the Evo 2 port ``rotary_patch.py``).
"""
import math
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.register_buffer("inv_freq", self._compute_inv_freq(device), persistent=False)

    def _compute_inv_freq(self, device=None):
        t = torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim
        return 1.0 / torch.exp(t * math.log(self.base))

    def cos_sin(self, positions, dtype):
        # flash-attn: positions in fp32; inv_freq recomputed in fp32 when the buffer is not fp32
        inv_freq = self.inv_freq
        if inv_freq.dtype != torch.float32:
            inv_freq = self._compute_inv_freq(device=positions.device)
        freqs = torch.outer(positions.to(torch.float32), inv_freq.to(positions.device))
        return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def _apply_rotary(x, cos, sin):
    # x: (B, S, H, d); cos/sin: (S, d/2) -> rotate-half (non-interleaved)
    d2 = cos.shape[-1]
    cos = torch.cat([cos, cos], -1)[None, :, None, :]
    sin = torch.cat([sin, sin], -1)[None, :, None, :]
    x1, x2 = x[..., :d2], x[..., d2:]
    rot = torch.cat([-x2, x1], -1)
    return x * cos + rot * sin


class MHA(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_heads_kv=None,
        qkv_proj_bias=True,
        out_proj_bias=True,
        causal=True,
        layer_idx=None,
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        use_flash_attn=False,
        softmax_scale=None,
        device=None,
        dtype=None,
        **unused,
    ):
        super().__init__()
        assert causal, "StripedHyena only uses causal self-attention"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv or num_heads
        assert num_heads % self.num_heads_kv == 0
        self.head_dim = embed_dim // num_heads
        self.layer_idx = layer_idx
        self.rotary_emb_dim = rotary_emb_dim
        self.softmax_scale = softmax_scale
        self.use_flash_attn = use_flash_attn  # recorded only
        fk = {"device": device, "dtype": dtype}
        qkv_dim = self.head_dim * (num_heads + 2 * self.num_heads_kv)
        self.Wqkv = nn.Linear(embed_dim, qkv_dim, bias=qkv_proj_bias, **fk)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias, **fk)
        if rotary_emb_dim > 0:
            assert rotary_emb_dim == self.head_dim, "partial rotary not used by StripedHyena"
            self.rotary_emb = RotaryEmbedding(rotary_emb_dim, base=rotary_emb_base, device=device)

    def _kv_cache(self, inference_params, B, dtype, device):
        d = inference_params.key_value_memory_dict
        if self.layer_idx not in d:
            d[self.layer_idx] = torch.zeros(
                inference_params.max_batch_size, inference_params.max_seqlen, 2,
                self.num_heads_kv, self.head_dim, dtype=dtype, device=device,
            )
        return d[self.layer_idx]

    def forward(self, x, inference_params=None, **kwargs):
        B, S, _ = x.shape
        H, Hkv, d = self.num_heads, self.num_heads_kv, self.head_dim
        qkv = self.Wqkv(x)
        q = qkv[..., : H * d].reshape(B, S, H, d)
        kv = qkv[..., H * d :].reshape(B, S, 2, Hkv, d)
        k, v = kv[:, :, 0], kv[:, :, 1]

        offset = 0 if inference_params is None else inference_params.seqlen_offset
        if self.rotary_emb_dim > 0:
            pos = torch.arange(offset, offset + S, device=x.device)
            cos, sin = self.rotary_emb.cos_sin(pos, q.dtype)
            q, k = _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)

        if inference_params is not None:
            cache = self._kv_cache(inference_params, B, k.dtype, x.device)
            cache[:B, offset : offset + S, 0] = k
            cache[:B, offset : offset + S, 1] = v
            k = cache[:B, : offset + S, 0]
            v = cache[:B, : offset + S, 1]

        # (B, H, S, d); GQA: repeat KV heads (flash-attn maps q head h -> kv head h // (H/Hkv))
        qh = q.transpose(1, 2)
        kh = k.transpose(1, 2).repeat_interleave(H // Hkv, dim=1)
        vh = v.transpose(1, 2).repeat_interleave(H // Hkv, dim=1)
        T = kh.shape[2]
        if S == 1:
            mask = None  # one query attends to all cached keys
        else:
            # bottom-right aligned causal mask (flash-attn semantics with a KV cache)
            mask = torch.ones(S, T, dtype=torch.bool, device=x.device).tril(diagonal=T - S)
        scale = self.softmax_scale if self.softmax_scale is not None else d ** -0.5
        ctx = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, scale=scale)
        ctx = ctx.transpose(1, 2).reshape(B, S, H * d)
        return self.out_proj(ctx)


def install():
    """Register stub ``flash_attn`` modules so ``stripedhyena.model`` imports this MHA."""
    if "flash_attn.modules.mha" in sys.modules and getattr(
        sys.modules["flash_attn.modules.mha"], "MHA", None
    ) is MHA:
        return
    fa = types.ModuleType("flash_attn")
    fa.__path__ = []
    mods = types.ModuleType("flash_attn.modules")
    mods.__path__ = []
    mha = types.ModuleType("flash_attn.modules.mha")
    mha.MHA = MHA
    layers = types.ModuleType("flash_attn.layers")
    layers.__path__ = []
    rot = types.ModuleType("flash_attn.layers.rotary")
    rot.RotaryEmbedding = RotaryEmbedding
    for n, m in {
        "flash_attn": fa,
        "flash_attn.modules": mods,
        "flash_attn.modules.mha": mha,
        "flash_attn.layers": layers,
        "flash_attn.layers.rotary": rot,
    }.items():
        sys.modules[n] = m
