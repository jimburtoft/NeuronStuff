"""Neuron attention for StripedHyena 1: GQA SDPA (``enable_gqa=True``) over a FIXED-WINDOW KV cache.

Subclass of ``port/sdpa_mha.MHA`` (same parameters, same RoPE numerics; validated against flash-attn
in measured). What changes for Neuron:

- KV cache is ``(2, B, Hkv, W, d)`` with W = ``inference_params.max_seqlen`` (the window), zero-
  initialised, written with ``index_copy_`` at a DEVICE position tensor. Decode attends over all W
  slots with an additive mask (slot <= pos), so every decode step has the same shapes -- no per-step
  eager re-lowering and no Dynamo value guard on ``seqlen_offset``.
- Prefill must start at offset 0 (the only way the model is used): causal SDPA over the prompt.
- GQA via ``enable_gqa=True`` (measured: ``repeat_interleave`` was 10x slower eager at B=8/32K).
- RoPE angles from the fp32 recomputed inv_freq (flash-attn semantics), positions in fp32.

Position source: ``inference_params.pos_t`` (0-D/1-elt long device tensor) if set by the driver,
else built from ``seqlen_offset``.
"""
import torch
import torch.nn.functional as F

from . import sdpa_mha

QCHUNK = 2048   # prefill query chunk (scores buffer H x QCHUNK x S)


def _rope(x, cos, sin):
    # x: (B, H, S, d); cos/sin: (S, d/2) in x.dtype -- rotate-half (non-interleaved)
    c = torch.cat([cos, cos], -1)
    s = torch.cat([sin, sin], -1)
    d2 = x.shape[-1] // 2
    rot = torch.cat([-x[..., d2:], x[..., :d2]], -1)
    return x * c + rot * s


class NeuronMHA(sdpa_mha.MHA):
    def _inv_freq32(self, device):
        inv = getattr(self, "_inv32", None)
        if inv is None or inv.device != device:
            inv = self.rotary_emb._compute_inv_freq(device=device)
            self._inv32 = inv
        return inv

    def _cos_sin(self, pos, dtype):
        freqs = pos.to(torch.float32)[:, None] * self._inv_freq32(pos.device)[None, :]
        return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)

    def forward(self, x, inference_params=None, **kwargs):
        B, S, _ = x.shape
        H, Hkv, d = self.num_heads, self.num_heads_kv, self.head_dim
        qkv = self.Wqkv(x)
        q = qkv[..., : H * d].reshape(B, S, H, d).transpose(1, 2)                  # (B,H,S,d)
        kv = qkv[..., H * d:].reshape(B, S, 2, Hkv, d).permute(2, 0, 3, 1, 4)      # (2,B,Hkv,S,d)
        k, v = kv[0], kv[1]
        scale = self.softmax_scale if self.softmax_scale is not None else d ** -0.5

        if inference_params is None or S > 1:
            offset = 0 if inference_params is None else inference_params.seqlen_offset
            assert offset == 0, "prefill must start at offset 0"
            pos = torch.arange(S, device=x.device)
            cos, sin = self._cos_sin(pos, q.dtype)
            q, k = _rope(q, cos, sin), _rope(k, cos, sin)
            if inference_params is not None:
                W = inference_params.max_seqlen
                assert S <= W, (S, W)
                cache = torch.zeros(2, B, Hkv, W, d, dtype=k.dtype, device=x.device)
                cache[0, :, :, :S] = k
                cache[1, :, :, :S] = v
                inference_params.key_value_memory_dict[self.layer_idx] = cache
            qc = max(128, min(QCHUNK, (1 << 27) // (H * S)))
            if S <= qc:
                ctx = F.scaled_dot_product_attention(q, k, v, is_causal=S > 1, scale=scale, enable_gqa=True)
            else:
                # query-chunked causal prefill: bounds the (H, qc, S) score buffer at long S. Every
                # chunk has the SAME shapes (full K/V + mask, q padded), so eager compiles once.
                n = -(-S // qc)
                qp = F.pad(q, (0, 0, 0, n * qc - S)) if n * qc != S else q
                kpos = torch.arange(S, device=x.device)[None, :]
                outs = []
                for i in range(n):
                    qpos = torch.arange(i * qc, (i + 1) * qc, device=x.device)[:, None]
                    outs.append(F.scaled_dot_product_attention(qp[:, :, i * qc:(i + 1) * qc], k, v,
                                                               attn_mask=kpos <= qpos, scale=scale,
                                                               enable_gqa=True))
                ctx = torch.cat(outs, 2)[:, :, :S]
        else:
            pos = getattr(inference_params, "pos_t", None)
            if pos is None:
                pos = torch.tensor([inference_params.seqlen_offset], device=x.device)
            pos = pos.reshape(1)
            cos, sin = self._cos_sin(pos, q.dtype)
            q, k = _rope(q, cos, sin), _rope(k, cos, sin)
            cache = inference_params.key_value_memory_dict[self.layer_idx]           # (2,B,Hkv,W,d)
            W = cache.shape[3]
            cache[0].index_copy_(2, pos, k)
            cache[1].index_copy_(2, pos, v)
            slots = torch.arange(W, device=x.device)
            mask = torch.where(slots <= pos, 0.0, float("-inf")).to(q.dtype)        # (W,)
            ctx = F.scaled_dot_product_attention(q, cache[0], cache[1], attn_mask=mask[None, None, None, :],
                                                 scale=scale, enable_gqa=True)
        ctx = ctx.transpose(1, 2).reshape(B, S, H * d)
        return self.out_proj(ctx)


def install():
    sdpa_mha.install()
    import sys
    sys.modules["flash_attn.modules.mha"].MHA = NeuronMHA
