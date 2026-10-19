"""PyTorch Native wrappers for the NewBERT (gte) NKI kernels.

    from nki_kernels import patch_model
    model = AutoModelForSequenceClassification.from_pretrained(..., trust_remote_code=True,
                                                               attn_implementation="eager")
    # tanh-approximate GELU (see README), BF16, then patch, then move to the device and compile
    model = model.to(torch.bfloat16)
    patch_model(model)
    model = model.to("neuron")
    compiled = torch.compile(model, backend="neuron", dynamic=False)

What patch_model replaces, per NewLayer:
  * attention half (qkv_proj, RoPE, masked softmax attention, o_proj, residual, attn_ln) -> attn_block_kernel
    for L in {128, 256, 384, 512}; other lengths keep the stock attention (with nkilib attention_cte for
    L >= 768) followed by the residual+LayerNorm kernel.
  * gated MLP + residual + mlp_ln -> gated_mlp_ln_kernel
  * embedding LayerNorm -> res_ln_kernel (no residual)
Shapes the kernels cannot take (token count not a multiple of 128 x LNC, batch not divisible by LNC at
L=128) fall back to the stock module; the shape is static per compiled graph, so the choice is made once.

NEURON_LOGICAL_NC_CONFIG must be set before this module is imported (it fixes the kernel grid).
Inference only. Right-padded attention masks only.
"""
import math
import os
import sys
import types

import torch
from torch_neuronx import nki_op

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attn_block_kernel import attn_block_kernel  # noqa: E402
from attn_block_pack import pack as _pack_attn  # noqa: E402
from gated_mlp_kernel import gated_mlp_ln_kernel  # noqa: E402
from res_ln_kernel import res_ln_kernel  # noqa: E402

LNC = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
ATTN_LENGTHS = (128, 256, 384, 512)
CTE_MIN_LEN = 768          # nkilib attention_cte beats the compiler's attention from here up


# ------------------------------------------------------------------------------- residual + LayerNorm
@nki_op("gte_nki::res_ln", mutates_args={})
def res_ln(x: torch.Tensor, r: torch.Tensor, g: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    return res_ln_kernel[LNC](x, r, g, b, eps, True)


@nki_op("gte_nki::ln", mutates_args={})
def ln(x: torch.Tensor, g: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    return res_ln_kernel[LNC](x, x, g, b, eps, False)


class NkiLN(torch.nn.Module):
    """Drop-in for nn.LayerNorm(H) over the last dim, with an optional fused residual add."""

    def __init__(self, ln_mod: torch.nn.LayerNorm):
        super().__init__()
        H = ln_mod.normalized_shape[0]
        self.H, self.eps = H, float(ln_mod.eps)
        self.register_buffer("g", ln_mod.weight.detach().float().reshape(1, H).expand(128, H).contiguous())
        self.register_buffer("b", ln_mod.bias.detach().float().reshape(1, H).expand(128, H).contiguous())

    def forward(self, x, residual=None):
        shp = x.shape
        x2 = x.reshape(-1, self.H)
        if x2.shape[0] % (128 * LNC) != 0:
            s = x if residual is None else x + residual
            return torch.nn.functional.layer_norm(s, (self.H,), self.g[0].to(s.dtype), self.b[0].to(s.dtype),
                                                  self.eps)
        if residual is None:
            y = ln(x2, self.g, self.b, self.eps)
        else:
            y = res_ln(x2, residual.reshape(-1, self.H), self.g, self.b, self.eps)
        return y.reshape(shp)


# ------------------------------------------------------------------------- gated MLP + residual + LN
@nki_op("gte_nki::gated_mlp_ln", mutates_args={})
def gated_mlp_ln(x: torch.Tensor, wupT: torch.Tensor, wgT: torch.Tensor, wdT: torch.Tensor,
                 bd: torch.Tensor, g: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    return gated_mlp_ln_kernel[LNC](x, wupT, wgT, wdT, bd, g, b, eps)


class NkiMlpLN(torch.nn.Module):
    """Replaces NewGatedMLP + mlp_ln: y = LN(x + down(gelu_tanh(gate(x)) * up(x)))."""

    def __init__(self, mlp, ln_mod):
        super().__init__()
        I = mlp.intermediate_size
        w = mlp.up_gate_proj.weight.detach()           # [2I, K]: rows [0:I] up, [I:2I] gate
        self.register_buffer("wupT", w[:I].t().contiguous())
        self.register_buffer("wgT", w[I:].t().contiguous())
        self.register_buffer("wdT", mlp.down_proj.weight.detach().t().contiguous())
        K = self.wupT.shape[0]
        bc = lambda v: v.detach().float().reshape(1, K).expand(128, K).contiguous()
        self.register_buffer("bd", bc(mlp.down_proj.bias))
        self.register_buffer("g", bc(ln_mod.weight))
        self.register_buffer("b", bc(ln_mod.bias))
        self.eps, self.K = float(ln_mod.eps), K
        self._mlp, self._ln = [mlp], [ln_mod]          # stock fallback (not registered as children)

    def forward(self, x):
        x2 = x.reshape(-1, self.K)
        if x2.shape[0] % (128 * LNC) != 0:
            return self._ln[0](x + self._mlp[0](x))
        return gated_mlp_ln(x2, self.wupT, self.wgT, self.wdT, self.bd, self.g, self.b, self.eps).reshape(x.shape)


# ---------------------------------------------------------------- nkilib attention_cte for long sequences
_cte = None


def _cte_attention(self, q, k, v, attention_bias, head_mask):
    """q/k/v [B, L, nh, hd]; attention_bias [B, 1, 1, L] additive (0 valid, large negative for right padding)."""
    global _cte
    B, L, nh, hd = q.shape
    if L < CTE_MIN_LEN or head_mask is not None:
        return self._stock_attention(q, k, v, attention_bias, head_mask)
    if _cte is None:
        from torch_neuronx import wrap_nki
        from nkilib.core.attention.attention_cte import attention_cte
        _cte = wrap_nki(attention_cte)
    scale = 1.0 / math.sqrt(hd)
    qh = (q * scale).permute(0, 2, 1, 3).reshape(B * nh, L, hd)     # the bounds path requires scale=1.0
    kh = k.permute(0, 2, 1, 3).reshape(B * nh, L, hd)
    vh = v.permute(0, 2, 1, 3).reshape(B * nh, L, hd)
    vl = (attention_bias.reshape(B, L) == 0).sum(-1).to(torch.float32)
    bmax = vl.repeat_interleave(nh)[:, None, None].expand(B * nh, L, 1).contiguous()
    o = _cte[LNC](qh, kh, vh, scale=1.0, causal_mask=False, tp_q=True, tp_k=True,
                  bound_min=torch.zeros_like(bmax), bound_max=bmax)
    return o.reshape(B, nh, L, hd).permute(0, 2, 1, 3).contiguous(), None


# ------------------------------------------------------------------------------ fused attention half
@nki_op("gte_nki::attn_block", mutates_args={})
def attn_block(x: torch.Tensor, wqk: torch.Tensor, bqk: torch.Tensor, wv: torch.Tensor, bv: torch.Tensor,
               wo: torch.Tensor, bo: torch.Tensor, g: torch.Tensor, b: torch.Tensor, cs: torch.Tensor,
               kb: torch.Tensor, eps: float) -> torch.Tensor:
    return attn_block_kernel[LNC](x, wqk, bqk, wv, bv, wo, bo, g, b, cs, kb, eps)


class NkiAttnBlock(torch.nn.Module):
    """attn_ln(x + attention(x)) for one NewLayer; RoPE tables packed once per supported length."""

    def __init__(self, attn, ln_mod, rotary, lengths=ATTN_LENGTHS):
        super().__init__()
        H = attn.hidden_size
        self.H, self.eps = H, float(ln_mod.eps)
        sc = 1.0 / math.sqrt(attn.attention_head_size)
        f = lambda t: t.detach().float().numpy()
        cos, sin = rotary(torch.zeros(1, dtype=torch.float32), seq_len=max(lengths))
        pk = _pack_attn(f(attn.qkv_proj.weight), f(attn.qkv_proj.bias), f(attn.o_proj.weight), f(attn.o_proj.bias),
                        f(ln_mod.weight), f(ln_mod.bias), cos.float().numpy(), sin.float().numpy(), H, sc)
        dt = attn.qkv_proj.weight.dtype
        for k in ("wqk", "wv", "wo"):
            self.register_buffer(k, torch.from_numpy(pk[k]).to(dt))
        for k in ("bqk", "bv", "bo", "g", "b"):
            self.register_buffer(k, torch.from_numpy(pk[k]).float())
        for L in lengths:
            self.register_buffer(f"cs{L}", torch.from_numpy(pk["cs"][:, :, :L].copy()).float())
        self.lengths = set(lengths)
        self._attn = [attn]
        self.fallback_ln = NkiLN(ln_mod)

    def forward(self, x, attention_bias, rope_embeds):
        B, L, H = x.shape
        query_split = (L // 128) % LNC == 0          # odd batches at LNC=2 split query tiles across cores
        if L not in self.lengths or (B % LNC and not query_split):
            a = self._attn[0](x, attention_bias, rope_embeds)[0]
            return self.fallback_ln(a, x)
        kb = attention_bias.reshape(B, L).float().clamp(min=-30000.0).to(x.dtype)
        y = attn_block(x.reshape(B * L, H), self.wqk, self.bqk, self.wv, self.bv, self.wo, self.bo,
                       self.g, self.b, getattr(self, f"cs{L}"), kb, self.eps)
        return y.reshape(B, L, H)


def _layer_forward(self, hidden_states, attention_bias, rope_embeds=None, padding_inputs=None,
                   attention_scale=None, subset_indices=None, head_mask=None,
                   output_attentions=False, qkv_inputs=None):
    hidden_states = self.attn_block(hidden_states, attention_bias, rope_embeds)
    hidden_states = self.mlp_block(hidden_states)
    return (hidden_states,)


def patch_model(model):
    """Patch a NewModel-based model (BF16, eager attention) in place. Returns the number of layers patched."""
    rot = None
    for mod in model.modules():
        if type(mod).__name__ == "NewEmbeddings":
            rot = mod.rotary_emb
            mod.LayerNorm = NkiLN(mod.LayerNorm)
    assert rot is not None, "not a NewBERT (Alibaba-NLP new-impl) model"
    n = 0
    for mod in model.modules():
        if type(mod).__name__ == "NewAttention":
            mod._stock_attention = mod._attention
            mod._attention = types.MethodType(_cte_attention, mod)
    for mod in model.modules():
        if type(mod).__name__ == "NewLayer":
            assert "Gated" in type(mod.mlp).__name__, "expects the gated MLP variant"
            mod.mlp_block = NkiMlpLN(mod.mlp, mod.mlp_ln)
            mod.attn_block = NkiAttnBlock(mod.attention, mod.attn_ln, rot)
            mod.forward = types.MethodType(_layer_forward, mod)
            n += 1
    return n
