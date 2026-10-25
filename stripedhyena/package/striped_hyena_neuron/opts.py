"""measured decode levers, each switchable for A/B (``opts.enable([...])`` before building the Decoder).

  pretrans   store every decode GEMV weight transposed (in, out) contiguous and compute ``x @ Wt``
             (measured: with (out, in) weights ~80% of TensorEngine FLOPs were compiler
             weight-transposes). Includes the LM head. Transposed via host memory, original freed.
  mlp_cat    l1 + l2 as ONE GEMV against cat([W1; W2]) (the Evo 2 port mlp_cat_patch), aliased so the
             weights are held once. With pretrans: one (D, 2I) weight.
  rms_custom RMSNorm via ``torch.rms_norm(x.float(), (D,), w.float(), eps)``: identical formula to
             port/rmsnorm (eps inside the root), but the form torch-neuronx 2.13 lowers to the AwsNeuronRmsNorm
             custom call.
  tkg        nkilib attention_tkg for decode attention (port/tkg.py).
  cascade    fused NKI Hyena decode cascade (kernels/sh1_cascade.py): FIR + split + IIR + gate in one
             launch per layer (implies colperm).
  colperm    permute Hyena projection rows (and FIR weight/bias) once at load so the column split
             is three contiguous slices [x2 | x1 | v] instead of a per-head gather. Exact.
  nki_mlp    nkilib core.mlp.mlp for the MLPs (nki_mlp.py): TKG at small 1-core row counts (post-norm
             fusion opt-in), CTE for large prefill rows, x @ Wt otherwise (all TP-shard decode). Exclusive with mlp_cat / pretrans.
  bucket     attention reads only a power-of-two bucket >= fill of the KV window:
             with tkg via curr_sprior, with SDPA via a static slice of the cache.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

ON = set()
VALID = {"pretrans", "mlp_cat", "rms_custom", "tkg", "bucket", "colperm", "cascade", "nki_mlp"}


def enable(names):
    names = {n for n in names if n}
    bad = names - VALID
    if bad:
        raise ValueError(f"unknown opts {bad}; valid {VALID}")
    ON.update(names)
    if "cascade" in ON:
        ON.add("colperm")
    if "colperm" in ON:
        from . import hyena
        hyena.COLPERM = True
    if "rms_custom" in ON:
        from . import rmsnorm

        def rn(x, scale, eps):
            return torch.rms_norm(x.float(), (x.shape[-1],), scale.float(), eps).to(x.dtype)
        rmsnorm.rmsnorm_flash_formula = rn
        from . import decode as D
        D.rmsnorm_flash_formula = rn
        from . import prefill as P
        P.rmsnorm_flash_formula = rn


class PreTLinear(nn.Module):
    def __init__(self, wt, bias):
        super().__init__()
        self.wt = nn.Parameter(wt, requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        y = x @ self.wt
        return y if self.bias is None else y + self.bias


def _host_t(w):
    dev = w.device
    h = w.detach().to("cpu").t().contiguous()
    return h, dev


def mlp_forward(self, z):
    I_ = self._I
    w = self._w12t if getattr(self, "_w12t", None) is not None else None
    zz = z @ w if w is not None else F.linear(z, self._w12)
    return self.l3(self.act(zz[..., :I_]) * zz[..., I_:])


def apply(model):
    """Apply the weight-layout levers to a loaded model (call before Decoder/Prefiller)."""
    import gc
    from stripedhyena.layers import ParallelGatedMLP
    n = 0
    if "colperm" in ON:
        from . import hyena
        for b in model.blocks:
            if hasattr(b, "filter"):
                f = b.filter
                idx = hyena.colperm_index(f.num_attention_heads, f.hidden_size_per_attention_head).to(
                    b.projections.weight.device)
                with torch.no_grad():
                    b.projections.weight.data = b.projections.weight.data[idx].contiguous()
                    b.projections.bias.data = b.projections.bias.data[idx].contiguous()
                    f.short_filter_weight.data = f.short_filter_weight.data[idx].contiguous()
                    f.short_filter_bias.data = f.short_filter_bias.data[idx].contiguous()
                n += 1
    if "nki_mlp" in ON:
        assert not ({"mlp_cat", "pretrans"} & ON), "nki_mlp is exclusive with mlp_cat / pretrans"
        from . import nki_mlp
        n += nki_mlp.apply(model)
    if "mlp_cat" in ON:
        for m in model.modules():
            if isinstance(m, ParallelGatedMLP):
                dev = m.l1.weight.device
                if "pretrans" in ON:
                    w1, _ = _host_t(m.l1.weight)
                    w2, _ = _host_t(m.l2.weight)
                    m.l1 = m.l2 = None
                    m._w12t = torch.cat([w1, w2], 1).contiguous().to(dev)
                else:
                    w12 = torch.cat([m.l1.weight.detach().cpu(), m.l2.weight.detach().cpu()], 0)
                    m.l1 = m.l2 = None
                    m._w12 = w12.to(dev)
                m._I = m._w12t.shape[1] // 2 if getattr(m, "_w12t", None) is not None else m._w12.shape[0] // 2
                m.forward = mlp_forward.__get__(m)
                n += 1
                gc.collect()
                from .decode import empty_cache
                empty_cache()
    if "pretrans" in ON:
        targets = []
        for parent in model.modules():
            for name, ch in parent.named_children():
                if type(ch) is nn.Linear:
                    targets.append((parent, name))
        for parent, name in targets:
            ch = getattr(parent, name)
            h, dev = _host_t(ch.weight)
            b = None if ch.bias is None else ch.bias.detach()
            ch.weight = None
            setattr(parent, name, PreTLinear(h.to(dev), b))
            n += 1
        # LM head: VocabParallelEmbedding.unembed = u @ weight.T
        ue = model.unembed
        if ue is not model.embedding_layer:
            h, dev = _host_t(ue.weight)
            ue.weight = None
            ue._wt = h.to(dev)
            ue.unembed = (lambda u, _ue=ue: u @ _ue._wt)
            n += 1
    gc.collect()
    from .decode import empty_cache
    empty_cache()
    return n
