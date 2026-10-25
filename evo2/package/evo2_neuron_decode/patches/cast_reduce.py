"""Reduce aten::_to_copy (79% of the eager decode step, 486 calls / 136 ms).

ATTRIBUTED CALL SITES (one decode step, 449 traced .to()/.float() calls):
  rmsnorm_patch.py:33  130  return (self.scale.to(torch.float32) * y).to(x.dtype)   <-- MINE
  rmsnorm_patch.py:30   65  xf = x.to(torch.float32)                               <-- MINE
  engine.py:428-431    144  weight/u/fir_state/bias .to(float32)  (upstream step_fir)
  engine.py:449         45  return y.to(input_dtype)
  layers.py:135         27  F.linear(x.to(self.weight.dtype), ...)
  model.py:374          27  y = y.to(dtype=self.data_dtype)
Dtype transitions: 229 bf16->fp32, 119 fp32->bf16, and 100 NO-OPS
  (55 bf16->bf16, 45 fp32->fp32) -- casts to the dtype the tensor already has.

THE NKI-AUTHORING SKILL, Anti-pattern 3 "Unnecessary type conversions":
  "BETTER: Keep in native type when precision allows. Use fp32 only where needed
   (reductions, accumulation)."

THIS PATCH does three things, cheapest first:

1. rmsnorm: `self.scale.to(torch.float32)` casts a FROZEN PARAMETER on every call
   (130 of 449 casts). Cache it once. Pure win, zero numerics change.

2. Skip no-op casts: guard every cast with a dtype check. 100 of 449 calls are
   casting to the dtype the tensor already has.

3. step_fir: keep the reduction in fp32 (precision matters there -- it is an
   accumulation over up to 127 taps) but avoid re-casting the WEIGHT every call by
   caching the fp32 weight. `u` and `fir_state` still upcast because the accumulate
   needs it.

Deliberately NOT doing: moving the FIR accumulation to bf16. The skill says use fp32
"where needed (reductions, accumulation)" and this is exactly that. Parity is the gate.
"""
import torch

_orig_rms = None
_orig_fir = None
_wcache = {}
_scache = {}


def install():
    global _orig_rms, _orig_fir
    from vortex.model.layers import RMSNorm
    from vortex.model.engine import HyenaInferenceEngine as H

    # ---- 1 + 2: rmsnorm -- cache the fp32 scale, skip no-op casts ----
    _orig_rms = RMSNorm.forward

    def rms_forward(self, x):
        if getattr(self, "use_flash_rmsnorm", False):
            return self.rmsnorm_func(x, self.scale, self.eps)
        key = id(self.scale)
        s32 = _scache.get(key)
        if s32 is None:
            s32 = self.scale.to(torch.float32)      # frozen param: cast ONCE
            _scache[key] = s32
        xf = x if x.dtype == torch.float32 else x.to(torch.float32)
        rms = torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True))
        y = xf / (rms + self.eps)          # EXACT upstream order: divide first,
        out = s32 * y                      # then scale. fp32 mul is not associative.
        return out if out.dtype == x.dtype else out.to(x.dtype)

    RMSNorm.forward = rms_forward

    # ---- 3: step_fir weight cache -- REMOVED, IT WAS A CORRECTNESS BUG ----
    # Keying on id(weight) is unsafe here: sequential_forward rebuilds the HCM weight
    # every call via h.repeat_interleave(...), so `weight` is a TEMPORARY tensor.
    # CPython reuses id()s after GC, so the cache could return a DIFFERENT layer's
    # weight -- parity fell to cos_sim 0.999929 with 3 diverging tokens.
    # The fp32 scale cache below is safe because self.scale is a real nn.Parameter
    # with a stable lifetime (verified bit-identical in isolation).

def clear():
    _wcache.clear(); _scache.clear()


def uninstall():
    from vortex.model.layers import RMSNorm
    from vortex.model.engine import HyenaInferenceEngine as H
    if _orig_rms is not None: RMSNorm.forward = _orig_rms
    if _orig_fir is not None: H.step_fir = _orig_fir
    clear()
