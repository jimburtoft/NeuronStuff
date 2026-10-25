"""Avoid aten::pow.Scalar_out in the RoPE inv_freq computation.

WHY: vortex/model/rotary.py:423 computes
    1.0 / (self.base ** (torch.arange(0, dim, 2, device=device)/dim))
`float ** Tensor` is a SCALAR-BASE pow -> `aten::pow.Scalar_out`, which has no Neuron
implementation and round-trips to the host. Isolated and confirmed: `2.0 ** tensor`
emits it, while `tensor ** 2`, `torch.pow(tensor, 2)`, and even
`torch.pow(scalar_tensor, tensor)` do NOT.

MATH (identical): base**t == exp(t * ln(base)).
"""
import torch, math

_orig = None

def install():
    global _orig
    from vortex.model import rotary as R
    tgt = R.RotaryEmbedding
    _orig = getattr(tgt, "_compute_inv_freq", None)

    def _compute_inv_freq(self, device=None):
        t = torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim
        # base**t -> exp(t * log(base)); avoids scalar-base pow
        return 1.0 / torch.exp(t * math.log(float(self.base)))

    if _orig is not None:
        tgt._compute_inv_freq = _compute_inv_freq
        return _orig
    return None

def uninstall():
    if _orig is not None:
        from vortex.model import rotary as R
        R.RotaryEmbedding._compute_inv_freq = _orig
