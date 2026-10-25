"""Neuron-safe RMSNorm: avoid aten::pow.Scalar_out (CPU fallback).

WHY: vortex RMSNorm.forward uses
    y = x / (x.norm(2, dim=-1, keepdim=True) * hidden_size**(-0.5) + eps)
`Tensor.norm(2, ...)` lowers to `aten::pow.Scalar_out`, which is NOT implemented on
Neuron -> silently round-trips to CPU, and the result comes back on the host device
(`RuntimeError: input tensor is on cpu device, expected neuron` at `self.scale * y`).

MATH (identical, just re-expressed):
    ||x||_2 = sqrt(sum(x^2))
    ||x||_2 * H^(-1/2) = sqrt(sum(x^2)/H) = sqrt(mean(x^2))     [RMS]
so
    y = x / (sqrt(mean(x^2)) + eps)
which uses only mul / mean / sqrt -- all natively supported. Note eps is OUTSIDE the
sqrt here, matching upstream exactly (this is NOT the more common
`rsqrt(mean+eps)` form -- using that would change numerics).
"""
import torch

_orig = None

def install():
    global _orig
    from vortex.model.layers import RMSNorm
    _orig = RMSNorm.forward

    def forward(self, x):
        if getattr(self, "use_flash_rmsnorm", False):
            return self.rmsnorm_func(x, self.scale, self.eps)
        xf = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True))
        y = xf / (rms + self.eps)
        return (self.scale.to(torch.float32) * y).to(x.dtype)

    RMSNorm.forward = forward
    return _orig

def uninstall():
    if _orig is not None:
        from vortex.model.layers import RMSNorm
        RMSNorm.forward = _orig
