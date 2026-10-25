"""Torch RMSNorm matching ``flash_attn.ops.rms_norm.rms_norm`` (the norm StripedHyena was trained with).

The standalone ``stripedhyena.layers.RMSNorm`` torch branch computes
``scale * x / (||x||_2 * d^-1/2 + eps)`` (eps OUTSIDE the root). flash-attn's fused kernel computes
``scale * x * rsqrt(mean(x^2) + eps)`` (eps INSIDE the root), in fp32, output cast to the input dtype.

These are NOT equivalent for this checkpoint: the token-embedding RMS is ~2e-3, so mean(x^2) ~ 4e-6 is
smaller than eps = 1e-5 and the first block's norm output differs by ~2x. Measured on the CUDA reference:
WikiText-2 PPL@2048 is 6.19 with the flash formula and 15.6 with the upstream torch branch.
``use_flash_rmsnorm: true`` in the shipped config is therefore load-bearing; any port must use this formula.

``install()`` patches ``RMSNorm.forward`` in the standalone package (and optionally the HF copy).
"""
import torch


def rmsnorm_flash_formula(x, scale, eps):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * scale.float()
    return y.to(x.dtype)


def _forward(self, x):
    return rmsnorm_flash_formula(x, self.scale, self.eps)


def install(module=None):
    if module is None:
        from stripedhyena import layers as module
    module.RMSNorm.forward = _forward
