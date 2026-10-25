"""Route vortex RMSNorm through the fused kernel (kernels/rmsnorm_fused.py), decode only.

Re-profile: with all 32 layers on fused cascades, per-layer cost is a uniform
~3.44 ms and the SHARED GLUE is ~100% of it. RMSNorm is the largest glue item --
0.791 ms/call x 64 calls/token (2 per layer x 32 layers) = ~50 ms of a ~104 ms step.
0.791 ms for a 4096-element reduction is pure launch overhead (~7 dispatched ops), not
arithmetic. Fused: 0.210 ms = 3.77x, predicting ~37 ms/token.

Matches upstream's NON-STANDARD formula exactly (layers.py:195): eps is added to the RMS
OUTSIDE the sqrt, not to the mean-square inside it.

Handles 1 row (BS=1) and R<=128 rows (batched decode) -- rows go on the partition axis so
the whole batch is still ONE launch. Prefill (many rows x many positions) keeps eager.

MEASURED BY ABLATION (the pipelining-immune method):
    norms removed entirely (ceiling) : 37.65 ms saved
    this kernel, gamma cached        : 38.82 ms saved  (at the ceiling)
    this kernel, gamma re-flattened  : 33.35 ms saved
RMSNorm is the single largest item in the step at BS=1 (37.5 ms of ~100 ms = 38.7%).
"""
import torch

_orig = None
_KERN = None
_KERN_ROWS = None


def install():
    global _orig, _KERN, _KERN_ROWS
    from torch_neuronx import wrap_nki
    from ..kernels.rmsnorm_fused import rmsnorm_fused, rmsnorm_fused_rows
    from vortex.model.layers import RMSNorm

    _KERN = wrap_nki(rmsnorm_fused)
    _KERN_ROWS = wrap_nki(rmsnorm_fused_rows)
    _orig = RMSNorm.forward
    # eps in units of 1e-6, as an int so it stays a compile-time constant
    def forward(self, x):
        if getattr(self, "use_flash_rmsnorm", False):
            return _orig(self, x)
        D = self.hidden_size
        # DEFECT (fixed): the old guard was `x.numel() != D`, which is FALSE for
        # BS>1 -- shape (4,1,4096) has numel=16384 -- so the kernel was silently disabled
        # for every batched run. The kernel is 1-row, so gate on ROW COUNT and loop.
        if x.shape[-1] != D:
            return _orig(self, x)
        rows = x.numel() // D
        if rows > 128:
            return _orig(self, x)       # partition limit; would need tiling
        e6 = getattr(self, "_eps_e6", None)
        if e6 is None:
            e6 = int(round(float(self.eps) * 1e6))
            if abs(e6 * 1e-6 - float(self.eps)) > 1e-12 * max(1.0, float(self.eps)):
                self._eps_e6 = -1                    # not representable -> never use kernel
            else:
                self._eps_e6 = e6
            e6 = self._eps_e6
        if e6 < 0:
            return _orig(self, x)
        # cache the flattened gamma: re-doing reshape+contiguous per call costs dispatched
        # ops that eat into the win (measured: cached 38.8 ms saved vs 33.4 uncached)
        gv = getattr(self, "_g2", None)
        if gv is None:
            gv = self.scale.reshape(1, D).contiguous()
            self._g2 = gv
        if rows > 1:
            # gamma must be PRE-BROADCAST to (rows, D): a (1,D) gamma is rejected on HARDWARE
            # (nisa.tensor_tensor_arith cannot broadcast on the PARTITION axis) even though
            # the simulator accepts it silently. Cache per row-count.
            key = "_g2r%d" % rows
            gr = getattr(self, key, None)
            if gr is None:
                gr = gv.expand(rows, D).contiguous()
                setattr(self, key, gr)
        if rows == 1:
            return _KERN(x.reshape(1, D), gv, e6).reshape(x.shape)
        # batched: rows on the partition axis -> still ONE launch
        return _KERN_ROWS(x.reshape(rows, D), gr, e6).reshape(x.shape)

    RMSNorm.forward = forward


def uninstall():
    if _orig is not None:
        from vortex.model.layers import RMSNorm
        RMSNorm.forward = _orig
