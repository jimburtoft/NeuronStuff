"""Fuse ParallelGatedMLP's l1 and l2 into ONE GEMV by concatenating their weights.

l1 and l2 both map D -> I from the SAME input z, so
    z1 = z @ w1.T ;  z2 = z @ w2.T
is one GEMV against cat([w1; w2], dim=0) followed by a split. Halves the number of
D->I projections per layer (64 -> 32 across the model).

MEASURED isolated: l1+l2 0.684 -> 0.515 ms = 1.33x. Whole identity-MLP 0.852 -> 0.804.
Ablation (post-norm-fusion config) puts l1+l2 at 13.70 ms = 21.8% of a 63.6 ms step, so
the predicted end-to-end saving is ~3-4 ms. Above the +/-1 ms noise of the current harness,
but small -- gated and A/B'd like everything else.

WHY THIS AND NOT A KERNEL: nkilib's fused mlp covers only 1 of 32 layers (31 use
nn.Identity via evo2_style_activations), and a hand-written identity-MLP kernel needs three
levels of tiling (PSUM free dim caps at 512, Dout=4096) for ~3500 instructions -- likely
slower than the stock GEMV. This is the cheap part of that win with none of the risk.

The concatenated weight is built ONCE per module and cached (frozen weights).
"""
import os

import torch

_orig = None
# Gate the pre-transposed weight layout so it can be A/B'd like every other
# optimization in this package. Default OFF until it is measured end-to-end.
_PRETRANS = os.environ.get("EVO2_PRETRANS", "0") == "1"
# Alias l1/l2 onto _w12 so the MLP up/gate weights are held ONCE, not twice.
# Default ON (measured). EVO2_MLP_ALIAS=0 disables.
_ALIAS = os.environ.get("EVO2_MLP_ALIAS", "1") == "1"


def install():
    global _orig
    from vortex.model.layers import ParallelGatedMLP, grab_first_if_tuple

    _orig = ParallelGatedMLP.forward

    def forward(self, z):
        I = self.l1.out_features
        _wt = getattr(self, "_w12t", None)
        if isinstance(_wt, torch.Tensor):           # EVO2_WT=1: built by api._pretranspose
            zz = z @ _wt
            z1, z2 = zz[..., :I], zz[..., I:]
            return grab_first_if_tuple(self.l3(self.act(z1) * z2))
        if _PRETRANS:
            # Cache ONLY the pre-transposed (D, 2I) weight and use z @ w_t.
            #
            # nc_matmul contracts over the PARTITION axis of BOTH operands, so F.linear
            # (= z @ w.T with w stored (out,in)) makes the compiler transpose w at RUN TIME,
            # every call. Profiling showed storing (in,out) cuts transposes and LDWEIGHTS.
            #
            # This line is the single largest MATMUL source in the model: a profile attributed
            # 7.81 ms (30.4% of all MATMUL time, n=22,016) to it at B=56.
            #
            # NOTE profile/pretrans_patch.py CANNOT reach this call -- it patches
            # nn.Linear.forward, while this is a FUNCTIONAL F.linear on a weight we build
            # ourselves. Applying that patch alone measures nothing here.
            #
            # MEMORY: build the transposed copy WITHOUT ever holding both. Caching both is
            # ~180 MB per block extra, ~5.8 GB over 32 blocks, which OOM'd LNC=2's 24 GB/core
            # ("AllocBind: NRT_RESOURCE ... Failed to allocate resource"). Keyed on _w12t so
            # the concatenation happens exactly once, not every step.
            wt = getattr(self, "_w12t", None)
            if wt is None:
                if self.l1.bias is not None or self.l2.bias is not None:
                    self._w12t = False      # biases: not handled, use eager
                else:
                    # cat directly into (D, 2I): transpose each operand, then cat on dim 1.
                    # Avoids materializing the (2I, D) form at all.
                    self._w12t = torch.cat(
                        [self.l1.weight.t(), self.l2.weight.t()], 1).contiguous()
                wt = self._w12t
            if wt is False:
                return _orig(self, z)
            zz = z @ wt                                    # ONE GEMV -> (..., 2I)
        else:
            w = getattr(self, "_w12", None)
            if w is None:
                if self.l1.bias is not None or self.l2.bias is not None:
                    self._w12 = False       # biases: not handled, use eager
                else:
                    w12 = torch.cat([self.l1.weight.detach(), self.l2.weight.detach()],
                                    0).contiguous()
                    # torch.cat allocates a NEW tensor, so l1.weight and
                    # l2.weight used to stay resident ALONGSIDE _w12 -- the MLP up/gate
                    # weights held twice. Measured: 5.77 GB of duplicated weight across
                    # 32 blocks (mlp.l1 + mlp.l2 bf16, from the checkpoint), against a
                    # 4.72 GB unexplained gap in the runtime's memory table at the
                    # B=8 @ max_seqlen=8192 OOM.
                    # Re-point l1/l2 at row-slices of _w12 so all three share ONE storage.
                    # Row slices of a row-major cat are contiguous, values are unchanged,
                    # and l1(x)/l2(x) stay valid for the eager fallback. Verified on CPU:
                    # resident bytes 1.67x -> 1.00x, fused == eager bit-exactly.
                    if _ALIAS:
                        I_ = self.l1.out_features
                        self.l1.weight = torch.nn.Parameter(w12[:I_], requires_grad=False)
                        self.l2.weight = torch.nn.Parameter(w12[I_:], requires_grad=False)
                    self._w12 = w12
                w = self._w12
            if w is False:
                return _orig(self, z)
            zz = torch.nn.functional.linear(z, w)          # ONE GEMV -> (..., 2I)
        z1, z2 = zz[..., :I], zz[..., I:]
        return grab_first_if_tuple(self.l3(self.act(z1) * z2))

    ParallelGatedMLP.forward = forward


def uninstall():
    if _orig is not None:
        from vortex.model.layers import ParallelGatedMLP
        ParallelGatedMLP.forward = _orig
