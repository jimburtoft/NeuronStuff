"""Genome-scale scoring: tokenize, window, run on the device, stitch per-site outputs.

    import phylogpn_neuron as pg
    scorer = pg.Scorer.build(L=16384, B=4)          # compiles once for this (B, L)
    logits = scorer.score(dna_string)               # [len(dna), 4] float32, ACGT log-rates

`Scorer` keeps `depth` batches in flight, so host-side tokenizing, H2D, compute and D2H overlap.
Each process owns its NeuronCores; run one process per logical core (see harness/bench.py) for
full-device throughput.
"""
import os

import numpy as np
import torch

from .model import HALF, PAD_ID, RECEPTIVE_FIELD, UNK_ID

_LUT = np.full(256, UNK_ID, np.uint8)
for _i, _c in enumerate(b"ACGT"):
    _LUT[_c] = _i
    _LUT[_c + 32] = _i  # lower case
_LUT[ord("-")] = PAD_ID


def tokenize(seq):
    """str or bytes (A/C/G/T, any case; anything else -> N) -> uint8 numpy ids."""
    b = seq.encode() if isinstance(seq, str) else bytes(seq)
    return _LUT[np.frombuffer(b, np.uint8)]


class Scorer:
    def __init__(self, fn, L, B, depth, device, out_dtype):
        self.f, self.L, self.B, self.depth, self.dev, self.out_dtype = fn, L, B, depth, device, out_dtype
        self.stride = L - (RECEPTIVE_FIELD - 1)

    @classmethod
    def build(cls, model=None, L=16384, B=4, depth=3, kernel=True, kernel_cfg=None, out_dtype=torch.bfloat16,
              weights=None):
        """Load (if needed), centre, cast to bf16, move to the NeuronCore, enable the kernel, compile, warm up."""
        import torch_neuronx  # noqa: F401  (registers the neuron device and backend)
        from .hub import load
        assert L > RECEPTIVE_FIELD, "L must exceed the 481-position receptive field"
        dev = torch.device("neuron")
        m = model if model is not None else load(weights)
        if kernel:
            m.center_()
        m = m.to(torch.bfloat16).to(dev).eval().requires_grad_(False)
        if kernel:
            m.use_kernel(kernel_cfg)

        class Head(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, ids):
                return self.m(ids.long())[1].to(out_dtype)

        compiled = torch.compile(Head(m), backend="neuron", dynamic=False)

        def f(ids):
            with torch.no_grad():
                return compiled(ids)

        s = cls(f, L, B, depth, dev, out_dtype)
        with torch.no_grad():
            for _ in range(2):
                f(torch.full((B, L), PAD_ID, dtype=torch.uint8).to(dev)).cpu()
        return s

    def __call__(self, ids):
        """Run one [B, L] batch of token ids (uint8/int64, CPU or device); returns [B, L-480, 4] on the device."""
        return self.f(ids.to(self.dev))

    def windows(self, n):
        return (n + self.stride - 1) // self.stride

    def _fill(self, ids, n, bi, buf):
        buf.fill_(PAD_ID)
        nb = buf.numpy()
        for j in range(self.B):
            w = bi * self.B + j
            if w * self.stride >= n:
                break
            s0 = w * self.stride - HALF
            lo, hi = max(s0, 0), min(s0 + self.L, n)
            nb[j, lo - s0:hi - s0] = ids[lo:hi]

    @torch.no_grad()
    def score(self, seq, out=None):
        """Per-site ACGT log-rates for a whole sequence (str/bytes or uint8 ids). Returns float32 [n, 4]."""
        ids = tokenize(seq) if not isinstance(seq, np.ndarray) else seq
        n = len(ids)
        if out is None:
            out = np.empty((n, 4), np.float32)
        nb = (self.windows(n) + self.B - 1) // self.B
        bufs = [torch.empty((self.B, self.L), dtype=torch.uint8) for _ in range(self.depth + 1)]
        inflight = []

        def drain(bi, res):
            r = res.float().numpy()
            for j in range(self.B):
                o0 = (bi * self.B + j) * self.stride
                if o0 >= n:
                    break
                k = min(self.stride, n - o0)
                out[o0:o0 + k] = r[j, :k]

        for bi in range(nb):
            buf = bufs[bi % (self.depth + 1)]
            self._fill(ids, n, bi, buf)
            inflight.append((bi, self.f(buf.to(self.dev, non_blocking=True))))
            if len(inflight) > self.depth:
                pbi, pres = inflight.pop(0)
                drain(pbi, pres.cpu())
        for pbi, pres in inflight:
            drain(pbi, pres.cpu())
        return out
