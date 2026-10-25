"""NKI simulator parity of the fused SH1 decode cascade kernel (striped_hyena_neuron/kernels/sh1_cascade.py)
against a NumPy reference of the decode HyenaStep cascade (colperm layout), B = 1 and 3.
No Trainium needed:  pytest tests -m simulator   (needs `pip install nki`)
"""
import importlib.util
import os

import numpy as np
import pytest

pytestmark = pytest.mark.simulator
nki = pytest.importorskip("nki")
_spec = importlib.util.spec_from_file_location(
    "sh1_cascade", os.path.join(os.path.dirname(__file__), "..", "striped_hyena_neuron", "kernels", "sh1_cascade.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)          # load the kernel file alone: the package itself needs torch
sh1_cascade = _mod.sh1_cascade

try:
    from ml_dtypes import bfloat16 as bf16
except ImportError:  # noqa: E722
    bf16 = np.float32


def ref(z, fir, s_re, s_im, w, bias, p_re, p_im, r_re, r_im, Dsk):
    """z (B, 3D); fir (2, B, 3D); w (3D, 3); bias (3D,); s (B, D, K); p/r (D, K); Dsk (D,)."""
    zf = w[:, 2] * z + fir[0] * w[:, 0] + fir[1] * w[:, 1] + bias
    D = Dsk.shape[0]
    x2, x1, v = zf[:, :D], zf[:, D:2 * D], zf[:, 2 * D:]
    x1v = x1 * v
    nre = p_re * s_re - p_im * s_im + x1v[..., None]
    nim = p_re * s_im + p_im * s_re
    res = (r_re * nre - r_im * nim).sum(-1)
    y = x2 * (res + Dsk * x1v)
    return y, np.stack([fir[1], z]), nre, nim


def test_sh1_cascade_simulator():
    rng = np.random.default_rng(0)
    P, G, K = 128, 32, 2
    D = P * G
    ok = True
    for B in (1, 3):
        z = rng.standard_normal((B, 3 * D)).astype(np.float32)
        fir = rng.standard_normal((2, B, 3 * D)).astype(np.float32)
        if bf16 is not np.float32:
            z = z.astype(bf16).astype(np.float32)
            fir = fir.astype(bf16).astype(np.float32)
        s_re = rng.standard_normal((B, D, K)).astype(np.float32)
        s_im = rng.standard_normal((B, D, K)).astype(np.float32)
        w = rng.standard_normal((3 * D, 3)).astype(np.float32)
        bias = rng.standard_normal(3 * D).astype(np.float32)
        p_re, p_im = (0.9 + 0.09 * rng.random((D, K))).astype(np.float32), (0.3 * rng.random((D, K))).astype(np.float32)
        r_re, r_im = rng.standard_normal((D, K)).astype(np.float32), rng.standard_normal((D, K)).astype(np.float32)
        Dsk = rng.standard_normal(D).astype(np.float32)
        want = ref(z, fir, s_re, s_im, w, bias, p_re, p_im, r_re, r_im, Dsk)

        # kernel layouts
        zk = z.reshape(B, 3, P, G).astype(bf16)
        fk = fir.reshape(2, B, 3, P, G).astype(bf16)
        sk = lambda s: s.reshape(B, P, G, K)  # noqa: E731
        wb = np.stack([w[:, 0], w[:, 1], w[:, 2], bias])                 # (4, 3D)
        wb = wb.reshape(4, 3, P, G).transpose(2, 0, 1, 3)                # (P, 4, 3, G)
        wb = np.ascontiguousarray(np.broadcast_to(wb[:, :, None], (P, 4, B, 3, G)), dtype=np.float32)
        pl = np.stack([p_re, p_im, r_re, r_im]).reshape(4, P, G, K).transpose(1, 0, 2, 3)   # (P, 4, G, K)
        pl = np.ascontiguousarray(np.broadcast_to(pl[:, :, None], (P, 4, B, G, K)), dtype=np.float32)
        dk = np.ascontiguousarray(np.broadcast_to(Dsk.reshape(P, 1, G), (P, B, G)), dtype=np.float32)
        y, fn, nr, ni = nki.simulate(sh1_cascade)(zk, fk, sk(s_re), sk(s_im), wb, pl, dk)
        got = (np.asarray(y, np.float32).reshape(B, D), np.asarray(fn, np.float32).reshape(2, B, 3 * D),
               np.asarray(nr).reshape(B, D, K), np.asarray(ni).reshape(B, D, K))
        for name, g, wv in zip(["y", "fir_n", "s_re", "s_im"], got, want):
            rel = np.linalg.norm(g - wv) / np.linalg.norm(wv)
            tol = 1e-2 if name in ("y", "fir_n") else 1e-5
            print(f"B={B} {name:6s} rel {rel:.2e}", "ok" if rel < tol else "FAIL")
            ok &= rel < tol
    print("PASS" if ok else "FAIL")
    assert ok


