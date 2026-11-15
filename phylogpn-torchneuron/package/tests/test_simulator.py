"""Fused block kernel vs the PyTorch block on the NKI CPU simulator (needs `pip install nki`)."""
import os

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.simulator
nki = pytest.importorskip("nki")
ml_dtypes = pytest.importorskip("ml_dtypes")

import phylogpn_neuron as pg  # noqa: E402
from phylogpn_neuron import kernel  # noqa: E402


@pytest.fixture(scope="module")
def centred():
    return pg.load().center_()


@pytest.mark.parametrize("dil,L,B", [(1, 1200, 1), (5, 2100, 2), (5, 40, 1)])
def test_block_kernel(centred, dil, L, B):
    m = centred
    blk = m.blocks[0 if dil == 1 else 1]
    ids = torch.randint(0, 4, (B, L), generator=torch.Generator().manual_seed(dil))
    with torch.no_grad():
        x = m.embedding(ids).transpose(1, 2)
        if dil == 5:
            x = m.blocks[0](x)
        x = x.bfloat16().float()
        ref = blk(x)
    w1, w2, w3, vecs = kernel.prepare_block_weights(blk)
    bf = ml_dtypes.bfloat16
    c = kernel.DEFAULT_CFG
    k = kernel.make_kernel(dil, c["nch"], c["sq"], c["rsq"], c["ev"], c["norm"], c["acc"])
    out = np.asarray(nki.simulate(k)(x.numpy().astype(bf), w1.numpy().astype(bf), w2.numpy().astype(bf),
                                     w3.numpy().astype(bf), vecs.numpy().astype(np.float32))).astype(np.float32)
    out = torch.from_numpy(out)
    br, rb = (out - x).double(), (ref - x).double()
    cos = (br * rb).sum() / (br.norm() * rb.norm())
    assert cos > 0.9999, cos
