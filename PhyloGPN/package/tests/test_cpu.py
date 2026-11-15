"""CPU parity: phylogpn_neuron vs the Hugging Face model (trust_remote_code) at the pinned revision."""
import numpy as np
import pytest
import torch

import phylogpn_neuron as pg

pytestmark = pytest.mark.cpu


@pytest.fixture(scope="module")
def models():
    transformers = pytest.importorskip("transformers")
    hf = transformers.AutoModel.from_pretrained(pg.HUB_ID, revision=pg.HUB_REVISION, trust_remote_code=True).eval()
    return hf, pg.load()


def _ids(L, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 4, (2, L), generator=g)


def test_matches_hub_model(models):
    hf, m = models
    ids = _ids(1500)
    with torch.no_grad():
        ref = hf(ids)
        ref = torch.stack([ref[k] for k in "ACGT"], -1)
        ref_e = hf.get_embeddings(ids)
        e, lg = m(ids)
    assert (lg - ref).abs().max() < 1e-4
    assert (e - ref_e).abs().max() < 1e-4


def test_centering_is_exact(models):
    _, m = models
    import copy
    mc = copy.deepcopy(m).center_()
    ids = _ids(1200, 1)
    with torch.no_grad():
        a, b = m(ids)[1], mc(ids)[1]
    assert (a - b).abs().max() < 1e-4


def test_tokenize():
    assert pg.tokenize("ACGTNacgtx-").tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5]
