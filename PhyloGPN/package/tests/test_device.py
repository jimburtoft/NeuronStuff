"""On-device end-to-end check (trn2): Scorer output vs CPU fp32 on the same sequence."""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.device
pytest.importorskip("torch_neuronx")

import phylogpn_neuron as pg  # noqa: E402


@pytest.mark.parametrize("kernel", [True, False])
def test_scorer_matches_cpu(kernel):
    rng = np.random.default_rng(0)
    seq = rng.choice(np.frombuffer(b"ACGT", np.uint8), 40_000).tobytes()
    s = pg.Scorer.build(L=8192, B=2, kernel=kernel)
    got = s.score(seq)
    cpu = pg.load()
    ids = torch.full((1, len(seq) + 480), pg.PAD_ID, dtype=torch.long)
    ids[0, 240:240 + len(seq)] = torch.from_numpy(pg.tokenize(seq).astype(np.int64))
    with torch.no_grad():
        ref = cpu(ids)[1][0].numpy()
    cos = float((got * ref).sum() / np.linalg.norm(got) / np.linalg.norm(ref))
    agree = float((got.argmax(-1) == ref.argmax(-1)).mean())
    assert cos > 0.9997 and agree > 0.998, (cos, agree)
