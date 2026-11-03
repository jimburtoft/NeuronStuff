"""CPU-simulator validation of the Flint MoE routing kernels (pack_scatter_positions, scatter_src_to_packed).

    python -m pip install --index-url https://pip.repos.neuron.amazonaws.com nki numpy ml_dtypes
    NEURON_LOGICAL_NC_CONFIG=2 python sim_pack_routing.py

Every case checks the NKI kernel in nki.simulate against a NumPy reference and must print exact=True.
"""
import os

os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")  # the routing kernels run at LNC=2

import numpy as np
import nki

from kernels.pack_routing import pack_scatter_positions, scatter_src_to_packed


def ref_positions(ids, counts, B):
    """Reference: block-padded per-expert packed slot for each (row, expert) entry."""
    E = counts.shape[0]
    blocks = (counts + B - 1) // B
    blk_off = np.concatenate([[0], np.cumsum(blocks)[:-1]])
    pad_row = (blk_off * B).astype(np.float32)
    cap = int(blocks.sum()) * B
    cursor = pad_row.astype(np.int64).copy()
    pos = np.empty(len(ids), dtype=np.int64)
    for i, e in enumerate(ids):
        pos[i] = cursor[e]
        cursor[e] += 1
    return pos, pad_row, cap


def main():
    for N, E, B, seed in [(256, 8, 128, 0), (512, 16, 256, 1), (300, 4, 128, 2), (1024, 64, 128, 3)]:
        rng = np.random.default_rng(seed)
        ids = rng.integers(0, E, size=N).astype(np.int32)
        counts = np.bincount(ids, minlength=E).astype(np.int32)
        pos_ref, pad_row, cap = ref_positions(ids, counts, B)
        out = np.asarray(nki.simulate(pack_scatter_positions)(ids, pad_row.reshape(1, E), cap)).reshape(-1)
        print(f"pack_scatter_positions N={N} E={E} B={B} cap={cap}: exact={np.array_equal(out, pos_ref)}")
        tpid_ref = np.full(cap, N, dtype=np.int64)
        tpid_ref[pos_ref] = np.arange(N)
        tpid = np.asarray(nki.simulate(scatter_src_to_packed)(pos_ref.astype(np.int32), cap)).reshape(-1)
        print(f"scatter_src_to_packed  N={N} E={E} B={B}: exact={np.array_equal(tpid, tpid_ref)}")


if __name__ == "__main__":
    main()
