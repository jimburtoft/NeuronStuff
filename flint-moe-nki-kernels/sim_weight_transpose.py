"""CPU-simulator validation of the per-expert weight pre-transpose kernels.

    NEURON_LOGICAL_NC_CONFIG=2 python sim_weight_transpose.py

Checks each transpose in nki.simulate against a NumPy reference. A transpose only moves data, so it must be exact.
"""
import os

os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")

import numpy as np
import nki

from kernels.weight_transpose import transpose_gate_up, transpose_down


def main():
    for E, H, I in [(4, 256, 128), (8, 512, 256), (16, 2048, 512)]:
        gu = np.random.randn(E, H, 2, I).astype(np.float32)
        out = np.asarray(nki.simulate(transpose_gate_up)(gu))
        print(f"transpose_gate_up E={E} H={H} I={I}: exact={np.allclose(out, gu.transpose(0, 2, 3, 1))} shape={out.shape}")
        dn = np.random.randn(E, I, H).astype(np.float32)
        out2 = np.asarray(nki.simulate(transpose_down)(dn))
        print(f"transpose_down    E={E} H={H} I={I}: exact={np.allclose(out2, dn.transpose(0, 2, 1))} shape={out2.shape}")


if __name__ == "__main__":
    main()
