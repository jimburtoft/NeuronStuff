"""Fused NKI kernels. Each is validated by a simulator test in ``tests/`` at
cos_sim 1.00000000 against an fp64 reproduction of the exact upstream op chain --
per-kernel, which is the gate that does not dilute (see api.verify's docstring).
"""
