"""On-device tests (trn2, PyTorch Native, real Hessian-7B weights). Each test runs in its OWN
process (one compiled shape set per process; a process holds its NeuronCores until exit):

    SHN_WEIGHTS=~/weights/hessian-7b pytest tests -m hardware -s        [SHN_REFDATA=<dir with ref/*.pt> optional]

Gates vs the flash-attn CUDA reference (upstream code, same weights), teacher-forced over 128 positions of two
prompts: with the full reference logits (SHN_REFDATA) rel L2 <= 0.03, top-1 >= 0.98, mean KL <= 0.005; with the
compact reference shipped in data/ (top-16 log-probs) top-1 >= 0.98 and top-16 KL <= 0.005. 0 CPU fallback ops.
"""
import json
import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.hardware, pytest.mark.weights]
HERE = os.path.dirname(os.path.abspath(__file__))
W = os.environ.get("SHN_WEIGHTS", os.path.expanduser("~/weights/hessian-7b"))
REF = os.environ.get("SHN_REFDATA", "")


def _run(args, nproc=1, timeout=3600, env=None):
    cmd = ([sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}"] if nproc > 1
           else [sys.executable]) + [os.path.join(HERE, "_hw_worker.py")] + args
    e = dict(os.environ, SHN_WEIGHTS=W, SHN_REFDATA=REF, **(env or {}))
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=e)
    lines = [l for l in p.stdout.splitlines() if l.startswith("HWRESULT ")]
    assert p.returncode == 0 and lines, (p.stdout[-3000:] + p.stderr[-4000:])
    r = json.loads(lines[-1][len("HWRESULT "):])
    print(json.dumps(r))
    return r


def _gate(r):
    assert r["fallback_ops"] == [], r["fallback_ops"]
    for k, v in r["parity"].items():
        if r.get("reference") == "compact":
            assert v["top1"] >= 0.98 and v["kl_top16"] <= 0.005, (k, v)
        else:
            assert v["rel_l2"] <= 0.03 and v["top1"] >= 0.98 and v["kl_mean"] <= 0.005, (k, v)


# Full reference logits (SHN_REFDATA/ref/*.pt) if present, else the compact top-16 reference in data/ (always shipped).
needs_ref = pytest.mark.skipif(not (os.path.isdir(os.path.join(REF, "ref")) or
                                    os.path.exists(os.path.join(HERE, "..", "data", "parity_reference.json"))),
                               reason="no parity reference (SHN_REFDATA or data/parity_reference.json)")


@needs_ref
def test_parity_single_core_default():
    _gate(_run(["parity"], env={"NEURON_RT_VISIBLE_CORES": os.environ.get("NEURON_RT_VISIBLE_CORES", "0")}))


@needs_ref
def test_parity_single_core_cascade():
    _gate(_run(["parity", "--cascade"], env={"NEURON_RT_VISIBLE_CORES": os.environ.get("NEURON_RT_VISIBLE_CORES", "0")}))


@needs_ref
def test_parity_tp4_step():
    _gate(_run(["parity", "--tp", "4", "--mode", "step"], nproc=4))


@needs_ref
def test_parity_tp4_tkg():
    _gate(_run(["parity", "--tp", "4", "--tkg", "--mode", "blocks:32"], nproc=4))


def test_generate_batch_rows_identical():
    """gen.generate at B=1 and B=4 (copies of one prompt), separate processes: rows identical, B=4 row 0
    = the B=1 tokens, coherent text, 0 fallbacks."""
    env = {"NEURON_RT_VISIBLE_CORES": os.environ.get("NEURON_RT_VISIBLE_CORES", "0")}
    r1 = _run(["generate", "--B", "1"], env=env)
    r4 = _run(["generate", "--B", "4"], env=env)
    assert r1["fallback_ops"] == [] and r4["fallback_ops"] == []
    assert r4["rows_identical"], r4
    assert r4["tokens_row0"] == r1["tokens_row0"], (r1["text"], r4["text"])
    assert len(r1["text"].split()) >= 10, r1
