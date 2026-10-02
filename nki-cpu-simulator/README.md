# NKI CPU Simulator on a non-Trainium instance

Install and use the **NKI CPU simulator** on a plain Ubuntu instance (laptop, dev box, CI
runner, or any EC2 instance) that has **no Trainium or Inferentia device**. The simulator runs
NKI kernels on CPU via NumPy, so you can develop and correctness-test kernels before touching a
Trn/Inf instance.

## Contents

- [`nki_cpu_simulator_setup.ipynb`](./nki_cpu_simulator_setup.ipynb) — step-by-step notebook:
  prerequisites, installing the `nki` wheel from the Neuron pip index, setting the simulator
  target on a hardware-less box, running a hand-written kernel, **downloading and running a
  published HuggingFace Kernel Hub kernel** (`jburtoft/gelu-erf-neuron`), and two built-in
  debugging features.

Verified end to end on a CPU-only Ubuntu box (Python 3.12, `nki 0.6.0`, no Neuron hardware):
the add kernel passes, and the published GELU kernel runs on the simulator at `cos_sim ≈ 1.0`
vs an exact-erf reference.

## Important: set the target on a hardware-less box

With no chip to auto-detect, the simulator **falls back to `trn3` (gen4)**. Set
`NEURON_PLATFORM_TARGET_OVERRIDE` (`trn1`/`gen2`, `trn2`/`gen3`, `trn3`/`gen4`) to match the
hardware you will deploy on. It matters for FP8 and generation-specific intrinsics.

## Scope

The simulator is for **development and correctness iteration only**. It does not model
instruction scheduling, engine parallelism, SBUF/PSUM overlap, or DMA ordering, and its
wall-clock is NumPy on CPU — not Trainium cycles. A green simulator run is a checkpoint, not a
sign-off: validate final correctness and measure **all** performance numbers on real
Trainium/Inferentia hardware.

---

## Agent / copy-paste quickstart

Self-contained steps for an automated agent or anyone who wants the commands without the
notebook narrative. Runs entirely on CPU; no Neuron hardware, driver, or runtime needed.

### 1. Environment (Python must be 3.10–3.13)

```bash
# NKI wheels are published for cp310–cp313 only (not 3.9, not 3.14).
python3.12 -m venv ~/nki-sim && source ~/nki-sim/bin/activate
```

### 2. Install (from the Neuron pip index — NOT the empty PyPI placeholder)

```bash
pip install --index-url https://pip.repos.neuron.amazonaws.com \
            --extra-index-url https://pypi.org/simple/ nki
# For the published-kernel example below:
pip install huggingface_hub scipy numpy
```

The simulator is pure NumPy and does **not** require `neuronx-cc`.

### 3. Run a hand-written kernel on the simulator

```python
import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"  # REQUIRED: else defaults to trn3

import nki, nki.language as nl, numpy as np

@nki.jit
def add_kernel(a_ptr, b_ptr):
    a = nl.load(a_ptr); b = nl.load(b_ptr)
    out = nl.ndarray(a_ptr.shape, dtype=a_ptr.dtype, buffer=nl.shared_hbm)
    nl.store(out, value=nl.add(a, b))
    return out

a = np.random.rand(128, 512).astype(np.float32)
b = np.random.rand(128, 512).astype(np.float32)
np.testing.assert_allclose(nki.simulate(add_kernel)(a, b), a + b, rtol=1e-5)
print("add OK")
```

Or run an existing script unchanged: `NKI_SIMULATOR=1 python my_script.py`
(env-var form does not support JAX — use the `nki.simulate()` API for JAX).

### 4. Run a published HuggingFace Kernel Hub kernel on the simulator

Download just the kernel **source** (the simulator needs the source, not the compiled
artifact; `get_kernel(...)` is for real Neuron hardware and needs the `torch.neuron` backend
registered, which a CPU box lacks).

```python
import os
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

import importlib.util, numpy as np, nki
from huggingface_hub import hf_hub_download
from scipy.special import erf

src = hf_hub_download(
    "jburtoft/gelu-erf-neuron",
    "build/torch-neuron/nki_kernels/gelu_erf.py",
    repo_type="kernel",
)
spec = importlib.util.spec_from_file_location("gelu_erf", src)
gelu_erf = importlib.util.module_from_spec(spec); spec.loader.exec_module(gelu_erf)

x = np.random.randn(256, 768).astype(np.float32)
y = nki.simulate(gelu_erf.gelu_fwd)(x)
r = x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
cos = float((y.ravel() @ r.ravel()) / (np.linalg.norm(y) * np.linalg.norm(r)))
print("gelu cos_sim:", round(cos, 8))   # ≈ 1.0
assert cos > 0.9999
```

### 5. Verify the target was pinned (sanity check)

```python
from nki.compiler.target import resolve_target, target_to_nc_version
t = resolve_target(); print(t, "gen%d" % target_to_nc_version(t))  # want: trn2 gen3
```

### Agent notes

- **`NEURON_PLATFORM_TARGET_OVERRIDE` must be set before importing NKI internals** — on a
  hardware-less box an unset target silently resolves to `trn3`.
- A float64 input raises `ValueError: Unsupported dtype`; cast inputs to `float32`/`bfloat16`.
- Simulator wall-clock is NumPy on CPU — **never** report it as kernel latency/throughput.
- Simulator-green ≠ ship-ready. Final correctness and all perf numbers require real hardware.

## References

- NKI CPU simulator guide: <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/nki_simulator.html>
- NKI language guide: <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/nki-language-guide.html>
- Example kernel repo: <https://huggingface.co/kernels/jburtoft/gelu-erf-neuron>
