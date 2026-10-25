#!/bin/bash
# One-command reproduction of the StripedHyena-Hessian-7B decode / TTFT benchmark on PyTorch Native
# (public Neuron SDK 2.33 DLAMI, torch-neuronx 2.14.3), one trn2.3xlarge = 1 Trainium2 device (4 logical cores, LNC=2).
#
# Prereqs (README.md): trn2.3xlarge, >= 64 GB swap, the SDK 2.33 DLAMI venv active
#   source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
#
# Usage:
#   bash run.sh            # install + CPU/simulator tests + headline benchmark (TP=4) -> results/results.json
#   bash run.sh bench      # headline benchmark only (BS=1 latency, best batch per context, TTFT)
#   bash run.sh test-hw    # on-device parity gates vs the CUDA reference + batched generation (~25 min cold)
#   bash run.sh quick      # one point: TP=4 B=1 @ 2K step graph + TTFT @ 2K (~15 min cold)
#
# Env: SHN_WEIGHTS (default ~/weights/hessian-7b; downloaded from the HF Hub if missing).
set -e
MODE="${1:-all}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# Environment check: the public Neuron SDK 2.33 torch-neuronx venv.
python - <<'PY2' || { echo "Activate /opt/aws_neuronx_venv_torch_neuronx_2_14_0 (Neuron SDK 2.33 DLAMI) first"; exit 1; }
import importlib.metadata as m, sys
v = {p: m.version(p) for p in ("torch", "torch-neuronx", "neuronx-cc")}
print("torch", v["torch"], "| torch-neuronx", v["torch-neuronx"], "| neuronx-cc", v["neuronx-cc"])
sys.exit(0 if v["torch-neuronx"].startswith("2.14.3") else 1)
PY2

export SHN_WEIGHTS="${SHN_WEIGHTS:-$HOME/weights/hessian-7b}"
export NEURON_LOGICAL_NC_CONFIG="${NEURON_LOGICAL_NC_CONFIG:-2}"
HF_REV=59e474c241746a77ff2439ed39dd5469381d3f92

# --- setup (idempotent) ---------------------------------------------------------------
python -m pip install -q -e package --no-deps          # --no-deps: never let pip touch the DLAMI torch stack
python -c "import striped_hyena_neuron" || { echo "package install failed"; exit 1; }
if ! ls "$SHN_WEIGHTS"/model-0000*-of-00002.safetensors >/dev/null 2>&1; then
    echo "==> Downloading StripedHyena-Hessian-7B (15.3 GB, rev $HF_REV) to $SHN_WEIGHTS"
    python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("togethercomputer/StripedHyena-Hessian-7B", revision="$HF_REV", local_dir="$SHN_WEIGHTS",
                  allow_patterns=["*.safetensors", "*.json"])
PY
fi
mkdir -p results
TP4="torchrun --standalone --nproc_per_node=4 harness/bench.py"
OUT="--out results/results.json"

if [ "$MODE" = "all" ]; then
    echo "==> CPU parity vs the unpatched upstream model + NKI simulator (no device)"
    (cd package && python -m pytest tests -m "cpu or simulator" -q 2>&1 | tail -2)
fi
if [ "$MODE" = "quick" ]; then
    rm -f results/results.json
    $TP4 decode --tp 4 --B 1 --ctx 2048 $OUT
    $TP4 ttft --tp 4 --L 2048 $OUT
fi
if [ "$MODE" = "all" ] || [ "$MODE" = "bench" ]; then
    # One configuration per process (PyTorch Native compiles one graph set per shape). First run compiles
    # every graph (~5-15 min per point); later runs hit the compile cache.
    rm -f results/results.json
    for C in 2048 8192 32768; do $TP4 decode --tp 4 --B 1 --ctx $C $OUT; done          # BS=1 latency (whole-step graph)
    $TP4 decode --tp 4 --B 512 --ctx 2048 $OUT                                         # best batch per context
    $TP4 decode --tp 4 --B 128 --ctx 8192 $OUT
    $TP4 decode --tp 4 --B 30 --ctx 32768 $OUT
    for L in 128 2048 8192 32512; do $TP4 ttft --tp 4 --L $L $OUT; done
fi
if [ "$MODE" = "test-hw" ]; then
    (cd package && python -m pytest tests -m hardware -q -s 2>&1 | tail -4) | tee results/hardware_tests.txt
fi
echo ""
echo "Done. Compare results/results.json with results/results_reference.json."
