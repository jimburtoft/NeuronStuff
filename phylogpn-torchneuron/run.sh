#!/bin/bash
# PhyloGPN on one trn2.3xlarge (1 Trainium2 device, 4 logical NeuronCores at LNC=2), public Neuron SDK 2.33 DLAMI.
#
#   source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
#   bash run.sh            # install + CPU tests + device test + headline benchmark -> results/results.json
#   bash run.sh bench      # headline benchmark only
#   bash run.sh quick      # short benchmark (3 M sites per core)
set -e
MODE="${1:-all}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
python - <<'PY' || { echo "Activate /opt/aws_neuronx_venv_torch_neuronx_2_14_0 (Neuron SDK 2.33 DLAMI) first"; exit 1; }
import importlib.metadata as m, sys
v = {p: m.version(p) for p in ("torch", "torch-neuronx", "neuronx-cc")}
print("torch", v["torch"], "| torch-neuronx", v["torch-neuronx"], "| neuronx-cc", v["neuronx-cc"])
sys.exit(0 if v["torch-neuronx"].startswith("2.14.3") else 1)
PY
export NEURON_LOGICAL_NC_CONFIG="${NEURON_LOGICAL_NC_CONFIG:-2}"
export NKI_ENABLE_TRACE_CACHE=0
python -m pip install -q -e package --no-deps
python -c "import phylogpn_neuron" || { echo "package install failed"; exit 1; }
mkdir -p results

if [ "$MODE" = "all" ]; then
    echo "==> CPU parity vs the Hugging Face model (downloads songlab/PhyloGPN, 332 MB)"
    (cd package && python -m pytest tests -m cpu -q -p no:cacheprovider 2>&1 | tail -2)
    echo "==> On-device check vs CPU fp32 (kernel and stock graph)"
    (cd package && NEURON_RT_VISIBLE_CORES=0 python -m pytest tests -m device -q -p no:cacheprovider 2>&1 | tail -2)
fi
if [ "$MODE" = "quick" ]; then
    python harness/bench.py --L 16384 --B 4 --sites 3000000 --check --out results/results.json
fi
if [ "$MODE" = "all" ] || [ "$MODE" = "bench" ]; then
    rm -f results/results.json
    python harness/bench.py --L 16384 --B 4 --sites 30000000 --check --out results/results.json   # NKI kernel
    python harness/bench.py --L 16384 --B 4 --sites 30000000 --no-kernel --out results/results.json  # stock graph
fi
cat results/results.json
