#!/bin/bash
# Evo 2 7B (arcinstitute/evo2_7b_base) batched decode on one Trainium2 device (trn2.3xlarge), TorchNeuron.
#
# Prereqs (README.md): trn2.3xlarge, Neuron SDK 2.33 DLAMI, >= 64 GB swap, venv active:
#   source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
#
# Usage:
#   bash run.sh setup      # install the package, clone vortex (pinned), download the weights (13 GB)
#   bash run.sh test-sim   # CPU + NKI-simulator tests (no device)
#   bash run.sh test-hw    # on-device tests (parity vs fp32 reference, batch invariance, TP sharding, prefill)
#   bash run.sh fill       # TP=4, max_seqlen 8192: KV fill 128 / 1024 / 4096 (B=88) and 7680 (B=80) -> results/fill.jsonl
#   bash run.sh dp4-256    # DP=4, max_seqlen 256, B=128 per core
#
# The benchmark modes re-run an arm (up to FILL_RETRIES, default 8) when it fails with the intermittent
# first-compile errors NCC_INLA001 / NCC_IBIR229 seen on SDK 2.33 (BENCHMARK.md), and never on any other error.
# Keep the compile cache (TORCH_NEURONX_NEFF_CACHE_DIR, default ~/.cache/evo2_neff) between runs.
#
# Env: VORTEX_REPO (default ~/vortex-repo), EVO2_WEIGHTS (default ~/weights).
set -e
MODE="${1:-}"
[ -n "$MODE" ] || { sed -n 2,19p "$0"; exit 1; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python - <<'PY' || { echo "Activate /opt/aws_neuronx_venv_torch_neuronx_2_14_0 (Neuron SDK 2.33 DLAMI) first"; exit 1; }
import importlib.metadata as m, sys
v = {p: m.version(p) for p in ("torch", "torch-neuronx", "neuronx-cc")}
print("torch", v["torch"], "| torch-neuronx", v["torch-neuronx"], "| neuronx-cc", v["neuronx-cc"])
sys.exit(0 if v["torch-neuronx"].startswith("2.14.3") else 1)
PY

export VORTEX_REPO="${VORTEX_REPO:-$HOME/vortex-repo}"
export EVO2_WEIGHTS="${EVO2_WEIGHTS:-$HOME/weights}"
export NEURON_LOGICAL_NC_CONFIG="${NEURON_LOGICAL_NC_CONFIG:-2}"
export TORCH_NEURONX_NEFF_CACHE_DIR="${TORCH_NEURONX_NEFF_CACHE_DIR:-$HOME/.cache/evo2_neff}"
VORTEX_COMMIT=cb229ae50581aa072972c33a81e23eb43a1330ab   # pinned; see package/VENDORED.md
mkdir -p results/logs "$TORCH_NEURONX_NEFF_CACHE_DIR"

# --- setup (idempotent) ---------------------------------------------------------------
python -m pip install -q -e package --no-deps          # --no-deps: never let pip touch the DLAMI torch stack
python -c "import rich, einops, yaml" 2>/dev/null || python -m pip install -q rich einops pyyaml
if [ ! -d "$VORTEX_REPO/.git" ]; then git clone -q https://github.com/Zymrael/vortex.git "$VORTEX_REPO"; fi
git -C "$VORTEX_REPO" fetch -q origin "$VORTEX_COMMIT" 2>/dev/null || true
git -C "$VORTEX_REPO" checkout -q "$VORTEX_COMMIT"
export PYTHONPATH="$VORTEX_REPO${PYTHONPATH:+:$PYTHONPATH}"
if [ ! -f "$EVO2_WEIGHTS/evo2_7b_base.pt" ]; then
    echo "==> Downloading evo2_7b_base.pt (13 GB) to $EVO2_WEIGHTS"
    python - <<PY
from huggingface_hub import hf_hub_download
for f in ["config.json", "evo2_7b_base.pt"]:
    print(hf_hub_download("arcinstitute/evo2_7b_base", f, local_dir="$EVO2_WEIGHTS"))
PY
fi
[ "$MODE" = "setup" ] && exit 0

if [ "$MODE" = "test-sim" ]; then
    echo "==> CPU + NKI simulator tests (no device)"
    (cd package && python -m pytest tests -m "not hardware" -q 2>&1 | tail -2) | tee results/sim_tests.txt
fi
if [ "$MODE" = "test-hw" ]; then
    echo "==> On-device tests (each test runs its device work in a subprocess)"
    (cd package && python -m pytest tests -m hardware -q 2>&1 | tail -3) | tee results/hardware_tests.txt
fi

# 8K TP=4 KV-fill sweep. FILL_MODE=offset (default, how the published rows were measured: short prompt, then the
# KV offset is set to the fill) or prompt (a real prompt of that length, bucketed prefill one row at a time:
# FILL_CHUNK_ROWS=1 default, since 8-row chunks of a 4,096-token prompt run out of device memory at B>=64).
# FILL_SPECS: B:fill pairs (best measured batch per fill). On SDK 2.33 a first compile of a new graph at B >= 48
# can fail with NCC_INLA001 / NCC_IBIR229; the same graph compiles on re-run, so
# each arm is retried up to FILL_RETRIES (default 8) times with the compile cache kept (some arms need several).
if [ "$MODE" = "fill" ]; then
    : > results/fill.jsonl
    for spec in ${FILL_SPECS:-88:128 88:1024 88:4096 80:7680}; do
        B=${spec%:*}; F=${spec#*:}
        for try in $(seq 1 "${FILL_RETRIES:-8}"); do
            NEURON_RT_VISIBLE_CORES="${NEURON_RT_VISIBLE_CORES:-0-3}" torchrun --standalone --nproc_per_node=4 \
                harness/bench_fill.py --tp 4 --bs "$B" --fill "$F" --fill-mode "${FILL_MODE:-offset}" --chunk-rows "${FILL_CHUNK_ROWS:-1}" \
                --out "results/fill_b${B}_f${F}.json" > "results/fill_b${B}_f${F}.log" 2>&1 && break
            grep -qE 'NCC_INLA001|NCC_IBIR229' "results/fill_b${B}_f${F}.log" || break
            echo "    B=$B fill=$F: first-compile NCC_INLA001/IBIR229, retry $try"
        done
        if grep -q '^RESULT ' "results/fill_b${B}_f${F}.log"; then
            grep -h '^RESULT ' "results/fill_b${B}_f${F}.log" | sed 's/^RESULT //' | tee -a results/fill.jsonl
        else
            echo "    B=$B fill=$F FAILED (see results/fill_b${B}_f${F}.log)"
        fi
    done
fi
# Short context, DP=4: four single-core processes at B=128, started 90 s apart (each loads fp32 weights on the host).
if [ "$MODE" = "dp4-256" ]; then
    # A first compile can hit the same NCC_INLA001 issue as fill; re-run the cores that produced no result.
    rm -f results/dp4_256_c*.json
    retried=0
    for try in $(seq 1 "${FILL_RETRIES:-8}"); do
        todo=""; for c in 0 1 2 3; do [ -s "results/dp4_256_c$c.json" ] || todo="$todo $c"; done
        [ -z "$todo" ] && break
        [ "$try" -gt 1 ] && { retried=1; echo "    DP=4 @256: cores$todo produced no result, retry $try"; }
        for c in $todo; do
            NEURON_RT_VISIBLE_CORES=$c python harness/bench_fill.py --tp 1 --bs 128 --max-seqlen 256 --steps 20 \
                --warmup 6 --out "results/dp4_256_c$c.json" > "results/dp4_256_c$c.log" 2>&1 &
            sleep 90
        done
        wait
    done
    # A core that needed a retry ran while fewer than four were active; for the published row, run once more with
    # the warm cache so all four measure concurrently.
    if [ "$retried" = 1 ]; then
        echo "    DP=4 @256: re-measuring all four cores concurrently on the warm cache"
        rm -f results/dp4_256_c*.json
        for c in 0 1 2 3; do
            NEURON_RT_VISIBLE_CORES=$c python harness/bench_fill.py --tp 1 --bs 128 --max-seqlen 256 --steps 20 \
                --warmup 6 --out "results/dp4_256_c$c.json" > "results/dp4_256_c$c.log" 2>&1 &
            sleep 90
        done
        wait
    fi
    python - <<'PY'
import json, glob
r = [json.load(open(f)) for f in sorted(glob.glob("results/dp4_256_c*.json"))]
print(f"DP=4 @256 B=128: {len(r)}/4 cores, step_ms {[x['step_ms'] for x in r]}, "
      f"aggregate {sum(x['tok_s'] for x in r):.0f} tok/s, toksha {sorted({x['toksha'] for x in r})}")
PY
fi
exit 0
