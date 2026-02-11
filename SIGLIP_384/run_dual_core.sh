#!/bin/bash
# Run dual-core inference
# Measured: Core 0 at 13.08 img/s, Core 1 at 13.44 img/s

set -e

echo "================================================================================"
echo "SigLIP 384px - Dual Core Inference"
echo "================================================================================"
echo "Starting two processes, one per NeuronCore"
echo "Measured: Each core runs at ~13.3 img/s independently"
echo "================================================================================"

# Activate virtual environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Check if model exists
if [ ! -f "siglip_384_neuron.pt" ]; then
    echo "✗ Error: siglip_384_neuron.pt not found"
    echo "Run: python3 compile_model.py"
    exit 1
fi

echo ""
echo "Starting Core 0..."
NEURON_RT_VISIBLE_CORES=0 python3 inference_worker.py 0 &
PID0=$!
echo "  PID: $PID0"

# Stagger startup to avoid contention
sleep 2

echo "Starting Core 1..."
NEURON_RT_VISIBLE_CORES=1 python3 inference_worker.py 1 &
PID1=$!
echo "  PID: $PID1"

echo ""
echo "Both cores running. Press Ctrl+C to stop."
echo ""

# Wait for interrupt
trap "echo ''; echo 'Stopping...'; kill $PID0 $PID1 2>/dev/null; exit 0" INT

wait
