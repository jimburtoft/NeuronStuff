#!/bin/bash
# Run multi-core inference with LNC=2 configuration
# Uses 4 logical cores on trn2.3xlarge

set -e

echo "================================================================================"
echo "SigLIP 384px - Multi-Core Inference (LNC=2)"
echo "================================================================================"
echo "Configuration: LNC=2 (4 logical cores)"
echo "Expected: Each core runs at ~21.9 img/s independently"
echo "================================================================================"

# Check if model exists
if [ ! -f "siglip_384_neuron.pt" ]; then
    echo "Error: siglip_384_neuron.pt not found"
    echo "Run: NEURON_LOGICAL_NC_CONFIG=2 python3 compile_model_lnc2.py"
    exit 1
fi

# Set LNC configuration
export NEURON_LOGICAL_NC_CONFIG=2

echo ""
echo "Starting 4 worker processes..."

# Start workers on cores 0-3
for core_id in 0 1 2 3; do
    echo "  Starting Core $core_id..."
    NEURON_RT_VISIBLE_CORES=$core_id python3 inference_worker.py $core_id &
    eval "PID$core_id=$!"
    
    # Stagger startup to avoid contention
    if [ $core_id -lt 3 ]; then
        sleep 2
    fi
done

echo ""
echo "All 4 cores running. Press Ctrl+C to stop."
echo ""

# Wait for interrupt
trap "echo ''; echo 'Stopping...'; kill $PID0 $PID1 $PID2 $PID3 2>/dev/null; exit 0" INT

wait
