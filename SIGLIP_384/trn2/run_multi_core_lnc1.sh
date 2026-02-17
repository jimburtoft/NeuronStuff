#!/bin/bash
# Run multi-core inference with LNC=1 configuration
# Uses 8 logical cores on trn2.3xlarge

set -e

echo "================================================================================"
echo "SigLIP 384px - Multi-Core Inference (LNC=1)"
echo "================================================================================"
echo "Configuration: LNC=1 (8 logical cores)"
echo "Expected: Each core runs at ~14.4 img/s independently"
echo "================================================================================"

# Check if model exists
if [ ! -f "siglip_384_neuron_lnc1.pt" ]; then
    echo "Error: siglip_384_neuron_lnc1.pt not found"
    echo "Run: NEURON_LOGICAL_NC_CONFIG=1 python3 compile_model_lnc1.py"
    exit 1
fi

# Set LNC configuration
export NEURON_LOGICAL_NC_CONFIG=1

echo ""
echo "Starting 8 worker processes..."

# Start workers on cores 0-7
for core_id in 0 1 2 3 4 5 6 7; do
    echo "  Starting Core $core_id..."
    NEURON_RT_VISIBLE_CORES=$core_id python3 inference_worker.py $core_id &
    eval "PID$core_id=$!"
    
    # Stagger startup to avoid contention
    if [ $core_id -lt 7 ]; then
        sleep 2
    fi
done

echo ""
echo "All 8 cores running. Press Ctrl+C to stop."
echo ""

# Wait for interrupt
trap "echo ''; echo 'Stopping...'; kill $PID0 $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7 2>/dev/null; exit 0" INT

wait
