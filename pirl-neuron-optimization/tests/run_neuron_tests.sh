#!/bin/bash

# PIRL Agent Neuron Compatibility Testing Script

echo "=== PIRL Agent Neuron Compatibility Testing ==="
echo "This script will test the PIRL agent for AWS Neuron compatibility"
echo ""

# Activate the AWS Neuron virtual environment
echo "Activating AWS Neuron virtual environment..."
source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate

# Check Python and PyTorch versions
echo "Python version:"
python --version
echo ""

echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo ""

# Check if torch_neuronx is available
echo "Checking torch_neuronx availability..."
python -c "
try:
    import torch_neuronx
    print(f'torch_neuronx version: {torch_neuronx.__version__}')
    print('✓ torch_neuronx is available')
except ImportError:
    print('✗ torch_neuronx not available')
    print('Note: Tests will run in CPU-only mode')
"
echo ""

# Run the basic CPU test first
echo "=== Running Basic CPU Test ==="
python test_pirl_agent.py
echo ""

# Run the comprehensive Neuron compatibility test
echo "=== Running Comprehensive Neuron Compatibility Test ==="
python test_neuron_compatibility.py
echo ""

echo "=== Testing Complete ==="
echo ""
echo "Summary:"
echo "1. Basic CPU functionality test completed"
echo "2. Neuron compatibility assessment completed"
echo "3. Performance benchmarks completed"
echo ""
echo "Check the output above for:"
echo "• Computational intensity ranking"
echo "• Neuron compatibility status"
echo "• Performance comparisons"
echo "• Deployment recommendations"
echo ""
echo "Next steps:"
echo "• Review the PIRL_torch_neuron.py file for the optimized implementation"
echo "• Consider using traced models for inference workloads"
echo "• Test on actual Trainium/Inferentia instances for production deployment"