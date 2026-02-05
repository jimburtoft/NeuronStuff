#!/bin/bash
#
# Llama 3.3 70B - OpenAI-Compatible API Server
# =============================================
#
# This script starts a vLLM server that provides an OpenAI-compatible API
# for Llama 3.3 70B on AWS Neuron.
#
# The server will be available at: http://localhost:8000
# OpenAI API endpoint: http://localhost:8000/v1/completions
# Chat endpoint: http://localhost:8000/v1/chat/completions
#
# Configuration:
# - 64K context length (maximum supported)
# - Batch size 1 (single sequence processing)
# - Port 8000 (default)
#
# Usage:
#   ./llama33_online_server.sh
#
# To test the server, use the test_api_client.py script.
#

# Activate the Neuron virtual environment
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# Model configuration
MODEL_PATH="models/Llama-3.3-70B-Instruct/"
MAX_MODEL_LEN=65536  # 64K context
MAX_NUM_SEQS=1       # Batch size
BLOCK_SIZE=128
TENSOR_PARALLEL_SIZE=64
DTYPE="bfloat16"
PORT=8000

echo "Starting Llama 3.3 70B vLLM server..."
echo "  Model: ${MODEL_PATH}"
echo "  Context length: ${MAX_MODEL_LEN}"
echo "  Batch size: ${MAX_NUM_SEQS}"
echo "  Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "  Data type: ${DTYPE}"
echo "  Port: ${PORT}"
echo ""
echo "Server will be available at: http://localhost:${PORT}"
echo "Press Ctrl+C to stop the server"
echo ""

# Start the vLLM OpenAI-compatible API server
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --max-model-len ${MAX_MODEL_LEN} \
    --block-size ${BLOCK_SIZE} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --dtype ${DTYPE} \
    --port ${PORT}
