#!/bin/bash
# upstream vLLM on B200. ROLE=producer|consumer|none
set -u
. ~/venv/bin/activate
ROLE=${ROLE:-consumer}; MODEL=${MODEL:-$HOME/models/Llama-3.2-1B-Instruct}; PORT=${PORT:-8200}
SIDE=${SIDE:-5659}; GPU=${GPU:-0}; TP=${TP:-1}; MML=${MML:-2048}
export CUDA_VISIBLE_DEVICES=$GPU VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE VLLM_NIXL_SIDE_CHANNEL_HOST=$(hostname -I | awk '{print $1}')
export VLLM_KV_CACHE_LAYOUT=${LAYOUT:-HND}
KV=()
if [ "$ROLE" != none ]; then
  KV=(--kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_$ROLE\",\"kv_buffer_device\":\"cuda\",\"kv_connector_extra_config\":{\"backends\":[\"LIBFABRIC\"],\"enforce_handshake_compat\":${ENFORCE:-true}}}")
fi
exec vllm serve "$MODEL" --served-model-name llama --port $PORT --tensor-parallel-size $TP \
  --dtype bfloat16 --max-model-len $MML --max-num-seqs 4 --block-size 32 --no-enable-prefix-caching \
  --gpu-memory-utilization 0.5 ${EXTRA:-} "${KV[@]}"
