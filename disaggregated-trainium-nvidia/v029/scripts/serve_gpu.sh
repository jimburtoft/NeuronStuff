#!/bin/bash
# vLLM 0.29 on an NVIDIA GPU host, FlashInfer with split K/V head slots (both v029 patches applied).
# ROLE=producer|consumer|none MODEL PORT SIDE GPU TP MML BS
set -u
ROLE=${ROLE:-consumer}; MODEL=${MODEL:?}; PORT=${PORT:-8200}; SIDE=${SIDE:-5659}
GPU=${GPU:-0}; TP=${TP:-1}; MML=${MML:-8448}; BS=${BS:-32}
export CUDA_VISIBLE_DEVICES=$GPU VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE VLLM_NIXL_SIDE_CHANNEL_HOST=$(hostname -I | awk '{print $1}')
export VLLM_KV_CACHE_LAYOUT=LBHNC VLLM_FLASHINFER_SPLIT_KV_SLOTS=1 FI_EFA_ENABLE_SHM_TRANSFER=0
KV=()
if [ "$ROLE" != none ]; then
  KV=(--kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_$ROLE\",\"kv_buffer_device\":\"cuda\",\"kv_connector_extra_config\":{\"backends\":[\"LIBFABRIC\"],\"enforce_handshake_compat\":false}}")
fi
exec vllm serve "$MODEL" --served-model-name llama --port $PORT --tensor-parallel-size $TP \
  --dtype bfloat16 --max-model-len $MML --block-size $BS --no-enable-prefix-caching \
  --attention-backend FLASHINFER --gpu-memory-utilization ${GMU:-0.90} ${EXTRA:-} "${KV[@]}"
