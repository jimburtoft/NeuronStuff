#!/bin/bash
# vllm-neuron on trn2. ROLE=producer|consumer|none  MODEL=<path>  PORT  SIDE  CORES  TP
set -u
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin/activate
ROLE=${ROLE:-producer}; MODEL=${MODEL:-$HOME/models/Llama-3.2-1B-Instruct}; PORT=${PORT:-8100}
SIDE=${SIDE:-5559}; CORES=${CORES:-0-3}; TP=${TP:-4}; MML=${MML:-2048}
export NEURON_VISIBLE_DEVICES=$CORES VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE VLLM_NIXL_SIDE_CHANNEL_HOST=$(hostname -I | awk '{print $1}')
export FI_EFA_ENABLE_SHM_TRANSFER=0 NEURON_RT_MAP_HBM=1
KV=()
if [ "$ROLE" != none ]; then
  KV=(--kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_$ROLE\",\"kv_buffer_device\":\"cuda\",\"kv_connector_extra_config\":{\"backends\":[\"LIBFABRIC\"],\"enforce_handshake_compat\":${ENFORCE:-true}}}")
fi
exec vllm serve "$MODEL" --served-model-name llama --port $PORT --tensor-parallel-size $TP \
  --dtype bfloat16 --max-model-len $MML --max-num-seqs 4 --max-num-batched-tokens $MML \
  --block-size 32 --no-enable-prefix-caching "${KV[@]}" \
  --additional-config "{\"neuron_config\":{\"on_device_sampling_config\":{\"all_greedy\":\"true\"},\"num_batched_tokens_buckets\":[$MML],\"kv_segment_size_buckets\":[$MML],\"num_seqs_buckets\":[4]}}"
