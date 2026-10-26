#!/bin/bash
# vllm-neuron 0.29 (Neuron SDK 2.33 DLAMI vLLM venv, remote-strides patch applied) on trn2.48xlarge.
# ROLE=producer|consumer|none MODEL PORT SIDE CORES TP MML BS MNS GMU
set -u
. ${VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0}/bin/activate
ROLE=${ROLE:-producer}; MODEL=${MODEL:?}; PORT=${PORT:-8100}; SIDE=${SIDE:-5559}
CORES=${CORES:-0-3}; TP=${TP:-4}; MML=${MML:-8448}; BS=${BS:-32}; MNS=${MNS:-4}; GMU=${GMU:-0.92}
export NEURON_VISIBLE_DEVICES=$CORES VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE VLLM_NIXL_SIDE_CHANNEL_HOST=$(hostname -I | awk '{print $1}')
export FI_EFA_ENABLE_SHM_TRANSFER=0 NEURON_RT_MAP_HBM=1
KV=()
if [ "$ROLE" != none ]; then
  KV=(--kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_$ROLE\",\"kv_buffer_device\":\"cuda\",\"kv_connector_extra_config\":{\"backends\":[\"LIBFABRIC\"],\"enforce_handshake_compat\":false}}")
fi
# Prefill buckets: one 8192-token bucket plus a 2048 bucket; num_seqs buckets must end at --max-num-seqs.
NC=${NC:-"{\"num_batched_tokens_buckets\":[2048,8192],\"num_seqs_buckets\":[1,$MNS],\"on_device_sampling_config\":{\"all_greedy\":true}}"}
exec vllm serve "$MODEL" --served-model-name llama --port $PORT --tensor-parallel-size $TP \
  --dtype bfloat16 --max-model-len $MML --max-num-seqs $MNS --max-num-batched-tokens 8192 \
  --block-size $BS --no-enable-prefix-caching --gpu-memory-utilization $GMU "${KV[@]}" \
  --additional-config "{\"neuron_config\":$NC}"
