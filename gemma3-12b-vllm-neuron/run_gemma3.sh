#!/bin/bash
# Gemma 3 12B on one Trainium2 device: everything in README.md sections 2-5, in order.
# Run on a fresh trn2.3xlarge with the SDK 2.33 DLAMI, after the swap file (section 1) and `hf auth login`.
#   bash run_gemma3.sh            # install, offline 2k + 4k, online 2k
#   STEPS="offline-2k" bash run_gemma3.sh
set -euo pipefail
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate

# Section 2: install the two branches
pip install --no-deps --force-reinstall "git+https://github.com/jimburtoft/vllm-neuron.git@gemma3-trn2-perf-0.29"
pip install --no-deps --force-reinstall "git+https://github.com/jimburtoft/nki-library.git@gemma3-trn2-perf-2.33"
rm -rf ~/.cache/neuron_libtorch
python -c "
import importlib
from vllm_neuron.model.registry import get_models
abt = importlib.import_module('nkilib.experimental.transformer.attention_block_tkg')
assert 'Gemma3ForConditionalGeneration' in dict(get_models()), 'Gemma 3 plugin not installed'
assert getattr(abt, 'SUPPORTS_D256_OUT_PROJ', False) and getattr(abt, 'SUPPORTS_D256_BLOCK_KV_UPDATE', False), 'NKI Library branch not installed'
print('branches OK')" 2>/dev/null

# Section 3: the model (gated; needs `hf auth login` first)
hf download google/gemma-3-12b-it --exclude "*.gguf"

# Section 5: the benchmark
[ -d ~/vllm-neuron-gemma3 ] || git clone -b gemma3-trn2-perf-0.29 https://github.com/jimburtoft/vllm-neuron.git ~/vllm-neuron-gemma3
cd ~/vllm-neuron-gemma3
for s in ${STEPS:-offline-2k offline-4k online-2k}; do
  echo "== $s $(date -u +%FT%TZ)"
  bash tools/gemma3/run.sh "$s"
done
grep -h '^POINT' out/offline_*.log
