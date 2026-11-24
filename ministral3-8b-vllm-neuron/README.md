# Running Ministral-3-8B-Instruct-2512 on Trainium2 with vLLM + vllm-neuron (Neuron SDK 2.33)

Serve [`mistralai/Ministral-3-8B-Instruct-2512-BF16`](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16)
on one `trn2.3xlarge` with vLLM, on the public **Neuron SDK 2.33 DLAMI**
(`Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0 20261020`). Text and images (Pixtral vision tower), BF16 or
the FP8 checkpoint, up to 32k context, optional EAGLE-3 speculative decoding. No NxD Inference.

The DLAMI's stock vllm-neuron (`0.29.0.1.2.0`) has no `Mistral3ForConditionalGeneration`. This guide installs two
public branches on top of the DLAMI's vLLM environment:

| Branch | What it is |
|---|---|
| [`jimburtoft/vllm-neuron@ministral3-trn2-0.29`](https://github.com/jimburtoft/vllm-neuron/tree/ministral3-trn2-0.29) | vllm-neuron 0.29.0.1.2.0 plus Ministral 3 on the plugin's Llama code: YaRN RoPE, Llama-4 query scaling past 16k on the fused decode kernel, the Pixtral vision tower, the static-FP8 checkpoint, packed prefill, real-peak KV admission, and an EAGLE-3 draft converter |
| [`jimburtoft/nki-library@ministral3/unified-2.33`](https://github.com/jimburtoft/nki-library/tree/ministral3/unified-2.33) | the SDK 2.33 NKI Library plus faster decode-attention kernels: batched block-KV decode, a single-token QKV layout, the per-token query scale for long context, and an unpacked-FP8 K load. It overlays the DLAMI's nkilib through `PYTHONPATH` |

**The model README on the branch has the details:** every serving recipe, setting, accuracy check and benchmark:
**[vllm_neuron/model/ministral3/README.md](https://github.com/jimburtoft/vllm-neuron/blob/ministral3-trn2-0.29/vllm_neuron/model/ministral3/README.md)**.
This page covers the instance, the install and one end-to-end run. Every command below was run as written on a
fresh `trn2.3xlarge` on 2026-11-24.

Measured on one Trainium2 device (TP=4) with the step-4 server (BF16 weights, packed FP8 KV cache, 2k context):

| Workload | Result (one device) |
|---|---:|
| One request, 1,024 in / 1,024 out | **109** output tok/s, time to first token **126 ms**, 9.0 ms per token |
| 64 concurrent, 1,024 in / 1,024 out | **2,583** output tok/s (TPOT 21.0 ms) |
| One request with EAGLE-3 (step 6), GSM8K prompts, 256 out | **196** output tok/s, **4.6 ms** per token (2.66 tokens per step) |
| GSM8K, 1,319 questions, zero-shot CoT, greedy | 88.9 % (BF16 + packed FP8 KV) |

Higher-throughput layouts reach 5,151 output tok/s (TP=2 x DP=2, 128 / 128) and 4,415 (FP8 checkpoint, packed FP8
KV, 1,024 / 1,024); see the model README.

---

## 1. Launch the instance

| Setting | Value |
|---|---|
| Instance type | `trn2.3xlarge` (1 Trainium2 device, 4 logical NeuronCores at LNC=2) |
| AMI name | **`Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0 20261020`** |
| Neuron SDK | **2.33** |
| Disk | **300 GB gp3** minimum |

`trn2` instances need a **Capacity Block**.

AMI IDs:

| Region | AMI ID |
|---|---|
| us-east-1 (N. Virginia) | `ami-091af6c736073d840` |
| us-east-2 (Ohio) | `ami-02eaa082292ed0cf9` |
| us-west-2 (Oregon) | `ami-0188c2226a35a50e8` |
| sa-east-1 (São Paulo) | `ami-017da676c0eb9337d` |
| ap-southeast-4 (Melbourne) | `ami-042081cf4060ec468` |

Any other region:

```bash
aws ec2 describe-images --owners amazon --region <REGION> \
  --filters "Name=name,Values=Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0 20261020" \
  --query 'Images[*].[ImageId,Name]' --output table
```

### Add swap first

The DLAMI ships with **no swap**, and `neuronx-cc` can use a lot of host memory while it compiles. When it runs
out of memory, the error blames the compiler, not memory.

```bash
sudo dd if=/dev/zero of=/swapfile bs=1M count=65536 status=none
sudo chmod 600 /swapfile
sudo mkswap --force /swapfile     # --force is REQUIRED on this kernel; without it mkswap silently does nothing
sudo swapon /swapfile
free -g                            # confirm ~63 GB swap
```

---

## 2. Get the code and set up the environment

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
cd ~
git clone -b ministral3-trn2-0.29 https://github.com/jimburtoft/vllm-neuron.git
git clone -b ministral3/unified-2.33 https://github.com/jimburtoft/nki-library.git
cd ~/vllm-neuron
source examples/vllm_neuron/models/ministral3/setup_optimized.sh ~/nki-library
```

`setup_optimized.sh` puts both checkouts first on `PYTHONPATH` (nothing is pip-installed, so the DLAMI's pinned
torch, vllm, transformers and nki stay as they are) and sets:

| Variable | Why |
|---|---|
| `VLLM_NEURON_FUSE_DECODE_NORMS=1` | Runs both RMSNorms inside the decode attention and MLP kernels (decode -8 %). |
| `VLLM_NEURON_SWA_REAL_ADMISSION=1` | Admits requests by their real KV size instead of `--max-model-len`. |
| `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.6` | KV cache share of free HBM (DLAMI default 0.30). |
| `NKI_TRACE_CACHE_URL`, `NKI_COMPILE_CACHE_URL`, `VLLM_CACHE_ROOT` | Per-nkilib-revision caches under `~/.cache/ministral3/<rev>`; the NKI trace cache does not hash nkilib source. |

It ends by printing the nkilib path. Check that the plugin sees the model:

```bash
python -c "
from vllm_neuron.model.registry import get_models
print('Ministral 3 plugin:', 'Mistral3ForConditionalGeneration' in dict(get_models()))" 2>/dev/null
# Ministral 3 plugin: True
```

**Source the script in every new shell** before `vllm serve`: the branches only take effect through `PYTHONPATH`.

Versions in the environment (unchanged DLAMI packages):

| Component | Version |
|---|---|
| vllm | `0.29.0` |
| vllm-neuron | `0.29.0.1.2.0` (branch `ministral3-trn2-0.29` @ `f0e1a3a` or later, from `PYTHONPATH`) |
| nkilib | branch `ministral3/unified-2.33` @ `fe8b59b` (from `PYTHONPATH`) |
| transformers | `5.17.0` |
| mistral_common | `1.12.0` |
| nki | `0.7.0` |
| neuronx-cc | `2.28.7631.0` |

---

## 3. Download the model

The model is not gated.

```bash
hf download mistralai/Ministral-3-8B-Instruct-2512-BF16 --exclude consolidated.safetensors
```

`--exclude consolidated.safetensors` skips the Mistral-native copy of the weights that vLLM does not use here.

---

## 4. Serve it

BF16 weights with the packed FP8 KV cache (FP8 KV keeps BF16 accuracy here and halves KV memory and traffic):

```bash
export NEURON_SKIP_EFA_AFFINITY=1    # REQUIRED on trn2.3xlarge (no EFA device)
export UNSAFE_FP8FNCAST=1            # REQUIRED for the packed FP8 KV cache

vllm serve mistralai/Ministral-3-8B-Instruct-2512-BF16 --tensor-parallel-size 4 \
  --tokenizer-mode mistral --language-model-only \
  --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 64 --no-enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --additional-config '{"neuron_config": {"fp8_packed_kv": true,
     "num_batched_tokens_buckets": [128, 256, 512, 1024, 2048],
     "num_seqs_buckets": [1, 2, 4, 8, 16, 32, 64],
     "decode_context_length_buckets": [512, 1024],
     "on_device_sampling_config": {"max_top_k": 32}}}'
```

**The first start compiles the model graphs**, which took **19 minutes** on a trn2.3xlarge. Run it in `tmux` or
`screen`. Later starts reuse the compile cache. The server is ready when the log shows
`Application startup complete`.

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
  "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
  "max_tokens": 32, "temperature": 0}' | python -m json.tool
```

```
"content": "The capital of France is **Paris**."
```

SDK 2.33 notes for this command:

| Setting | Why |
|---|---|
| `--tokenizer-mode mistral` | The repository has no Jinja chat template for the Hugging Face tokenizer; the tekken tokenizer gives the same chat token ids as transformers' `apply_chat_template`. |
| `--language-model-only` | Text decoder only. For images, see "Images" in the model README. |
| `NEURON_SKIP_EFA_AFFINITY=1` | A trn2.3xlarge has no EFA device; without it the engine fails at start with `No EFA device found`. |
| `--kv-cache-dtype fp8_e4m3` + `"fp8_packed_kv": true` | The packed FP8 layout is the fastest KV layout from 1k contexts up. GSM8K stays at the BF16 level. Without `fp8_packed_kv` (unpacked FP8 KV), segmented prefill is off and `--max-num-batched-tokens` must be at least `--max-model-len`. |
| `decode_context_length_buckets` | Must stay below `--max-model-len`. On this release the buckets switch the engine to Model Runner V1 (the log says so), which is expected. |
| `"max_top_k": 32` | Sampling supports `top_k` <= 32; 3 % faster than the default 256 with the 131k vocabulary. |

Other recipes (10k context with 128 concurrent requests, TP=2 x DP=2 for short prompts, 32k context, images,
the FP8 checkpoint) are in the
[model README](https://github.com/jimburtoft/vllm-neuron/blob/ministral3-trn2-0.29/vllm_neuron/model/ministral3/README.md).

---

## 5. Check it

[`gsm8k_server.py`](./gsm8k_server.py) runs GSM8K against the server from step 4:

```bash
pip install datasets    # only for downloading GSM8K
python -c "
from datasets import load_dataset; import json
with open('gsm8k_test.jsonl','w') as f:
    [f.write(json.dumps({'question':r['question'],'answer':r['answer']})+'\n') for r in load_dataset('openai/gsm8k','main',split='test')]"
python gsm8k_server.py run --data gsm8k_test.jsonl
vllm bench serve --model mistralai/Ministral-3-8B-Instruct-2512-BF16 --tokenizer-mode mistral --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 --ignore-eos --num-prompts 128 --max-concurrency 64
```

Output on the run above:

```
GSM8K run n=1319 acc=0.8893
Output token throughput (tok/s):         2583.05
Median TTFT (ms):                        3896.95
Median TPOT (ms):                        20.98
```

With `--num-prompts 4 --max-concurrency 1`: 109.5 tok/s, TTFT 126 ms, TPOT 9.0 ms.

---

## 6. Optional: EAGLE-3 speculative decoding (lower latency at low concurrency)

A public EAGLE-3 draft head for this model,
[`flaitenberger/economize-eagle3-Ministral-3-8B-Instruct-2512`](https://huggingface.co/flaitenberger/economize-eagle3-Ministral-3-8B-Instruct-2512)
(Apache-2.0), is published in the economize format. Convert it once to the vLLM EAGLE-3 layout. The plugin's draft
loader reads the token embedding from the draft folder, so the converter copies it from the target:

```bash
hf download flaitenberger/economize-eagle3-Ministral-3-8B-Instruct-2512 --local-dir ~/economize-draft
BF16_DIR=$(hf download mistralai/Ministral-3-8B-Instruct-2512-BF16 --exclude consolidated.safetensors --quiet)
python ~/vllm-neuron/vllm_neuron/model/ministral3/convert_economize_eagle3.py ~/economize-draft \
  $BF16_DIR/config.json ~/m3-eagle3 $BF16_DIR
# aux [2, 17, 31] heads 32 8 128 draft_vocab 32768 tensors 16

vllm serve mistralai/Ministral-3-8B-Instruct-2512-BF16 --tensor-parallel-size 4 \
  --tokenizer-mode mistral --language-model-only \
  --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 8 --no-enable-prefix-caching \
  --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [2048], "num_seqs_buckets": [1, 4, 8],
     "decode_context_length_buckets": [1024], "on_device_sampling_config": {"all_greedy": "true"}}}' \
  --speculative-config '{"method": "eagle3", "model": "/home/ubuntu/m3-eagle3", "num_speculative_tokens": 3}'
```

The first start compiled in 14 minutes. One request on 32 GSM8K prompts (256 output tokens, greedy) on this run:
**196 output tok/s, 4.6 ms per token, 2.66 tokens per step**.

Output tok/s relative to the same server without the draft (TP=4, BF16, 256 output tokens, greedy; model README):

| Prompts | Concurrency 1 | 4 | 8 |
|---|---|---|---|
| GSM8K | 1.89x | 1.48x | 1.31x |
| Code | 1.71x | 1.32x | 1.14x |
| Chat (UltraChat) | 1.40x | 1.09x | 0.94x |

With the **FP8 checkpoint plus packed FP8 KV** the draft stacks on the faster target: 195 / 523 / 728 tok/s on GSM8K
at concurrency 1 / 4 / 8 (FP8 alone: 125 / 316 / 473). See the model README for the FP8 conversion and settings.

GSM8K is unchanged by the draft (89.1-90.0 % across configurations). Speculation helps latency at low concurrency;
keep `--max-num-seqs` at 8 or less, and serve without it for maximum throughput.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Model architectures ['PixtralForConditionalGeneration'] failed to be inspected` / `cannot import name 'PixtralRotaryEmbedding'` | The branch is older than `f0e1a3a` (it then needs `--config-format hf --load-format safetensors`), or `--config-format mistral` was passed: pull the branch, or drop the flag. |
| Engine start rejects the `Mistral3ForConditionalGeneration` architecture, or the plugin check in step 2 prints `False` | The branches are not on `PYTHONPATH`: `source examples/vllm_neuron/models/ministral3/setup_optimized.sh ~/nki-library` in this shell (step 2). |
| `ChatTemplateResolutionError` | Add `--tokenizer-mode mistral`. |
| `No EFA device found at /sys/bus/pci/devices/...` | `export NEURON_SKIP_EFA_AFFINITY=1`. |
| `decode_context_length_buckets[...] must be strictly less than max_model_len` | Drop `--max-model-len` from the bucket list. |
| `Chunked prefill ... requires segmented prefill ... cannot read a non-packed FP8` | Unpacked FP8 KV: add `"fp8_packed_kv": true`, or set `--max-num-batched-tokens` >= `--max-model-len`. |
| `Model warmup failed ... Allocation Failure` | `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` too high for the layout; lower it (the model README lists the caps that passed). |
| Compiler error with `Killed` / `F137` | Add swap (step 1). |
| Engine start is very slow the first time | Expected: the cold compile. Later starts use the caches under `~/.cache/ministral3/`. |

## References

- Model: <https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16> (FP8: <https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512>)
- vllm-neuron branch: <https://github.com/jimburtoft/vllm-neuron/tree/ministral3-trn2-0.29>
- NKI Library branch: <https://github.com/jimburtoft/nki-library/tree/ministral3/unified-2.33>
- Model README (settings, accuracy, performance, images, FP8, long context): <https://github.com/jimburtoft/vllm-neuron/blob/ministral3-trn2-0.29/vllm_neuron/model/ministral3/README.md>
- EAGLE-3 draft: <https://huggingface.co/flaitenberger/economize-eagle3-Ministral-3-8B-Instruct-2512>
- vLLM Neuron documentation: <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/>
