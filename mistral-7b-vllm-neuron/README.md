# Running Mistral-7B-Instruct-v0.3 on Trainium2 with vLLM + vllm-neuron (Neuron SDK 2.33)

Serve [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) on one
`trn2.3xlarge` with vLLM, on the public **Neuron SDK 2.33 DLAMI**
(`Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0 20261020`). It uses the full 32k context, BF16, with no
NxD Inference.

The DLAMI's stock vllm-neuron (`0.29.0.1.2.0`) has no `MistralForCausalLM`, so engine start fails with
`type object 'MistralForCausalLM' has no attribute 'from_configs'`. This guide installs one public branch on top
of the DLAMI's vLLM environment:

| Branch | What it is |
|---|---|
| [`jimburtoft/vllm-neuron@mistral-trn2-0.29`](https://github.com/jimburtoft/vllm-neuron/tree/mistral-trn2-0.29) | vllm-neuron 0.29.0.1.2.0 (the DLAMI's version) plus: `MistralForCausalLM` on the Llama implementation; GQA decode on the fused NKI kernel; a segmented-prefill KV write that does not transpose the whole KV cache; packed prefill (several prompts per prefill step); and real-peak KV admission |

**The model README on the branch has the details:** serving recipes, every setting, accuracy and performance:
**[vllm_neuron/model/mistral/README.md](https://github.com/jimburtoft/vllm-neuron/blob/mistral-trn2-0.29/vllm_neuron/model/mistral/README.md)**.
This page covers what you need on top of it for the SDK 2.33 DLAMI: the instance, the install, and one
end-to-end run. Every command below was run as written on a fresh `trn2.3xlarge` on 2026-11-10.

Measured on one Trainium2 device (TP=4), BF16, greedy, with the step-4 server (32k context):

| Workload | Result (one device) |
|---|---:|
| One request, 1,024 in / 1,024 out | **113** output tok/s, time to first token **62 ms**, 8.8 ms per token |
| 32 concurrent, 1,024 in / 1,024 out | **2,048** output tok/s (TPOT 14.8 ms) |
| 32 concurrent, 30,720 in / 1,024 out | 182 output tok/s |
| One request with a 30,000-token prompt | time to first token 2.3 s |
| Needle retrieval at 836 / 14.7k / 30.0k prompt tokens | 6 / 6 |
| GSM8K, 1,319 questions, zero-shot CoT, greedy | 59.6 % (Hugging Face transformers BF16: 60.1 %) |

Higher-concurrency configurations reach 3,164 output tok/s (10k context, 1,024 / 1,024) and 4,591 (2k context,
128 / 128); see the model README.

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

## 2. Install the branch into the DLAMI's vLLM environment

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate

pip install --no-deps --force-reinstall "git+https://github.com/jimburtoft/vllm-neuron.git@mistral-trn2-0.29"
rm -rf ~/.cache/neuron_libtorch     # start with a clean compile cache
```

Check that it took effect:

```bash
python -c "
from vllm_neuron.model.registry import get_models
print('Mistral plugin:', 'MistralForCausalLM' in dict(get_models()))" 2>/dev/null
# Mistral plugin: True
```

- **`--force-reinstall` is required.** The branch has the same version number as the DLAMI's plugin
  (`0.29.0.1.2.0`), so a plain `pip install git+...` decides the requirement "is already satisfied" and installs
  nothing.
- **`--no-deps`** keeps the DLAMI's pinned torch, vllm, transformers and nki versions.
- **Always `source .../activate`.** The plugin starts `neuronx-cc` from `PATH`.

Versions in the environment after the install:

| Component | Version |
|---|---|
| vllm | `0.29.0` |
| vllm-neuron | `0.29.0.1.2.0` (branch `mistral-trn2-0.29` @ `26259e6` or later) |
| transformers | `5.17.0` |
| nki | `0.7.0` |
| neuronx-cc | `2.28.7631.0` |

No NxD Inference package is installed or used: `neuronx_distributed_inference` and `neuronx_distributed` are not
in the 2.33 vLLM environment, and the model runs in the plugin's own Llama model code.

---

## 3. Download the model

The model is not gated.

```bash
hf download mistralai/Mistral-7B-Instruct-v0.3 --exclude consolidated.safetensors
```

`--exclude consolidated.safetensors` skips the Mistral-native copy of the weights (14.5 GB) that vLLM does not
use here.

---

## 4. Serve it (full 32k context)

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
export NEURON_SKIP_EFA_AFFINITY=1               # REQUIRED on trn2.3xlarge (no EFA device)
export VLLM_NEURON_SWA_REAL_ADMISSION=1         # admit requests by their real KV size, not max_model_len
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.6

vllm serve mistralai/Mistral-7B-Instruct-v0.3 --tensor-parallel-size 4 \
  --max-model-len 32768 --max-num-batched-tokens 8192 --max-num-seqs 32 --no-enable-prefix-caching \
  --additional-config '{"neuron_config": {
     "num_batched_tokens_buckets": [128, 256, 512, 1024, 2048, 4096, 8192],
     "num_seqs_buckets": [1, 2, 4, 8, 16, 32],
     "decode_context_length_buckets": [2048, 4096, 8192, 16384]}}'
```

**The first start compiles the model graphs**, which took **38 minutes** on a trn2.3xlarge. Run it in `tmux` or
`screen`. Later starts reuse the compile cache (`~/.cache/neuron_libtorch`). The server is ready when the log
shows `Application startup complete`.

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "mistralai/Mistral-7B-Instruct-v0.3",
  "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
  "max_tokens": 32, "temperature": 0}' | python -m json.tool
```

```
"content": " The capital of France is Paris."
```

SDK 2.33 notes for this command:

| Setting | Why |
|---|---|
| `NEURON_SKIP_EFA_AFFINITY=1` | A trn2.3xlarge has no EFA device. Without it the engine fails at start with `No EFA device found at /sys/bus/pci/devices/...`. |
| `VLLM_NEURON_SWA_REAL_ADMISSION=1` | Without it the scheduler charges every request the full `--max-model-len`. At 32k only about 13 requests then run at once, even short ones. With it, admission sums each request's real KV need. (The name comes from the sliding-window models the feature was written for; dense models use it unchanged.) |
| `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.6` | The DLAMI default is 0.30 of the free HBM. 0.6 is the largest value that still leaves room for the 8,192-token prefill graph; 0.7 fails at load with `Allocation Failure`. |
| `--no-enable-prefix-caching` | With prefix caching on, vllm-neuron 0.29 needs `--max-num-batched-tokens` to be one of 512 ... 8192 and below `--max-model-len`. That holds here, so prefix caching can be turned on. The numbers below were measured with it off. |
| `decode_context_length_buckets` | Must not include `--max-model-len` (the implicit top bucket); the server rejects the config otherwise. On this release the buckets switch the engine to Model Runner V1 (the log says `does not yet support decode context-length bucketing; using V1`), which is expected. Without them, every decode step reads the whole 32k window, and 1k/1k throughput drops from 2,000 to 465 tok/s. |
| `--max-num-seqs 32` | At 32k, the cold compile of batch-64/128 decode graphs at long context ran the trn2.3xlarge out of host memory (124 GB RAM + 64 GB swap). For more concurrency, use the 10k recipe in the model README (`--max-num-seqs 128`). |

For shorter-context deployments (10k with 128 concurrent requests, or a 2k server that packs 8 short prompts per
prefill step) use the recipes in the
[model README](https://github.com/jimburtoft/vllm-neuron/blob/mistral-trn2-0.29/vllm_neuron/model/mistral/README.md#serving-trn23xlarge-tp4).

---

## 5. Check it

The scripts below are on this page's folder: [`needle_server.py`](./needle_server.py) and
[`gsm8k_server.py`](./gsm8k_server.py). Run them against the server from step 4.

```bash
pip install datasets    # only for downloading GSM8K
python -c "
from datasets import load_dataset; import json
with open('gsm8k_test.jsonl','w') as f:
    [f.write(json.dumps({'question':r['question'],'answer':r['answer']})+'\n') for r in load_dataset('openai/gsm8k','main',split='test')]"
python needle_server.py mistralai/Mistral-7B-Instruct-v0.3 920,15500,31500 2
python gsm8k_server.py run --data gsm8k_test.jsonl
vllm bench serve --model mistralai/Mistral-7B-Instruct-v0.3 --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 --ignore-eos --num-prompts 64 --max-concurrency 32
```

Output on the run above:

```
NEEDLE len~920 prompt_tokens=836 depth=0.71 ok=True e2e=0.1s out=' 985440'
NEEDLE len~920 prompt_tokens=836 depth=0.31 ok=True e2e=0.1s out=' 541001'
NEEDLE len~15500 prompt_tokens=14738 depth=0.42 ok=True e2e=1.0s out=' 636110'
NEEDLE len~15500 prompt_tokens=14740 depth=0.87 ok=True e2e=1.0s out=' 921872'
NEEDLE len~31500 prompt_tokens=30028 depth=0.81 ok=True e2e=2.3s out=' 475441'
NEEDLE len~31500 prompt_tokens=30028 depth=0.21 ok=True e2e=2.3s out=' 329053'
NEEDLE_TOTAL 6/6
GSM8K run n=1319 acc=0.5959
Output token throughput (tok/s):         2047.86
Median TTFT (ms):                        899.13
Median TPOT (ms):                        14.76
```

---

## Results

One trn2.3xlarge (one Trainium2 device), TP=4, LNC=2, BF16 weights and KV cache, greedy, SDK 2.33 DLAMI,
vllm-neuron branch `mistral-trn2-0.29` installed as in step 2, server exactly as in step 4. `vllm bench serve`,
random prompts, EOS ignored. No requests were preempted.

| Online | Concurrency | Output tok/s | TTFT p50 | TPOT p50 |
|---|---:|---:|---:|---:|
| 1,024 in / 1,024 out | 1 | 113 | 62 ms | 8.8 ms |
| 1,024 / 1,024 | 32 | 2,048 | 899 ms | 14.8 ms |
| 30,720 / 1,024 | 32 | 182 | 95 s (queueing; about 13 such requests fit in the KV cache at once) | 55.5 ms |

Same branch, other measured configurations (from the
[model README](https://github.com/jimburtoft/vllm-neuron/blob/mistral-trn2-0.29/vllm_neuron/model/mistral/README.md)):

| Configuration | Peak output tok/s |
|---|---:|
| 10k context, `--max-num-seqs 128`, 1,024 / 1,024 | 3,164 |
| 10k context, 4,096 / 512 | 993 |
| 32k context, 16,384 / 1,024 | 282 |
| 2k context, 8 prompts packed per prefill step, `--max-num-seqs 256`, 128 / 128 | 4,591 |

| Accuracy | |
|---|---|
| Teacher-forced greedy tokens vs Hugging Face fp32 (32 chat prompts, ~3,500 positions) | 99.4 % match (Hugging Face's own BF16 model: 99.35 %) |
| GSM8K, 1,319 questions | 59.6 % (Hugging Face transformers BF16: 60.1 %) |
| Needle retrieval up to 30k prompt tokens | 6 / 6 |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `type object 'MistralForCausalLM' has no attribute 'from_configs'` | The stock vllm-neuron is still installed. Reinstall with `--force-reinstall` (step 2). |
| `No EFA device found at /sys/bus/pci/devices/...` | `export NEURON_SKIP_EFA_AFFINITY=1`. |
| `decode_context_length_buckets[...]=32768 must be strictly less than max_model_len=32768` | Drop `--max-model-len` from the bucket list. |
| `Automatic Prefix Caching (APC) requires segmented prefill` | `--max-num-batched-tokens` must be below `--max-model-len` and in {512 ... 8192}, or add `--no-enable-prefix-caching`. |
| `Model warmup failed for prefill bucket 8192 ... Allocation Failure` | `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` above 0.6; lower it. |
| `NotImplementedError: MistralForCausalLM with sliding_window=4096 is not supported` | Mistral-7B v0.1 uses a sliding window, which this branch does not implement. Use v0.3. |
| Compiler error with `Killed` / `F137` | Add swap (step 1). |
| Engine start is very slow the first time | Expected: the cold compile. Later starts use `~/.cache/neuron_libtorch`. |

## References

- Model: <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3>
- vllm-neuron branch: <https://github.com/jimburtoft/vllm-neuron/tree/mistral-trn2-0.29>
- Model README (settings, accuracy, performance): <https://github.com/jimburtoft/vllm-neuron/blob/mistral-trn2-0.29/vllm_neuron/model/mistral/README.md>
- vLLM Neuron documentation: <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/>
