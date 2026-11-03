# Running Mellum2.1 12B-A2.5B Thinking on Trainium2 with vLLM + vllm-neuron (Neuron SDK 2.33)

Serve [`JetBrains/Mellum2.1-12B-A2.5B-Thinking`](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking)
(a 12B mixture-of-experts reasoning model with 2.5B active parameters) on one `trn2.3xlarge` with vLLM, on the
public **Neuron SDK 2.33 DLAMI** (`Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0 20261020`). Every command
below was run as written on a `trn2.3xlarge`, starting from the stock DLAMI environment, on 2026-11-03.

The DLAMI's stock vllm-neuron does not support Mellum. This guide installs one public branch on top of the
DLAMI's vLLM environment:

| Branch | What it is |
|---|---|
| [`jimburtoft/vllm-neuron@mellum-trn2-0.29`](https://github.com/jimburtoft/vllm-neuron/tree/mellum-trn2-0.29) | vllm-neuron 0.29.0.1.2.0 (the DLAMI's version) plus the Mellum model (sliding-window + full attention, YaRN RoPE, 64-expert MoE), sliding-window KV admission and ring, and a fix for vLLM's Mellum config |

The DLAMI's own NKI Library is used as is.

Measured on one Trainium2 device (TP=4, expert parallel 4), BF16, greedy, no speculative decoding (details in
[Results](#results)):

| Workload | Output tokens/s (one device) |
|---|---:|
| Offline, 64 requests at once, ~70-token prompts, 1,024 output tokens | **1,975** |
| Online (`vllm bench serve`), 32 concurrent requests, 1,024 in / 1,024 out | **1,405** (TPOT 20.0 ms) |
| Offline, one request | **103** (9.7 ms per token) |

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

pip install --no-deps --force-reinstall "git+https://github.com/jimburtoft/vllm-neuron.git@mellum-trn2-0.29"
rm -rf ~/.cache/neuron_libtorch     # start with a clean compile cache

git clone -b mellum-trn2-0.29 https://github.com/jimburtoft/vllm-neuron.git ~/vllm-neuron   # run scripts + references
```

Check that it took effect:

```bash
python -c "
from vllm_neuron.model.registry import get_models
print('Mellum plugin:', 'MellumForCausalLM' in dict(get_models()))" 2>/dev/null
# Mellum plugin: True
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
| vllm-neuron | `0.29.0.1.2.0` (branch `mellum-trn2-0.29` @ `159f307`) |
| transformers | `5.17.0` (has `MellumConfig`) |
| nki | `0.7.0` |
| neuronx-cc | `2.28.7631.0` |
| torch | `2.13.0` |

---

## 3. Download the model

The model is not gated.

```bash
hf download JetBrains/Mellum2.1-12B-A2.5B-Thinking
```

---

## 4. Serve it

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
source ~/vllm-neuron/tools/mellum/env.sh

vllm serve JetBrains/Mellum2.1-12B-A2.5B-Thinking \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser hermes
```

**The first start compiles the model graphs**, which takes a while on a trn2.3xlarge. Run it in `tmux` or
`screen`. Later starts reuse the compile cache (`~/.cache/neuron_libtorch`). The server is ready when the log
shows `Application startup complete`.

Query it with JetBrains' recommended sampling settings:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "JetBrains/Mellum2.1-12B-A2.5B-Thinking",
  "messages": [{"role": "user", "content": "Find the bug in this function and explain the fix: def mean(xs): return sum(xs) / len(xs) - 1"}],
  "max_tokens": 2048, "temperature": 0.6, "top_p": 0.95, "top_k": 20}' | python -m json.tool
```

The thinking trace comes back in `message.reasoning`, and the answer in `message.content`:

````
The function subtracts `1` from the result of `sum(xs) / len(xs)`. This changes the intended calculation of the
mean; the mean should be just `sum(xs) / len(xs)`.
**Fix:**
Remove the `- 1` so the function returns the correct mean:
```python
def mean(xs):
    return sum(xs) / len(xs)
```
````

Tool calls work with the `hermes` parser:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "JetBrains/Mellum2.1-12B-A2.5B-Thinking",
  "messages": [{"role": "user", "content": "What is the weather in Paris right now?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
  "max_tokens": 1024}' | python -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['tool_calls'])"
# [{'id': 'chatcmpl-tool-...', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"city": "Paris"}'}}]
```

What the settings do:

| Setting | Why |
|---|---|
| `--tensor-parallel-size 4` | Uses all 4 cores of the device (8 query heads and 1 KV head per core). |
| `--enable-expert-parallel` | 16 of the 64 experts per core, at full width. This is faster than splitting every expert 4 ways, and lets the prefill expert kernel run. |
| `VLLM_NEURON_SWA_ALIGN_TOKENS=1024` (in `env.sh`) | 21 of the 28 layers use a 1,024-token sliding window. The decode attention kernel needs that window's block table to be a multiple of 256 tokens at LNC=2. |
| `VLLM_NEURON_SWA_RING=1`, `VLLM_NEURON_SWA_REAL_ADMISSION=1`, `VLLM_NEURON_SWA_EARLY_FREE=1` (in `env.sh`) | Sliding-window layers keep a fixed ring of window blocks per request, and requests are admitted by their real KV peak instead of a full-attention worst case. The ring needs single-shot prefill (`--max-num-batched-tokens` >= `--max-model-len`) and no prefix caching. |
| `--max-model-len 8192` | Prompt plus output. Mellum2.1 thinks before it answers (GSM8K-style questions: ~550 tokens on average, a few thousand for hard ones). Raise it for long agentic contexts. |
| `--reasoning-parser qwen3` | Splits `<think>...</think>` into `reasoning`. |

### Prefix caching

Prefix caching needs segmented prefill, which the ring does not support. Turn the ring off and chunk prefill:

```bash
export VLLM_NEURON_SWA_RING=0
vllm serve JetBrains/Mellum2.1-12B-A2.5B-Thinking --tensor-parallel-size 4 --enable-expert-parallel \
  --max-model-len 8192 --max-num-seqs 32 --max-num-batched-tokens 1024 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes
```

This configuration passed the same accuracy gate in three repeated offline runs with prefix-cache hits (same first divergence and gap in each run; later tokens can differ between runs, as is normal for bf16).

---

## 5. Reproduce the results

### Accuracy

```bash
cd ~/vllm-neuron && source tools/mellum/env.sh
python tools/mellum/fp32_gate.py --model JetBrains/Mellum2.1-12B-A2.5B-Thinking --enable-expert-parallel
```

The gate generates greedily for 7 prompts (5 short ones to EOS, and two ~3,500 / ~2,200-token code-review
prompts with 1,024 output tokens, which cross the sliding window) and compares them with an fp32 Hugging Face
reference on CPU. At the first token where they differ, the fp32 log-prob gap between the reference token and
the Neuron token must be below 0.35, and the Neuron token must be in the fp32 top 5. That marks the divergence as
a bf16 near-tie. For scale: Hugging Face's own bf16 model on CPU reaches a gap of 0.49 on one of these prompts.

```
== short prompts
prompt 0: no divergence over 337 tokens
prompt 1: first divergence at token 5, fp32 gap 0.2286, target in fp32 top-5 True -> PASS
prompt 2: first divergence at token 6, fp32 gap 0.0161, target in fp32 top-5 True -> PASS
prompt 3: first divergence at token 41, fp32 gap 0.0135, target in fp32 top-5 True -> PASS
prompt 4: first divergence at token 68, fp32 gap 0.0053, target in fp32 top-5 True -> PASS
GATE PASS
== long prompts
prompt 0: first divergence at token 159, fp32 gap 0.1924, target in fp32 top-5 True -> PASS
prompt 1: first divergence at token 84, fp32 gap 0.1175, target in fp32 top-5 True -> PASS
GATE PASS
```

GSM8K (first 100 test questions, greedy, thinking on):

```bash
pip install datasets
python tools/mellum/gsm8k_eval.py --backend vllm --model JetBrains/Mellum2.1-12B-A2.5B-Thinking --ep \
  --n 100 --max-tokens 4096 --max-model-len 4608 --mnbt 4608 --out gsm8k.json
# GSM8K backend=vllm n=100 acc=0.970 ...
```

### Throughput

```bash
python tools/mellum/bench.py JetBrains/Mellum2.1-12B-A2.5B-Thinking --ep --n 128 --max-num-seqs 64 \
  --fixed 1024 --max-model-len 2048
# BENCH rep 1: 131072 out tok / 66.3s = 1977.8 out tok/s ...
```

Online, against the server from step 4:

```bash
vllm bench serve --model JetBrains/Mellum2.1-12B-A2.5B-Thinking --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 --num-prompts 64 --max-concurrency 32 --ignore-eos
```

---

## Results

One trn2.3xlarge (one Trainium2 device), TP=4, expert parallel 4, LNC=2, BF16 weights and KV cache, greedy,
SDK 2.33 DLAMI, vllm-neuron branch `mellum-trn2-0.29` @ `159f307`. Throughput numbers are the second of two
runs after warm-up.

| Offline (`bench.py`, ~70-token GSM8K prompts, 1,024 output tokens, EOS ignored) | Output tok/s |
|---|---:|
| 1 request | 102.6 |
| 32 requests at once | 1,743 |
| 64 requests at once | 1,975 |

| Online (`vllm bench serve`, random 1,024 in / 1,024 out, 64 prompts) | |
|---|---:|
| Concurrency 32: output tok/s | 1,405 |
| Median TTFT | 2,877 ms |
| Median / P99 TPOT | 20.0 / 22.6 ms |

Without `--enable-expert-parallel` (every core holds all 64 experts at 1/4 width), short-prompt decode is faster
(1 request: 163.9 tok/s; 64 requests: 2,111 tok/s), but prefill falls back to a torch MoE, and online with
1,024-token prompts drops to 615 tok/s (TPOT 50 ms) because prefills stall decode. Use expert parallelism unless
your prompts are short.

| Accuracy | |
|---|---|
| fp32 first-divergence gate (7 prompts) | PASS, max gap 0.229 |
| GSM8K, 100 questions | 97 % |

---

## Notes

- **Why the stock DLAMI fails.** vLLM 0.29 knows `MellumForCausalLM`, but its `MellumConfig` subclasses
  `Qwen3MoeConfig`, which adds a flat `rope_theta` next to Mellum's per-layer-type RoPE settings. Engine start then
  fails in `validate_rope` with `AttributeError: 'float' object has no attribute 'get'`. Past that, the stock
  plugin has no Mellum model. The branch uses transformers' `MellumConfig` and adds the model.
- **MoE.** The router runs in fp32 (softmax over 64 experts, top 8, renormalized). In decode, only the routed
  experts' weights are read while that is fewer than the experts on the core, then every local expert once.
  Prefill uses the NKI Library's blockwise expert kernel. The experts' hidden size (2,304) is zero-padded to
  2,560 inside the MoE block, because the kernel works in tiles of 512.
- **RoPE.** Sliding-window layers use plain RoPE. The 7 full-attention layers use YaRN (factor 16 over an 8,192
  original context, attention factor 1.277), as in transformers.
- **Not supported yet:** MTP speculative decoding (the checkpoint has no MTP head), FP8 weights or KV cache.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `AttributeError: 'float' object has no attribute 'get'` in `validate_rope` | The stock vllm-neuron is still installed. Reinstall with `--force-reinstall` (step 2). |
| `NCC_INKI016 ... Number of blocks per batch must be a multiple of (sprior_n_prgs * p_max)` | `source tools/mellum/env.sh` (sets `VLLM_NEURON_SWA_ALIGN_TOKENS=1024`). |
| `VLLM_NEURON_SWA_RING=1 does not support segmented prefill` | Set `--max-num-batched-tokens` to `--max-model-len`, or `export VLLM_NEURON_SWA_RING=0`. |
| `No EFA device found ...` | `export NEURON_SKIP_EFA_AFFINITY=1` (in `env.sh`; the trn2.3xlarge has no EFA). |
| `Automatic Prefix Caching (APC) requires segmented prefill` | Add `--no-enable-prefix-caching`, or see [Prefix caching](#prefix-caching). |
| Compiler error with `Killed` / `F137` | Add swap (step 1). |

## References

- Model: <https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking>
- vllm-neuron branch: <https://github.com/jimburtoft/vllm-neuron/tree/mellum-trn2-0.29> (model code in
  `vllm_neuron/model/mellum/`, run scripts in `tools/mellum/`)
- vLLM Neuron documentation: <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/>
