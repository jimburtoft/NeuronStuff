# Running GPT-OSS-20B on Trainium2 with vLLM + vllm-neuron (Neuron SDK 2.32)

Tested and working **2026-09-30** on a `trn2.3xlarge` using the **Neuron SDK 2.32 DLAMI (20260818)**.

GPT-OSS does **not** run out of the box on Trainium2. The stock invocation fails in about 9 seconds.
It needs **three configuration changes**, documented below. The good news: all three are
configuration — **no source patches, no forks, no custom kernels.**

This document covers only what is required to get it running. No performance tuning.

---

## Why the obvious approach fails

OpenAI ships `openai/gpt-oss-20b` in **MXFP4** (4-bit microscaling). Its `config.json` contains:

```json
"quantization_config": { "quant_method": "mxfp4" }
```

MXFP4 *compute* requires NeuronCore-v4 (Trainium3). Trainium2 is NeuronCore-v3, so it cannot
execute MXFP4 matmuls. vllm-neuron **does** ship an MXFP4→BF16 dequantizer that runs on the host at
load time, but you have to ask for it explicitly — and the error you get if you don't is unhelpful.

So on Trainium2 the model runs in **BF16**. The MXFP4 weights are dequantized while loading, which
means HBM holds roughly 4x the on-disk checkpoint size. That is expected, not a misconfiguration.

---

## 1. Launch the instance

| Setting | Value |
|---|---|
| Instance type | `trn2.3xlarge` (4 logical NeuronCores at LNC=2) |
| AMI name | **`Deep Learning AMI Neuron (Ubuntu 24.04) 20260818`** |
| Neuron SDK | **2.32** |
| Disk | **600 GB gp3** minimum |

`trn2` instances require a **Capacity Block** (`trn1` and `inf2` do not).

AMI IDs — verified 2026-09-30:

| Region | AMI ID |
|---|---|
| us-east-1 (N. Virginia) | `ami-0f8b952eba2c08d62` |
| us-east-2 (Ohio) | `ami-0222021b369f03219` |
| us-west-2 (Oregon) | `ami-05a802a92b721afb9` |
| sa-east-1 (São Paulo) | `ami-06f5c32a48089ad7a` |
| ap-southeast-4 (Melbourne) | `ami-01f66e576e60931ed` |

Look it up yourself in any other region:

```bash
aws ec2 describe-images --owners amazon --region <REGION> \
  --filters "Name=name,Values=Deep Learning AMI Neuron (Ubuntu 24.04) 20260818" \
  --query 'Images[*].[ImageId,Name]' --output table
```

**On disk size**: 300 GB is the usual recommendation, but GPT-OSS-20B needs more. `hf download`
pulls ~39 GB (the repo includes `metal/` and `original/` subdirectories you don't need, on top of the
13.7 GB of safetensors), and the compiler cache plus a swap file add more. 600 GB avoids trouble.

### Add swap before you do anything else

The DLAMI ships with **zero** swap, and `neuronx-cc` wants ~40-45 GB of host RAM per graph. Without
swap, compilation can be OOM-killed, and the error message blames the compiler rather than memory.

```bash
sudo dd if=/dev/zero of=/swapfile bs=1M count=65536 status=none
sudo chmod 600 /swapfile
sudo mkswap --force /swapfile     # --force is REQUIRED on this DLAMI's kernel
sudo swapon /swapfile
free -g                            # confirm ~63 GB swap
```

`sudo mkswap` without `--force` silently does nothing on this kernel. Check `free -g`.

---

## 2. Activate the environment

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin/activate
```

Exact versions in that venv, as verified on the instance:

| Component | Version |
|---|---|
| vllm | `0.24.0` |
| **vllm-neuron** | **`0.24.0.1.1.0`** |
| neuronx-cc | `2.27.5334.0+f702b353` |
| nki | `0.6.0+31049202112.g85070674` |
| torch / torch-xla | `2.11.0` / `2.11.0` |
| transformers | `5.15.0` |
| numpy | `2.3.5` |
| libtorch-neuronx-lite | `2.11.0.1.0.1284+f49d8626` |

Host packages (`dpkg -l | grep aws-neuronx`):

| Package | Version |
|---|---|
| aws-neuronx-dkms | `2.30.2.0` |
| aws-neuronx-runtime-lib | `2.34.10.0-ac18d186d` |
| aws-neuronx-collectives | `2.34.10.0-74eaafac6` |
| aws-neuronx-tools | `2.32.28.0-526c2b7f6` |

One thing to know about this venv: it ships `libtorch-neuronx-lite`, not full `torch-neuronx`, so
`import torch_neuronx` fails and there is no `torch_neuronx.trace()`. That is fine here — vllm-neuron
owns the device path for vLLM serving. It only matters if you wanted to trace a model yourself.

Confirm the hardware:

```bash
neuron-ls
# trn2.3xlarge -> 1 device, 4 cores, 96 GB, logical-neuroncore-config: 2
```

---

## 3. Download the model

```bash
hf download openai/gpt-oss-20b --local-dir ~/models/gpt-oss-20b
```

Takes a few minutes. You only need the three `model-*.safetensors` files plus the configs and
tokenizer; the `metal/` and `original/` directories are other formats and are not used here.

---

## 4. The three required changes

### Change 1 — set `NEURON_SKIP_EFA_AFFINITY=1`

```bash
export NEURON_SKIP_EFA_AFFINITY=1
```

Without this, **all four tensor-parallel workers die immediately**:

```
FileNotFoundError: [Errno 2] No such file or directory:
  '/sys/bus/pci/devices/0000:c9:00.0/infiniband'
```

`trn2.3xlarge` has no EFA device. vllm-neuron tries to set EFA affinity unconditionally. The library's
own error text concedes this is "a CPU performance optimization, not a correctness requirement" and
names this environment variable as the fix — it just doesn't apply it automatically. (The message
mentions `trn3 3xlarge`, but the same thing happens on `trn2.3xlarge`.)

### Change 2 — select the BF16 path, in two places

Both of these are needed. Either alone is not enough:

```python
hf_overrides={"quantization_config": None}
additional_config={"neuron_config": {"quantization": "bf16"}}
```

Without them:

```
ValidationError: Value error, gpt_oss_mxfp4 quantization is currently not supported in neuron.
```

vLLM checks the checkpoint's quantization method against the Neuron platform's allowlist, which
contains only `neuron_quant`, `compressed-tensors`, and `modelopt`. `gpt_oss_mxfp4` isn't on it, so
the model is rejected before vllm-neuron's own GPT-OSS code is ever reached.

- `hf_overrides={"quantization_config": None}` hides the checkpoint's MXFP4 marker so that allowlist
  check passes.
- `additional_config` then explicitly selects the BF16 implementation, which triggers the host-side
  MXFP4→BF16 dequantization at load.

Setting only `additional_config` still fails — the vLLM-level check runs first. Setting only
`hf_overrides` is fragile, because you are then relying on a fallback rather than asking for BF16.

> Worth knowing: vllm-neuron contains a much more helpful message —
> `quantization='mxfp4' is not supported on TRN2. Please use quantization='bf16'` — but the generic
> vLLM check fires first, so you never see it. If you hit the `gpt_oss_mxfp4` error, that second
> message is the one that actually tells you what to do.

### Change 3 — set `max_num_batched_tokens`

```python
max_num_batched_tokens=2048    # or 512, 1024, 4096, 8192
```

Without it:

```
ValueError: Automatic Prefix Caching (APC) requires segmented prefill to be enabled.
Either disable APC with '--no-enable-prefix-caching' or set 'max_num_batched_tokens' to a
supported segmented prefill size [512, 1024, 2048, 4096, 8192] to auto-enable segmented prefill.
```

In vLLM 0.24, both `enable_prefix_caching` and `enable_chunked_prefill` default to `True`, but the
default `max_num_batched_tokens` is not one of the supported segmented-prefill sizes — so the two
defaults contradict each other. Pick a value from that list. `max_num_batched_tokens` must be at
least as large as your longest prompt.

---

## 5. Working example

```python
# gpt_oss_run.py
import os
from vllm import LLM, SamplingParams

llm = LLM(
    model=os.path.expanduser("~/models/gpt-oss-20b"),
    tensor_parallel_size=4,            # 4 logical cores at LNC=2; TP=4 is REQUIRED (see below)
    max_model_len=4096,
    max_num_seqs=4,
    trust_remote_code=True,

    max_num_batched_tokens=2048,                                    # Change 3
    hf_overrides={"quantization_config": None},                     # Change 2a
    additional_config={"neuron_config": {"quantization": "bf16"}},   # Change 2b
)

prompts = [
    "Explain in two sentences why the sky appears blue.",
    "Write a Python function that reverses a linked list.",
    "What is 17 * 24? Show your reasoning briefly.",
]

outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=128))
for o in outputs:
    print(f"\n--- {o.prompt}\n{o.outputs[0].text}")
```

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin/activate
export NEURON_SKIP_EFA_AFFINITY=1          # Change 1
python gpt_oss_run.py
```

**First run takes about 7-8 minutes** (456 s measured) for weight loading, MXFP4→BF16 dequantization,
and compilation. Later runs reuse the compiler cache and start much faster.

Use `tmux` or `screen`. If your SSH session drops during compilation, `nohup` and `setsid` will not
reliably save the process.

### Confirmed output

```
BS=1: 128 tokens in 1.92 s = 66.8 tok/s
Batch of 3: 384 tokens in 4.60 s = 83.5 tok/s aggregate
```

Sample generation:

> *The sky appears blue because of Rayleigh scattering of sunlight by the atmosphere, which scatters
> shorter wavelengths of light more efficiently than longer wavelengths.*

Batching works correctly — no cross-request contamination. These numbers are from an untuned
configuration; they are a correctness check, not a benchmark.

---

## Notes and gotchas

**`tensor_parallel_size=4` is required.** `tensor_parallel_size=1` fails while loading weights:

```
ERROR - checkpoints.py:630 - [Rank 0] Error in device_load_checkpoint: nrt_tensor_allocate status=4
```

20B in BF16 does not fit on one LNC=2 core (24 GB). TP=4 spreads it across all four cores on the chip.

**Greedy decoding is not reproducible.** With `temperature=0.0`, six identical requests produced two
different outputs. The cause is the continuous-batching scheduler: setting `max_num_seqs=1` gives
bit-identical results every time. If you are comparing outputs or validating accuracy, use
`max_num_seqs=1` and accept the lower throughput.

**Don't expect much from batching.** 4x the batch size gave roughly 1.5x the aggregate throughput on
a single chip. Scale with more replicas rather than larger batches.

**`neuron_config.fp8_packed_kv=True` is silently ignored** on this path. It is accepted without
warning and has no effect — don't assume you've reduced KV-cache memory. `kv_cache_dtype="fp8"` fails
outright at engine init.

**Version pinning matters.** This was tested on SDK 2.32 / DLAMI `20260818` with
vllm-neuron `0.24.0.1.1.0`. The GPT-OSS support and the `neuron_config.quantization` option used here
arrived in the vllm-neuron 0.2x series; older plugin versions do not have this code path, and the
SDK 2.32 runtime ABI differs from earlier SDKs. Use this DLAMI, or a later one with
vllm-neuron >= 0.24.

---

## Troubleshooting

| Error | Change needed |
|---|---|
| `gpt_oss_mxfp4 quantization is currently not supported in neuron` | Change 2 — both `hf_overrides` and `additional_config` |
| `FileNotFoundError: ... /infiniband` | Change 1 — `NEURON_SKIP_EFA_AFFINITY=1` |
| `Automatic Prefix Caching (APC) requires segmented prefill` | Change 3 — `max_num_batched_tokens` |
| `nrt_tensor_allocate status=4` during load | Use `tensor_parallel_size=4` |
| `neuronx-cc was forcibly killed` / compile dies | Add the swap file |

---

## References

- GPT-OSS: <https://huggingface.co/openai/gpt-oss-20b>
- vllm-neuron: <https://github.com/vllm-project/vllm-neuron>
- Neuron SDK 2.32 release notes:
  <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/2.32.0.html>
- Capacity Blocks pricing: <https://aws.amazon.com/ec2/capacityblocks/pricing/>
