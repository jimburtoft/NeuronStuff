# Running GPT-OSS-20B on Trainium2 with vLLM + vllm-neuron (Neuron SDK 2.33)

Tested on Trainium2 with the public **Neuron SDK 2.33 DLAMI**
(`Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0 20261020`): measured on `trn2.3xlarge`
2026-10-20, and every command below re-run as written on a fresh `trn2.3xlarge` on 2026-10-24
(served throughput matched the table within 2%).

The DLAMI's stock vllm-neuron runs GPT-OSS, but slowly and only with extra flags. This guide installs
two public branches on top of the DLAMI's vLLM environment. They carry the GPT-OSS changes we have
validated on Trainium2:

| Branch | What it is |
|---|---|
| [`jimburtoft/vllm-neuron@gptoss-trn2-perf-0.29`](https://github.com/jimburtoft/vllm-neuron/tree/gptoss-trn2-perf-0.29) | vllm-neuron 0.29.0.1.2.0 (the DLAMI's version) plus GPT-OSS model changes: FP8 expert weights, faster prefill and decode, and start-up fixes |
| [`jimburtoft/nki-library@gptoss-trn2-perf-2.33`](https://github.com/jimburtoft/nki-library/tree/gptoss-trn2-perf-2.33) | NKI Library 2.33 plus small changes to five kernels (MoE decode, MoE prefill, attention). See [`GPT_OSS_TRN2_PATCHES.md`](https://github.com/jimburtoft/nki-library/blob/gptoss-trn2-perf-2.33/GPT_OSS_TRN2_PATCHES.md) |

Both are drop-in replacements for the DLAMI's copies. The model code detects the patched kernels and
still works with the stock NKI Library, just more slowly.

Compared with the DLAMI's stock plugin, same instance (details in [Results](#results)):

- **Time to first token: 5.5x lower** (51 ms vs 283 ms at 128 tokens, 412 ms vs 2,262 ms at 4,096 tokens).
- **Decode: 1.7-2.1x faster per step** (5.1 ms vs 8.5 ms at batch 1, 20.7 ms vs 43.8 ms at batch 32).
- **Served throughput: 2.0-3.5x with ~400-token prompts and 3.3-5.8x with ~4,000-token prompts**, from 1 to 32 concurrent requests.
- **No `--hf-overrides` and no `NEURON_SKIP_EFA_AFFINITY`** needed. The published checkpoint starts as is.
- **GSM8K accuracy unchanged** within run-to-run noise.

---

## 1. Launch the instance

| Setting | Value |
|---|---|
| Instance type | `trn2.3xlarge` (1 Trainium2 device, 4 logical NeuronCores at LNC=2) |
| AMI name | **`Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0 20261020`** |
| Neuron SDK | **2.33** |
| Disk | **300 GB gp3** minimum (600 GB if you keep several compile caches) |

`trn2` instances need a **Capacity Block**.

AMI IDs (verified 2026-10-23):

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

The DLAMI ships with **no swap**, and `neuronx-cc` can use 40+ GB of host RAM per graph. When it runs
out of memory, the error blames the compiler, not memory.

```bash
sudo dd if=/dev/zero of=/swapfile bs=1M count=65536 status=none
sudo chmod 600 /swapfile
sudo mkswap --force /swapfile     # --force is REQUIRED on this kernel; without it mkswap silently does nothing
sudo swapon /swapfile
free -g                            # confirm ~63 GB swap
```

On a fresh gp3 volume this takes about 10-15 minutes. `mkswap` waits for the 64 GB file to reach
disk. It is not hung.

---

## 2. Install the two branches into the DLAMI's vLLM environment

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate

pip install --no-deps --force-reinstall "git+https://github.com/jimburtoft/vllm-neuron.git@gptoss-trn2-perf-0.29"
pip install --no-deps --force-reinstall "git+https://github.com/jimburtoft/nki-library.git@gptoss-trn2-perf-2.33"
rm -rf ~/.cache/neuron_libtorch     # the compile cache does not notice kernel changes; start clean
```

Check that both took effect:

```bash
python -c "
from nkilib.core.moe.moe_tkg import selective_expert_impl as m
import vllm_neuron.vllm.platform as p
print('patched kernels:', getattr(m, 'SUPPORTS_GATE_UP_MAJOR', False))
print('patched plugin: ', hasattr(p.NeuronPlatform, '_cap_memory_for_gpt_oss_fp8'))"
# patched kernels: True
# patched plugin:  True
```

Why the flags matter:

- **`--force-reinstall` is required for vllm-neuron.** The branch has the same version number as the
  DLAMI's plugin (`0.29.0.1.2.0`). A plain `pip install git+...` builds the branch, decides the
  requirement "is already satisfied", and **installs nothing**. You would run the stock plugin
  without knowing.
- **`--no-deps`** keeps the DLAMI's pinned torch, vllm and nki versions.
- **Always `source .../activate`.** Calling `/opt/.../bin/python` directly fails at the first compile
  (`posix_spawn failed: No such file or directory`), because the plugin starts `neuronx-cc` from `PATH`.

Versions in the environment after the install:

| Component | Version |
|---|---|
| vllm | `0.29.0` |
| vllm-neuron | `0.29.0.1.2.0` (branch `gptoss-trn2-perf-0.29`) |
| nki | `0.7.0` |
| NKI Library | branch `gptoss-trn2-perf-2.33` |
| neuronx-cc | `2.28.7631.0` |
| torch | `2.13.0` |
| Neuron driver / runtime | `2.31.7.0` / `2.35.30.0` |

---

## 3. Download the model

```bash
hf download openai/gpt-oss-20b --local-dir ~/models/gpt-oss-20b --exclude "original/*" --exclude "metal/*"
```

`original/` and `metal/` are other formats and are not used.

---

## 4. Serve it

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate

vllm serve ~/models/gpt-oss-20b \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 64 \
  --block-size 128 \
  --no-enable-prefix-caching \
  --additional-config '{"neuron_config": {
      "quantization": "fp8",
      "num_batched_tokens_buckets": [128, 256, 512, 1024, 2048],
      "decode_context_length_buckets": [1024, 2048, 4096, 7168],
      "on_device_sampling_config": {"all_greedy": true}}}'
```

**The first start compiles for about 20-50 minutes** (42 minutes on a fresh trn2.3xlarge in our last run). Run it in `tmux` or `screen`. Later starts reuse
the compile cache (`~/.cache/neuron_libtorch`). The server is ready when the log shows
`Application startup complete`.

What the settings do:

| Setting | Why |
|---|---|
| `--tensor-parallel-size 4` | Uses all 4 cores of the device. This is the fastest layout we measured. Two TP=2 copies on one device were slower (see [Notes](#notes)). |
| `"quantization": "fp8"` | Stores the expert weights as FP8 (converted exactly from the MXFP4 checkpoint), so decode reads half as many bytes. A BF16 copy of the experts is kept for prefill. Because of that copy, the plugin lowers `gpu_memory_utilization` from 0.92 to 0.80 and logs a warning. That is expected. |
| `--max-num-batched-tokens 2048` + `num_batched_tokens_buckets` | Prompts are processed in chunks of up to 2048 tokens, and the buckets compile smaller prefill graphs so a short prompt does not pay for a 2048-token chunk. One server is then as fast as `512` for short prompts and as fast as `2048` for long ones (see [Results](#results)). Each bucket must be a multiple of 128 and the last must equal `--max-num-batched-tokens`. |
| `decode_context_length_buckets` | Compiles decode at several context lengths, so short requests do not pay for `max_model_len` on every step. |
| `"all_greedy": true` | Skips the general sampler. Remove it if you need temperature or top-p sampling. |
| `--no-enable-prefix-caching` | Our numbers are without prefix caching. It works (it needs a `max-num-batched-tokens` from the list above), but we have not measured it. |
| `--max-num-seqs 64` | Throughput peaks at about 64 concurrent requests. More concurrency made it slower. |

`quantization: "bf16"` also works. It is slower at decode (6.0 vs 5.1 ms per step at batch 1) and
has the same prefill.

**`max_model_len` for long contexts.** If every request (prompt + output) stays under about 6,000 tokens,
`--max-model-len 6144` with `"decode_context_length_buckets": [1024, 2048, 4096]` decodes about 10 % faster
at 32 concurrent requests with ~4,000-token contexts than 8192 or 4608. At those two sizes the decode
attention kernel has to use a smaller KV block. Keep 8192 when you need full-length requests.

Query it:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "'$HOME'/models/gpt-oss-20b",
  "messages": [{"role": "user", "content": "Explain in two sentences why the sky appears blue."}],
  "max_tokens": 256}' | python -m json.tool
```

For offline (Python) use, see [`gpt_oss_run.py`](./gpt_oss_run.py). It uses the same settings, checks
that both branches are installed, and sends three chat prompts through `llm.chat()`. Use `llm.chat()`
(or the OpenAI server) for questions. `llm.generate()` on a raw string skips the chat template, so the
model continues the text instead of answering it.

On a trn2.48xlarge, pin a server to one device with `NEURON_VISIBLE_DEVICES=0-3` (logical cores
0-3). vllm-neuron rejects `NEURON_RT_VISIBLE_CORES`.

---

## Results

trn2.3xlarge, SDK 2.33 DLAMI, TP=4, `max_model_len` 8192, greedy, prefix caching off. Every run uses
the same synthetic prompts.

### Prefill and decode (offline, `max-num-batched-tokens 512`)

| | stock DLAMI plugin (BF16) | **these branches, FP8** | these branches, BF16 |
|---|---:|---:|---:|
| Time to first token, 128 / 512 / 2,048 / 4,096 tokens (ms) | 283 / 285 / 1,124 / 2,262 | **51 / 57 / 201 / 412** | 51 / 58 / 202 / 412 |
| Decode step, batch 1 / 8 / 32 (ms) | 8.5 / 36.5 / 43.8 | **5.1 / 13.4 / 20.7**\* | 6.0 / 23.5 / 26.3 |

\* Batch 8 measured 2026-10-26, after the branch switched batch 8 to the selective-expert decode kernel (17.8 ms before).

With `max-num-batched-tokens 2048` and the prefill buckets from the command above, FP8 time to first
token is 44 / 56 / 130 / 254 ms at 128 / 512 / 2,048 / 4,096 tokens: as fast as 512 for short prompts,
and 1.6x faster for long ones. Without the buckets, 2048 makes short prompts slower (77 ms at 128 tokens).

The offline runs above set `VLLM_NEURON_BACKEND=vllm_neuron` (the older XLA compile route). **Do not
set it.** The plugin's default (native) route, which `vllm serve` above uses, measured the same within
a few percent with these branches (re-tested 2026-10-23: FP8 decode 5.2 / 17.2 / 20.7 ms at batch
1 / 8 / 32, time to first token 47 / 54 / 197 ms at 128 / 512 / 2,048 tokens), and its first compile is
about 2x faster. Also **do not pass `--optimization-level`**: the default `-O2` is the fastest level we
measured on the native route, and `-O1` makes decode about 15% slower.

### Served throughput (`vllm serve`, 128 output tokens per request)

Output tokens/s at 1 / 4 / 16 / 32 / 64 concurrent requests:

| Configuration | ~400-token prompts | ~4,000-token prompts |
|---|---|---|
| stock DLAMI plugin (BF16)\* | 92 / 135 / 196 / 264 / - | 35 / 40 / 42 / 40 / - |
| FP8, `max-num-batched-tokens 512` | 178 / 383 / 640 / 935 / 1,025 | 112 / 173 / 211 / 235 / 239 |
| FP8, `max-num-batched-tokens 2048` | 171 / 352 / 557 / 768 / 789 | 134 / 230 / 303 / 356 / 348 |
| **FP8, 2048 + prefill buckets (the command above)** | **177 / 379 / 633 / 922 / 955** | **134 / 230 / 303 / 356 / 347** |

\* 1 to 32 concurrent requests only, with `--max-num-seqs 32`, `max-num-batched-tokens 512`,
`NEURON_SKIP_EFA_AFFINITY=1` and `--hf-overrides`.

The rows for 512, 2048 and buckets were measured on 2026-10-24 on one Trainium2 device (4 cores of a
trn2.48xlarge, SDK 2.33 release container, same branches). Our earlier trn2.3xlarge numbers for 512 and 2048 match
them within 1-3 %.

- **Use 2048 + buckets as the default.** Short prompts get the speed of 512 (time to first token
  44 ms at 128 tokens, vs 77 ms with 2048 alone) and long prompts the speed of 2048 (1.3-1.5x the
  throughput of 512).
- **The one exception is short prompts at 64 concurrent requests**, where 512 is 7 % faster: 2048 leaves
  less room for the KV cache (353K vs 468K tokens), so not all 64 requests fit at once.
- **More than 64 concurrent requests made it slower.** Each decode step got slower faster than the
  batch grew.

### Accuracy

GSM8K chain-of-thought, 250 questions, flexible-extract:

| | Score |
|---|---:|
| stock DLAMI plugin (BF16) | 0.876 |
| these branches, FP8 | 0.856-0.868 (three runs) |

The difference is within noise. A paired test on the questions that changed gives p = 0.23. Greedy
output tokens matched the stock plugin in our offline checks.

---

## Notes

- **Stock vllm-neuron without the branches.** The DLAMI's own plugin needs
  `NEURON_SKIP_EFA_AFFINITY=1` (trn2.3xlarge has no EFA, and all four workers die with
  `No EFA device found ... /infiniband`). It also needs
  `--hf-overrides '{"quantization_config": {}}'`, or it rejects the checkpoint with
  `gpt_oss_mxfp4 quantization is currently not supported in neuron`. On Trainium2 it runs BF16 only.
  The branches remove both requirements.
- **Stock vllm-neuron is about 3x slower at decode on its default compile route.** On Trainium2 the
  stock plugin's GPT-OSS decode takes 26.8 ms per step at batch 1 on the default (native) route, against
  8.5 ms with `VLLM_NEURON_BACKEND=vllm_neuron`, and its first compile takes about 50 minutes. The
  branches do not have this problem (5.2 ms on the default route). If you must run the stock plugin,
  set `VLLM_NEURON_BACKEND=vllm_neuron`.
- **Two smaller copies were slower than one TP=4 copy.** Running two TP=2 servers on one device
  (cores 0-1 and 2-3) peaked at 829 / 364 tokens/s (short / long prompts), against 1,014 / 380 for one
  TP=4 server. Each TP=2 copy decodes about 1.5x slower, which cancels out the second copy.
- **Don't go below TP=2.** The weights do not fit on one core. TP=2 works (see the previous note), but
  one TP=4 server is faster.
- **FP8 and the KV cache.** With the BF16 prefill copy, FP8 leaves about 350K tokens of KV cache with the
  command above (470K at `max-num-batched-tokens 512`), about 43 full-length 8K requests. Set
  `GPTOSS_HYBRID_PREFILL=0` to drop the copy: 2.1x the KV cache (756K tokens) at the same prefill and decode
  speed (time to first token 127.5 vs 129.0 ms at 2,048 tokens; decode 5.20 vs 5.20 ms at batch 1, 20.6 vs
  20.7 ms at batch 32). Served throughput (output tok/s, same command, 2026-10-26):

  | | c1 / 4 / 16 / 32 / 64, ~400-token prompts | ~4,000-token prompts |
  |---|---|---|
  | with the copy (default) | 177 / 378 / 627 / 911 / 945 | 135 / 231 / 304 / 358 / 349 |
  | `GPTOSS_HYBRID_PREFILL=0` | 176 / 372 / 613 / 888 / **1,118** | 134 / 228 / 298 / 352 / **384** |

  Without the copy, 1-3 % lower at 4-32 concurrent requests (time to first token under load is a little higher)
  and 10-18 % higher at 64. Use `GPTOSS_HYBRID_PREFILL=0` for about 64 concurrent requests or when you need
  the KV cache. GSM8K-CoT (250 questions): 0.880-0.892 in all four runs (with and without the copy).
- **Every change can be switched off** with an environment variable (`GPTOSS_*`). See the top of
  `vllm_neuron/model/gpt_oss/model_bf16.py` on the branch, and the
  [GPT-OSS recipe on the branch](https://github.com/jimburtoft/vllm-neuron/blob/gptoss-trn2-perf-0.29/docs/model-recipes/gpt-oss.md).
- **Greedy output can vary between runs under load.** The scheduler batches requests differently from
  run to run. Use `--max-num-seqs 1` if you need bit-identical output for comparisons.
- **The compile cache does not track kernel source.** After you change the NKI Library install, run
  `rm -rf ~/.cache/neuron_libtorch`, or you keep running the old kernels.

## Troubleshooting

| Symptom | Fix |
|---|---|
| No speed-up, and the check in step 2 prints `False` for the plugin | Reinstall vllm-neuron with `--no-deps --force-reinstall` |
| Check prints `False` for the kernels | Reinstall the NKI Library branch, then `rm -rf ~/.cache/neuron_libtorch` |
| `posix_spawn failed: No such file or directory` at the first compile | `source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate` instead of calling its `python` directly |
| `gpt_oss_mxfp4 quantization is currently not supported in neuron` | You are on the stock plugin. Install the branch (step 2) |
| `No EFA device found ... /infiniband` | You are on the stock plugin. Install the branch, or set `NEURON_SKIP_EFA_AFFINITY=1` |
| `gpu_memory_utilization is reduced from 0.92 to 0.80` warning | Expected with `quantization: "fp8"` |
| `neuronx-cc was forcibly killed` / compile dies | Add the swap file |
| Stock plugin: decode ~3x slower than expected, first compile ~50 min | Install the branches, or set `VLLM_NEURON_BACKEND=vllm_neuron` on the stock plugin |

## Trainium3

On Trainium3 GPT-OSS runs the published MXFP4 checkpoint natively. Use the Trainium3 branch
[`jimburtoft/vllm-neuron@gptoss-trn3-mx-0.29`](https://github.com/jimburtoft/vllm-neuron/tree/gptoss-trn3-mx-0.29)
instead of the Trainium2 one (no `quantization` setting, no NKI Library branch), with
`tensor_parallel_size=8`, `max_num_batched_tokens=2048` and
`"num_batched_tokens_buckets": [128, 256, 512, 1024, 2048]`. See the
[Trn3 section of the recipe on that branch](https://github.com/jimburtoft/vllm-neuron/blob/gptoss-trn3-mx-0.29/docs/model-recipes/gpt-oss.md#trn3-performance-options).
Leave `VLLM_NEURON_BACKEND` and `--optimization-level` unset there too.

## References

- GPT-OSS: <https://huggingface.co/openai/gpt-oss-20b>
- vllm-neuron branch: <https://github.com/jimburtoft/vllm-neuron/tree/gptoss-trn2-perf-0.29>
- NKI Library branch: <https://github.com/jimburtoft/nki-library/tree/gptoss-trn2-perf-2.33>
- Upstream vllm-neuron: <https://github.com/vllm-project/vllm-neuron>
- Neuron SDK 2.33 release notes: <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/2.33.0.html>
