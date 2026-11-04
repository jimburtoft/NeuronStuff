# Running Gemma 3 12B on Trainium2 with vLLM + vllm-neuron (Neuron SDK 2.33)

Serve [`google/gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it) (text in, text out) on one
`trn2.3xlarge` with vLLM, on the public **Neuron SDK 2.33 DLAMI**
(`Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0 20261020`). Every command below was run as written
on a `trn2.3xlarge` starting from the stock DLAMI environment on 2026-11-03/04 (see [Changelog](#changelog)).

The DLAMI's stock vllm-neuron does not support Gemma 3. This guide installs two public branches on top of
the DLAMI's vLLM environment:

| Branch | What it is |
|---|---|
| [`jimburtoft/vllm-neuron@gemma3-trn2-perf-0.29`](https://github.com/jimburtoft/vllm-neuron/tree/gemma3-trn2-perf-0.29) | vllm-neuron 0.29.0.1.2.0 (the DLAMI's version) plus the Gemma 3 text decoder, sliding-window-aware KV admission, a sliding-window KV ring, an argmax fix for large batches, and a decode-mask fix for odd batch sizes |
| [`jimburtoft/nki-library@gemma3-trn2-perf-2.33`](https://github.com/jimburtoft/nki-library/tree/gemma3-trn2-perf-2.33) | NKI Library 2.33 plus `head_dim` 256 support in the fused decode attention kernel (in-kernel output projection and KV write). See [`GEMMA3_TRN2_PATCHES.md`](https://github.com/jimburtoft/nki-library/blob/gemma3-trn2-perf-2.33/GEMMA3_TRN2_PATCHES.md) |

**Install both.** The model also runs with the DLAMI's own NKI Library, but slower: in our earlier tests the NKI Library branch added about 9 % throughput (not re-measured for this guide).

Measured on one Trainium2 device (TP=4), 8 English Wikipedia articles translated to Hindi, greedy, BF16
weights and KV cache, no speculative decoding (details in [Results](#results)):

| Workload | Output tokens/s (offline, one device) |
|---|---:|
| 2,048-token prompts, up to 2,048 output tokens, 83 requests at once | **1,826** |
| 4,096-token prompts, up to 4,096 output tokens, 63 requests at once | **1,416** |

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

The DLAMI ships with **no swap**, and `neuronx-cc` can use a lot of host memory while it compiles. When it
runs out of memory, the error blames the compiler, not memory.

```bash
sudo dd if=/dev/zero of=/swapfile bs=1M count=65536 status=none
sudo chmod 600 /swapfile
sudo mkswap --force /swapfile     # --force is REQUIRED on this kernel; without it mkswap silently does nothing
sudo swapon /swapfile
free -g                            # confirm ~63 GB swap
```

---

## 2. Install the two branches into the DLAMI's vLLM environment

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate

pip install --no-deps --force-reinstall "git+https://github.com/jimburtoft/vllm-neuron.git@gemma3-trn2-perf-0.29"
pip install --no-deps --force-reinstall "git+https://github.com/jimburtoft/nki-library.git@gemma3-trn2-perf-2.33"
rm -rf ~/.cache/neuron_libtorch     # the compile cache does not notice kernel changes; start clean
```

Check that both took effect:

```bash
python -c "
import importlib
from vllm_neuron.model.registry import get_models
abt = importlib.import_module('nkilib.experimental.transformer.attention_block_tkg')
print('Gemma 3 plugin:', 'Gemma3ForConditionalGeneration' in dict(get_models()))
print('d256 kernels:  ', getattr(abt, 'SUPPORTS_D256_OUT_PROJ', False) and getattr(abt, 'SUPPORTS_D256_BLOCK_KV_UPDATE', False))" 2>/dev/null
# Gemma 3 plugin: True
# d256 kernels:   True
```

- **`--force-reinstall` is required for vllm-neuron.** The branch has the same version number as the DLAMI's
  plugin (`0.29.0.1.2.0`), so a plain `pip install git+...` decides the requirement "is already satisfied" and
  installs nothing.
- **`--no-deps`** keeps the DLAMI's pinned torch, vllm and nki versions.
- The NKI Library branch installs as `nkilib_src`; the DLAMI's bundled `nkilib` hands over to it on import.
  `NKILIB_FORCE_BUNDLED_LIBRARY=1` switches back to the bundled copy without uninstalling.
- **Always `source .../activate`.** The plugin starts `neuronx-cc` from `PATH`.

Versions in the environment after the install:

| Component | Version |
|---|---|
| vllm | `0.29.0` |
| vllm-neuron | `0.29.0.1.2.0` (branch `gemma3-trn2-perf-0.29` @ `48d3630`) |
| NKI Library | branch `gemma3-trn2-perf-2.33` @ `84e6346` |
| nki | `0.7.0` |
| neuronx-cc | `2.28.7631.0` |
| torch | `2.13.0` |

---

## 3. Download the model

The model is gated. Accept the license on its [Hugging Face page](https://huggingface.co/google/gemma-3-12b-it) first.

```bash
hf auth login
hf download google/gemma-3-12b-it --exclude "*.gguf"
```

The checkpoint includes the vision tower. This port is text only and skips it at load time.

---

## 4. Serve it

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
export NEURON_SKIP_EFA_AFFINITY=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_NEURON_MODULAR_FLOW_MAC_THRESHOLD=1000000000000000
export VLLM_NEURON_SWA_REAL_ADMISSION=1 VLLM_NEURON_SWA_EARLY_FREE=1 VLLM_NEURON_SWA_RING=1
export VLLM_NEURON_SWA_CTX_BUCKETS=1 VLLM_NEURON_SWA_ALIGN_TOKENS=1024
export GEMMA3_PACKED_PREFILL=1 VLLM_NEURON_MAX_PREFILLS_PER_BATCH=1
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.55

vllm serve google/gemma-3-12b-it \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 83 \
  --max-num-batched-tokens 2176 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --additional-config '{"neuron_config": {
      "decode_context_length_buckets": [2560, 3072, 3584],
      "on_device_sampling_config": {"all_greedy": true},
      "max_logprobs": 0}}'
```

This is the configuration for requests up to 4,096 tokens (prompt + output). For requests up to 8,192 tokens, use
[the 8k settings](#settings-for-8192-token-requests).

**The first start compiles for about 105 minutes** on a trn2.3xlarge. Run it in `tmux` or `screen`. Later
starts reuse the compile cache (`~/.cache/neuron_libtorch`). The server is ready when the log shows
`Application startup complete`.

Query it:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "google/gemma-3-12b-it",
  "messages": [{"role": "user", "content": "Explain in two sentences why the sky appears blue."}],
  "max_tokens": 256}' | python -m json.tool
```

What the settings do:

| Setting | Why |
|---|---|
| `--tensor-parallel-size 4` | Uses all 4 cores of the device (2 of the 8 KV heads per core). |
| `VLLM_NEURON_SWA_REAL_ADMISSION=1`, `VLLM_NEURON_SWA_EARLY_FREE=1` | 40 of Gemma 3's 48 layers use a 1,024-token sliding window. vLLM charges each request the KV of a full-attention layer for all of them, so far fewer requests fit than the memory allows. These admit by each request's real peak and free blocks that leave the window. |
| `VLLM_NEURON_SWA_RING=1` | Each sliding-window layer keeps a fixed ring of window blocks per request. Without it, 48 requests fit instead of 83, and throughput halves. The ring needs single-shot prefill (`--max-num-batched-tokens` >= the prompt length), no prefix caching, and no speculative decoding. |
| `VLLM_NEURON_MODULAR_FLOW_MAC_THRESHOLD=1e15` | Lets the decode kernel write the new KV in place (with the NKI Library branch). |
| `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.55` | Share of device memory for the KV cache. 0.55 fits 83 requests of 4,096 tokens. |
| `decode_context_length_buckets` + `VLLM_NEURON_SWA_CTX_BUCKETS=1` | Decode graphs at several context lengths, so short contexts read less KV. Buckets must be multiples of 256. |
| `"all_greedy": true` | Skips the general sampler. Remove it if you need temperature or top-p sampling. |
| `"max_logprobs": 0` | Skips gathering the full logits every step. Remove it if you need logprobs. |
| `--max-num-seqs 83` | The most 4,096-token requests that fit at 0.55. |

### Settings for 8,192-token requests

Same environment, with `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.60` (the highest value that loads at this
length), and:

```bash
vllm serve google/gemma-3-12b-it --tensor-parallel-size 4 --dtype bfloat16 \
  --max-model-len 8192 --max-num-seqs 63 --max-num-batched-tokens 4224 \
  --gpu-memory-utilization 0.90 --no-enable-prefix-caching \
  --additional-config '{"neuron_config": {
      "decode_context_length_buckets": [5120, 6144, 7168],
      "on_device_sampling_config": {"all_greedy": true},
      "max_logprobs": 0}}'
```

---

## 5. Reproduce the benchmark

The vllm-neuron branch ships the benchmark, the prompts and the accuracy reference under
[`tools/gemma3/`](https://github.com/jimburtoft/vllm-neuron/tree/gemma3-trn2-perf-0.29/tools/gemma3). [`run_gemma3.sh`](./run_gemma3.sh) in this
directory runs everything below in order.

```bash
git clone -b gemma3-trn2-perf-0.29 https://github.com/jimburtoft/vllm-neuron.git ~/vllm-neuron-gemma3
cd ~/vllm-neuron-gemma3
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
bash tools/gemma3/run.sh offline-2k     # writes out/target_wiki_2048.json, prints one POINT line per timed window
bash tools/gemma3/run.sh offline-4k     # writes out/target_wiki_4096.json
CONC="16 32 56 83" bash tools/gemma3/run.sh online-2k   # vllm serve + streaming client
```

`run.sh` sets the environment from section 4. Each offline run compiles its own graphs on first use. It then runs one
correctness window of 83 (or 63) requests (the 8 prompts, cycled) and times 3 more windows of the same size, all submitted
at once. The accuracy rows come from the correctness window, so they are decoded with the same batch size and compiled
graph as the timed windows. Every request in every window is compared with the accuracy row of its prompt: the
`CORRECTNESS` line and each `POINT` line print `match_gated_rows`. Any copy that differs is written to
`out/target_wiki_<N>.json.variants.json`, which `fp32_gate.py` also accepts.

---

## Results

trn2.3xlarge, SDK 2.33 DLAMI, branches as above, TP=4, LNC=2, BF16, greedy, prefix caching off, 2026-11-03/04.
Workload: 8 English Wikipedia articles (Anarchism, Albedo, Alabama, Achilles, Abraham Lincoln, Aristotle,
Academy Awards, Altruism), each in the chat template with the system prompt "Translate the text below to
Hindi." and cut to exactly 2,048 or 4,096 prompt tokens. Output stops at end of sequence or at the cap; in
every run all requests ran to the cap.

### Offline (`LLM.generate`, all requests submitted at once)

| Workload | `--max-num-seqs` | Output tokens/s, 3 timed windows | Median |
|---|---:|---|---:|
| 2,048-token prompts, up to 2,048 output tokens (`run.sh offline-2k`) | 83 | 1,826.3 / 1,825.4 / 1,825.8 | **1,825.8** |
| 4,096-token prompts, up to 4,096 output tokens (`run.sh offline-4k`) | 63 | 1,416.2 / 1,415.6 / 1,415.6 | **1,415.6** |

### Served (`vllm serve`, streaming `/v1/completions`, token-ID prompts, 2 waves per level)

2,048-token prompts, up to 2,048 output tokens, server settings from section 4:

| Concurrent requests | Output tokens/s | Time to first token p50 / p95 | Time per output token p50 |
|---:|---:|---|---:|
| 16 | 723.8 | 1.29 s / 2.15 s | 21.6 ms |
| 32 | 1,199.6 | 2.42 s / 4.40 s | 25.6 ms |
| 56 | 1,450.7 | 4.11 s / 7.57 s | 36.7 ms |
| 83 | 1,822.3 | 5.97 s / 11.25 s | 42.6 ms |

Time to first token rises quickly with concurrency. The server runs one 2,048-token prefill per step, so a burst
of requests waits for the prefills ahead of it. For lower time to first token, run fewer concurrent requests per server.

### Accuracy

The accuracy rows of each offline run (one per prompt, taken from the correctness window of 83 or 63 requests) are compared token by token with an fp32 reference (Hugging Face
transformers, float32, eager attention, CPU, same prompts). BF16 outputs eventually diverge from fp32 at a position
where the top two tokens are nearly tied. For each prompt, the first divergent position is fed to the fp32 model,
and the run passes if, at every prompt's first divergence, the fp32 logit gap between the two tokens is below 0.35,
the fp32 top token is the reference token, and the Neuron token is in the fp32 top 5. The 0.35 threshold is the
spread seen across recompiles of one unchanged BF16 configuration (max 0.277).

| Run | Pass | Largest fp32 gap at a first divergence | First divergent position, prompts 0-7 |
|---|---|---:|---|
| 2,048 | **yes**, 8/8 | 0.218 | 94, 179, 516, 38, 683, 365, 1245, 299 |
| 4,096 | **yes**, 8/8 | 0.131 | 380, 1063, 128, 33, 647, 477, 467, 63 |

In both runs every request of every window (4 x 83 and 4 x 63) produced exactly its prompt's accuracy row, so these results cover
all timed requests. The gates were run with `fp32_gate.py` on a `c7i.12xlarge`.

The outputs are fluent Hindi translations that start with a one-line English preamble. The script is
`tools/gemma3/fp32_gate.py`. It needs about 50 GB of RAM, so run it on a CPU instance (we used `c7i.12xlarge`):

```bash
python tools/gemma3/fp32_gate.py tools/gemma3/data/ref_fp32_wiki_2048.json out/target_wiki_2048.json gate_2048.json
```

---

## Notes

- **Text only.** The vision tower is dropped at load time. Image input is not supported on this branch.
- **The sliding-window settings are off by default in the plugin** so other models are unaffected. Set them
  as in section 4.
- **Prefix caching and chunked prefill are not supported with `VLLM_NEURON_SWA_RING=1`.** Keep
  `--max-num-batched-tokens` at or above the longest prompt.
- **`VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` above 0.60 fails at model load** with `--max-model-len 8192`.
- **The compile cache does not track kernel source.** After you change the NKI Library install, run
  `rm -rf ~/.cache/neuron_libtorch`.
- **Greedy output can vary between runs under load.** BF16 results depend on how requests are batched.
  Use `--max-num-seqs 1` if you need bit-identical output for comparisons.
- On a trn2.48xlarge, pin a server to one device with `NEURON_VISIBLE_DEVICES=0-3`. vllm-neuron rejects
  `NEURON_RT_VISIBLE_CORES`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Check in step 2 prints `False` for the plugin | Reinstall vllm-neuron with `--no-deps --force-reinstall` |
| Check prints `False` for the kernels | Reinstall the NKI Library branch, then `rm -rf ~/.cache/neuron_libtorch` |
| `No EFA device found ... /infiniband` | `export NEURON_SKIP_EFA_AFFINITY=1` |
| `posix_spawn failed: No such file or directory` at the first compile | `source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate` instead of calling its `python` directly |
| `neuronx-cc was forcibly killed` / compile dies | Add the swap file |
| Far fewer running requests than `--max-num-seqs` | The `VLLM_NEURON_SWA_*` variables are not set in the server's shell |

## Changelog

- **2026-11-03/04:** The vllm-neuron branch fixes the decode attention mask for odd decode batch sizes (such as 83 and 63)
  with 2 KV heads per core. The mask was built for a different layout from the one the decode kernel uses, so requests
  at those batch sizes attended to the wrong KV positions. The numbers in earlier versions of this guide (offline 1,838 / 1,418 and the
  online table) came from a build with this issue. Their accuracy check ran only 8 requests, which used an even decode batch
  that did not hit it. The benchmark now takes its accuracy rows from a full window of 83 / 63 requests and checks every timed
  request. All numbers above were re-measured with the fix (vllm-neuron `48d3630`, NKI Library `84e6346`).
- 2026-10-30/31: first version.

## References

- Gemma 3: <https://huggingface.co/google/gemma-3-12b-it>
- vllm-neuron branch: <https://github.com/jimburtoft/vllm-neuron/tree/gemma3-trn2-perf-0.29> (model README: `vllm_neuron/model/gemma3/README.md`)
- NKI Library branch: <https://github.com/jimburtoft/nki-library/tree/gemma3-trn2-perf-2.33>
- Upstream vllm-neuron: <https://github.com/vllm-project/vllm-neuron>
- Prompt text: English Wikipedia, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) (see `tools/gemma3/data/README.md` on the branch)
