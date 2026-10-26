# Muse-Glimmer-30B on Trainium2 with vLLM-Neuron (Neuron SDK 2.33)

Serve [`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B), a dense 30B vision-language model (text, image and video in; text out), on one `trn2.3xlarge` with vLLM. Two branches are installed into the public Neuron SDK 2.33 DLAMI's vLLM environment:

- **vllm-neuron:** [`jimburtoft/vllm-neuron@contrib/muse-glimmer-0.29`](https://github.com/jimburtoft/vllm-neuron/tree/contrib/muse-glimmer-0.29). This is vllm-neuron `0.29.0.1.2.0` plus the MuseGlimmer model and the plugin fixes it needs. Model README: [`vllm_neuron/model/muse_glimmer/README.md`](https://github.com/jimburtoft/vllm-neuron/blob/contrib/muse-glimmer-0.29/vllm_neuron/model/muse_glimmer/README.md).
- **NKI Library (optional):** [`jimburtoft/nki-library@contrib/muse-glimmer-2.33`](https://github.com/jimburtoft/nki-library/tree/contrib/muse-glimmer-2.33). This is the SDK 2.33 nkilib plus the kernel fixes found during the port. **The model runs on the stock SDK 2.33 nkilib.** The plugin branch carries its own copy of the gated decode kernel and works around the rest. Install this branch only if you want the upstream form of the kernel fixes, for example the un-chunked FP8 prefill path.

Without the plugin branch, stock vllm-neuron 0.29.0.1.2.0 does not know this model. vLLM 0.29 picks its own built-in implementation of the model and `vllm serve` fails.

Measured on trn2.3xlarge (one Trainium2 device, TP=4) with the recommended config below:

| Workload (in / out tokens) | Peak output tok/s (concurrency) | Single request: TTFT / time per output token |
|---|---|---|
| 128 / 512 | 1,061 (256) | 76 ms / 21.1 ms |
| 1024 / 256 | 773 (128) | 136 ms / 21.1 ms |
| 2048 / 512 | 787 (256) | 247 ms / 21.1 ms |
| 896x672 image + 256 / 128 | 304 (128) | 343 ms / 21.1 ms |

With DFlash speculative decoding (optional, section 6), a single request runs at **133 output tok/s** on natural text, 2.8x faster than without it.

## 1. Launch an instance

- **Instance:** `trn2.3xlarge` (1 Trainium2 device, 4 logical NeuronCores at the default LNC=2).
- **AMI:** *Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0* (multi-framework). To get the latest ID:
  ```bash
  aws ssm get-parameter --region <region> \
    --name /aws/service/neuron/dlami/multi-framework/ubuntu-24.04/latest/image_id \
    --query Parameter.Value --output text
  ```
- **Disk:** 300 GB or more. The BF16 checkpoint is 56 GB.

## 2. Add swap (once per instance)

The DLAMI ships with no swap, and `neuronx-cc` can use a lot of host memory while it builds the TP=4 graphs. When it runs out, the error blames the compiler.

```bash
sudo dd if=/dev/zero of=/swapfile bs=1M count=65536 status=none
sudo chmod 600 /swapfile
sudo mkswap --force /swapfile     # --force is required on this kernel
sudo swapon /swapfile
free -g                            # about 63 GB swap
```

## 3. Install the branches into the DLAMI's vLLM environment

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate   # vllm 0.29.0, vllm-neuron 0.29.0.1.2.0

pip install --no-deps --force-reinstall "git+https://github.com/jimburtoft/vllm-neuron.git@contrib/muse-glimmer-0.29"
rm -rf ~/.cache/neuron_libtorch     # the compile cache does not notice code changes; start clean

python -c "import vllm_neuron.model.muse_glimmer as m, os; print(os.path.dirname(m.__file__))"
```

Optional: the NKI Library branch. nkilib ships inside the `nki` wheel as the top-level package `nkilib`, so put the branch's `src/nkilib_src` first on `PYTHONPATH` to shadow it. `pip install` of the repo would install it under a different package name (`nkilib_src`) and would not take effect.

```bash
git clone -b contrib/muse-glimmer-2.33 https://github.com/jimburtoft/nki-library.git ~/nki-library-muse
export PYTHONPATH=~/nki-library-muse/src/nkilib_src${PYTHONPATH:+:$PYTHONPATH}
python -c "import nkilib, os; print(os.path.dirname(nkilib.__file__))"   # must point into ~/nki-library-muse
```

Set `PYTHONPATH` the same way in every shell that runs `vllm serve`.

- **`--force-reinstall`** is required. The branch has the same version number as the DLAMI's plugin, so a plain `pip install` decides the requirement is satisfied and installs nothing.
- **`--no-deps`** keeps the DLAMI's pinned torch, vllm and nki.
- **Always `source .../activate`.** The plugin starts `neuronx-cc` from `PATH`.

## 4. Download the model

The model is gated. Accept the license on its Hugging Face page first.

```bash
hf auth login
hf download meta-models/Muse-Glimmer-30B --local-dir ~/models/Muse-Glimmer-30B
# for DFlash (section 6):
hf download meta-models/Muse-Glimmer-30B-assistant --local-dir ~/models/Muse-Glimmer-30B-assistant
curl -sLO https://raw.githubusercontent.com/jimburtoft/NeuronStuff/main/muse-glimmer-vllm-neuron/kv_scales.json
```

`kv_scales.json` holds per-layer V scales for the FP8 KV cache. They were calibrated on the HF BF16 model over 28 natural text prompts and 4 images. The K scale needs no calibration: K is stored after the model's weightless QK-norm, so it is bounded by sqrt(head_dim).

## 5. Serve

**Recommended config: FP8 MLP weights plus FP8 KV cache.**
- MLP weights are self-quantized from the BF16 checkpoint at load time, per output channel.
- 58 sequences of 8192 tokens fit in the KV cache.

```bash
export NEURON_SKIP_EFA_AFFINITY=1 VLLM_NEURON_SWA_ALIGN_TOKENS=4096
export MUSE_GLIMMER_FP8=mlp MUSE_GLIMMER_KV_SCALES=$PWD/kv_scales.json
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.27
export VLLM_NEURON_BARRIER_TIMEOUT=14400            # FP8 prefill graphs take long to compile
export MUSE_GLIMMER_VIS_SMALL_BLOCK=1 MUSE_GLIMMER_VIS_MLP=nki MUSE_GLIMMER_VIS_LN=fused

vllm serve ~/models/Muse-Glimmer-30B --served-model-name muse --tensor-parallel-size 4 \
  --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8 --max-logprobs 0 --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' --mm-processor-kwargs '{"max_image_tokens": 1024}' \
  --additional-config '{
    "neuron_config": {"num_batched_tokens_buckets": [512, 1024, 2048, 4096, 8192],
                      "num_seqs_buckets": [1, 2, 4, 8, 16, 32, 48, 64],
                      "decode_context_length_buckets": [4096],
                      "on_device_sampling_config": {"all_greedy": true}},
    "vision_neuron_config": {"vision_attention_block_size": 4096,
                             "num_vision_tokens_buckets": [1024, 2048, 3072, 4096],
                             "encoder_cache_num_blocks": 16, "tp_size": 4}}' \
  --port 8000
```

The first start compiles every graph and takes a while; later starts reuse `~/.cache/neuron_libtorch`. Test it with a local image:

```bash
python - <<'PY'
import base64, io, json, urllib.request
from PIL import Image, ImageDraw
img = Image.new("RGB", (896, 672), "white"); d = ImageDraw.Draw(img)
d.ellipse((100, 100, 400, 400), fill="red"); d.rectangle((500, 200, 800, 550), fill="blue")
buf = io.BytesIO(); img.save(buf, format="PNG")
url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
body = {"model": "muse", "max_tokens": 128, "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "Describe this image."}]}]}
req = urllib.request.Request("http://localhost:8000/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req))["choices"][0]["message"]["content"])
PY
```

**BF16 instead of FP8** (output closest to the HF reference):
- Remove `MUSE_GLIMMER_FP8`, `MUSE_GLIMMER_KV_SCALES` and `--kv-cache-dtype fp8`.
- Set `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.15` and `--max-num-seqs 16`.
- Use `"num_seqs_buckets": [1, 2, 4, 8, 16]` and `"num_vision_tokens_buckets": [4096]`.

BF16 has room for 16 sequences: 27 ms per output token, 184-455 peak output tok/s.

**Settings that matter:**
- **KV cap.** The default `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` (0.30) leaves no device memory for the compiled graphs, and the server fails to load. Use the values above.
- **Validate before changing caps or buckets.** If you change the KV cap, `max_num_seqs` or the bucket lists, run a load test that touches every bucket before trusting it. Runtime DMA buffers are allocated the first time each graph runs, so a short test can pass while full load fails with `status=4`.
- **`VLLM_NEURON_SWA_ALIGN_TOKENS=4096`** makes sliding-window decode move larger KV chunks (+12-14 % throughput under load). Keep it below `--max-model-len`.
- **Videos** can use up to 96 frames at `--max-model-len 8192`. They need `"video": 1` in `--limit-mm-per-prompt` and a vision bucket large enough for the frame count. See the model README.

## 6. Optional: DFlash speculative decoding (lower latency at low concurrency)

DFlash was validated with vLLM's offline engine. This is the configuration that was measured:

```python
import os
os.environ.update(MUSE_GLIMMER_FP8="mlp", VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION="0.10",
                  VLLM_NEURON_SWA_ALIGN_TOKENS="4096", VLLM_NEURON_BARRIER_TIMEOUT="14400",
                  NEURON_SKIP_EFA_AFFINITY="1")
from vllm import LLM, SamplingParams

llm = LLM(model=os.path.expanduser("~/models/Muse-Glimmer-30B"), tensor_parallel_size=4,
          max_model_len=8192, max_num_batched_tokens=2048, max_num_seqs=8, enable_prefix_caching=False,
          speculative_config={"method": "dflash", "num_speculative_tokens": 15,
                              "model": os.path.expanduser("~/models/Muse-Glimmer-30B-assistant")},
          additional_config={"mm_language_model_only": True,
                             "neuron_config": {"num_batched_tokens_buckets": [512, 2048],
                                               "num_seqs_buckets": [1, 4, 8],
                                               "on_device_sampling_config": {"all_greedy": True}}},
          limit_mm_per_prompt={"image": 0, "video": 0})
out = llm.generate(["Explain how a heat pump works."], SamplingParams(max_tokens=256, temperature=0))
print(out[0].outputs[0].text)
```

On natural text the drafter averages about 4.1 accepted tokens per step (256 output tokens):

| Concurrency | FP8 MLP | + DFlash |
|---|---|---|
| 1 | 48 tok/s | 133 tok/s |
| 4 | 131 tok/s | 291 tok/s |
| 8 | 292 tok/s | 355 tok/s |

**Scope:**
- **Latency, not maximum throughput.** DFlash runs with a smaller KV cap and stops helping above about 8 concurrent requests.
- **BF16 instead of FP8:** drop `MUSE_GLIMMER_FP8` and use cap 0.08.
- **Text only:** images with DFlash have not been tested.
- **Offline engine only:** `vllm serve` with DFlash was not separately validated.
- **Long prompts:** the vllm-neuron branch block-aligns prefill chunks under speculative decoding. Without that, prompts longer than one chunk gave wrong output.

## Accuracy

Greedy output was compared with Hugging Face transformers in BF16 on 66 prompts (43 text, 17 image, 6 video), 16 tokens each:

| Config | Prompts with all 16 tokens equal to HF |
|---|---|
| BF16 | 66/66 |
| FP8 MLP | 61-63/66 |
| FP8 MLP + FP8 KV (recommended) | 58-59/66 |

The first token matched on every prompt in every config, and every divergence was a near-tie: HF's top two tokens were within the FP8 noise. FP8 is close to BF16 but not identical, so run a task-level eval on your data before relying on it.

## Limitations

- Tested only on trn2.3xlarge at TP=4, `--max-model-len` 8192, with prefix caching off.
- DFlash has not been tested together with the FP8 KV cache or with images.
- Changing the KV cap or `--max-num-seqs` recompiles every graph.
