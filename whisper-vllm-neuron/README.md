# Whisper large-v3 on Trainium2 with vllm-neuron (Neuron SDK 2.33)

Serve [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) with vLLM on a trn2.3xlarge,
behind the standard OpenAI-compatible `/v1/audio/transcriptions` endpoint. The environment is the public
Neuron SDK 2.33 DLAMI plus the Whisper model from our vllm-neuron branch
[`jimburtoft/vllm-neuron` `whisper/0.29-public`](https://github.com/jimburtoft/vllm-neuron/tree/whisper/0.29-public).

- **Model:** `openai/whisper-large-v3`, BF16. Native vllm-neuron model (no NxD Inference).
- **Optional drafts for speculative decoding:** [`distil-whisper/distil-large-v3`](https://huggingface.co/distil-whisper/distil-large-v3)
  (2 decoder layers, the same encoder as large-v3), or the DFlash block draft
  [`jburtoft/whisper-large-v3-DFlash`](https://huggingface.co/jburtoft/whisper-large-v3-DFlash) (proposes 15 tokens in one
  pass; lowest single-request latency; training code [jimburtoft/whisper-dflash](https://github.com/jimburtoft/whisper-dflash)).
- **Configs:** one for throughput (8 replicas, one per NeuronCore), one for single-request latency (tensor parallel 4),
  each with and without speculative decoding.

Why the extra steps: vllm-neuron 0.29.0.1.2.0, the version in the SDK 2.33 DLAMI, has no Whisper model.
The branch is the public `v0.29.0.1.2.0` release plus the Whisper model, two NKI decode kernels (cross-attention and
self-attention over a per-request KV cache), greedy draft-model and DFlash speculative decoding for Whisper, and these framework
fixes that Whisper needs:

| Fix | What it does |
|---|---|
| GQA decode-mask layout | The fused NKI decode kernel folds the KV heads into the batch; the mask now gets the same layout. Without it, some batch sizes give wrong output and nothing fails. |
| TP1 NKI argmax | Compiles the TP=1 on-device argmax at batch size 2 or more. |
| LNC-aware launch grid | `flash_attention`, `qkv`, `o_proj` and the multimodal merge kernel launch with the configured LNC, so they compile at LNC=1. |
| Per-API-worker `SO_REUSEPORT` | Each internal-DP API server gets its own listener, so requests spread evenly over the replicas. |
| Compile-cache key | The compile cache keys on the KV pool size and the `WHISPER_*` settings, and compiles cold when a restore fails. |
| Spec-decode position correction | Keeps padded rows in range in the async position correction. |

You install the branch into the DLAMI vLLM environment in place of the shipped plugin; nothing else in the
environment changes.

## Results (one trn2.3xlarge = one Trainium2 device)

Public data only: Earnings22 ([`ArtificialAnalysis/Earnings22-Cleaned-AA`](https://huggingface.co/datasets/ArtificialAnalysis/Earnings22-Cleaned-AA),
512 x 30 s windows, and the first 8 s of each), LibriSpeech test-clean / test-other, and an 18-clip latency set
from LibriSpeech and Earnings22 (`scripts/manifest.csv`, 3 clips per 5 s length bucket, 0-30 s).
RTFx = seconds of audio transcribed per second of wall time.

| Config | What | Result |
|---|---|---|
| `tput` | Throughput, BF16, 8 replicas (TP1 x DP8, LNC=1), 22 requests per replica, concurrency 256 | **1,239 RTFx on 30 s clips, 514 on 8 s clips** |
| `tput-spec` | `tput` + speculative decoding (K=4), 16 requests per replica | 1,275 RTFx on 30 s clips; lower latency below the knee (E2E p50 559 ms vs 828 ms at concurrency 16) |
| `lat` | One request at a time, TP4 (LNC=2) | **256 ms** mean over the 18 clips (p50 197, p90 487) |
| `lat-spec` | `lat` + speculative decoding (K=10, async) | **98.6 ms** mean (p50 86, p90 156); 8.8 tokens per verify step (7.8 accepted draft tokens + 1) |
| `lat-dflash` | `lat` + DFlash speculative decoding (block 16, async) | **89.6 ms** mean (p50 72, p90 140); 7.1 tokens per verify step. Same session: `lat` 249.9 ms, `lat-spec` 95.3 ms |
| `tput-fp8` (option) | `tput` + FP8 (e4m3) cross-attention KV cache, 36 requests per replica. **Precision change.** | 1,417 / 558 RTFx |

Accuracy (WER %, open_asr_leaderboard normalizer, `tput` config):

| Earnings22 | LibriSpeech test-clean | LibriSpeech test-other |
|--:|--:|--:|
| 8.56 | 1.49 | 3.14 |

- 16 of the 18 latency clips give the same tokens as Hugging Face transformers in FP32 (greedy). The other two are
  BF16-vs-FP32 near-ties (one capitalized first word, one dropped final word).
- Under load (the 18 clips x 4 at concurrency 128), every transcript is the same as at one request at a time.
- With speculative decoding, the 18 clips give the same tokens as without it. On larger sets, a few near-ties
  flip (about 1 in 15 Earnings22 chunks); WER does not change.
- `lat-dflash` WER (full sets): Earnings22 8.65, LibriSpeech test-clean 1.47, test-other 3.13 (the `lat` server: 8.52 / 1.47 /
  3.12; `lat-spec`: 8.65 / 1.47 / 3.12). DFlash accepts the most tokens on LibriSpeech (about 9 per verify step on test-clean)
  and the fewest on Earnings22 chunks (about 4.5). It is a latency option: in the 8-replica throughput layout it runs
  but is slower than `tput` (530 vs 1,256 RTFx at concurrency 256), so use `tput` or `tput-spec` there.

## 1. Launch an instance

- **Instance:** `trn2.3xlarge` (1 Trainium2 device, 4 logical NeuronCores at the default LNC=2, 8 at LNC=1).
- **AMI:** *Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0* (multi-framework). Latest ID:
  ```bash
  aws ssm get-parameter --region <region> \
    --name /aws/service/neuron/dlami/multi-framework/ubuntu-24.04/latest/image_id \
    --query Parameter.Value --output text
  ```
  (us-east-1: `ami-091af6c736073d840`, `20261020`.)
- **Disk:** 700 GB. The throughput config compiles 8 replicas, and each keeps its own compile artifacts.

## 2. Add swap (once per instance)

The DLAMI ships without swap. The first start of the 8-replica config compiles 8 copies of the model at once and
needs more host memory than the instance has.
```bash
sudo fallocate -l 64G /swapfile && sudo chmod 600 /swapfile && sudo mkswap --force /swapfile && sudo swapon /swapfile
sudo fallocate -l 96G /swapfile2 && sudo chmod 600 /swapfile2 && sudo mkswap --force /swapfile2 && sudo swapon /swapfile2
```

## 3. Install the Whisper branch into the DLAMI vLLM environment

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate   # vllm 0.29.0, vllm-neuron 0.29.0.1.2.0

git clone https://github.com/jimburtoft/vllm-neuron.git -b whisper/0.29-public ~/vllm-neuron
git -C ~/vllm-neuron checkout aa283ad906
pip install --no-deps --no-build-isolation -e ~/vllm-neuron      # replaces the shipped plugin; nothing else changes

# vLLM's "audio" extra is not in the DLAMI venv. Without it every transcription request returns HTTP 400.
# Install it in a side directory without dependencies, so the venv's numpy and numba stay as shipped.
pip install --target ~/vn_audio --no-deps av soundfile soxr librosa audioread lazy_loader pooch msgpack standard-aifc standard-sunau
export PYTHONPATH=~/vn_audio

python -c "import vllm_neuron; print(vllm_neuron.__file__)"   # must print ~/vllm-neuron/vllm_neuron/__init__.py
```

Set `PYTHONPATH` the same way in every shell that runs `vllm serve`.

## 4. Serve

All configs use these settings:
```bash
export NEURON_SKIP_EFA_AFFINITY=1           # trn2.3xlarge has no EFA device
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_READY_TIMEOUT_S=10800    # the first start of the 8-replica configs compiles for 30-60 min
export VLLM_NEURON_PARALLEL_COMPILE_WORKERS=2   # per replica; the default runs out of host memory with 8 replicas
export NEURON_RT_XU_WORKER_ENABLE_LOW_LATENCY_COMPLETION=0   # frees ~0.75 vCPU per replica: +2.6-3.2 % throughput
```

The first start of each config downloads the model and compiles it (about 10 min for `lat` and `lat-spec`, 30-60 min
for the 8-replica configs). Later starts reuse the compiled graphs: about 3 min for `lat`, and about 20 min for the
8-replica configs (each replica still traces its graphs at start). Wait for `curl -sf localhost:8000/health` to succeed.

### Throughput (`tput`)

```bash
export NEURON_LOGICAL_NC_CONFIG=1 OMP_NUM_THREADS=1
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.10 VLLM_NEURON_MIN_KV_BUDGET_GIB=0
export WHISPER_FUSED_DECODE=1 WHISPER_FUSED_DECODE_MAX_B=32 WHISPER_ENCODER_FLASH=1 WHISPER_SELF_KV=slot \
  WHISPER_CROSS_MODE=xkernel WHISPER_CROSS_KV_FUSED=1 WHISPER_IN_OUT=decoder,lm_head,encoder \
  WHISPER_SELF_ATTN=kernel WHISPER_LM_HEAD=bf16w

vllm serve openai/whisper-large-v3 --served-model-name whisper \
  --tensor-parallel-size 1 --data-parallel-size 8 --max-num-seqs 22 --max-model-len 448 \
  --max-num-batched-tokens 1536 --dtype bfloat16 --no-enable-prefix-caching --async-scheduling \
  --mm-processor-cache-gb 0 --limit-mm-per-prompt '{"audio": 1}' \
  --num-gpu-blocks-override 336 --gpu-memory-utilization 0.98 \
  --additional-config '{
    "neuron_config": {"quantization": "bf16", "num_batched_tokens_buckets": [16, 448],
                      "num_seqs_buckets": [1, 2, 4, 8, 16, 22], "decode_context_length_buckets": [128, 256],
                      "on_device_sampling_config": {"all_greedy": true}},
    "encoder_neuron_config": {"num_encoder_input_buckets": [1500], "encoder_attention_block_size": 1500,
                              "tp_size": 1, "dp_size": 1, "encoder_cache_num_blocks": 24}}' \
  --port 8000
```

- `--num-gpu-blocks-override` sizes vLLM's paged KV pool to the requests: the model keeps its own per-request KV
  cache (`WHISPER_SELF_KV=slot`), so a larger pool only takes memory. Use `(max-num-seqs + 2) * 14`.
- `encoder_cache_num_blocks` is `max-num-seqs + 2`.

**Speculative decoding (`tput-spec`):** the same, with `--max-num-seqs 16`, `"num_seqs_buckets": [1, 2, 4, 8, 16]`,
`--num-gpu-blocks-override 252`, `"encoder_cache_num_blocks": 18`, and
```bash
  --speculative-config '{"method": "draft_model", "model": "distil-whisper/distil-large-v3", "num_speculative_tokens": 4}'
```
It gives about the same peak throughput as `tput` on 30 s clips (2-3 % higher) and lower latency at moderate load; on
8 s clips `tput` is 2-3 % faster.

**FP8 cross-attention KV (`tput-fp8`, precision change):** add `export WHISPER_CROSS_FP8=1`, and use
`--max-num-seqs 36`, `"num_seqs_buckets": [1, 2, 4, 8, 16, 32, 36]`, `--num-gpu-blocks-override 532`,
`"encoder_cache_num_blocks": 38`. WER 8.65 / 1.50 / 3.11.

### Single-request latency (`lat`)

```bash
export NEURON_LOGICAL_NC_CONFIG=2 OMP_NUM_THREADS=4
export WHISPER_FUSED_DECODE=1 WHISPER_ENCODER_FLASH=1 WHISPER_LM_HEAD=bf16w \
  WHISPER_IN_OUT=decoder,lm_head,encoder WHISPER_CROSS_KV_FUSED=1

vllm serve openai/whisper-large-v3 --served-model-name whisper \
  --tensor-parallel-size 4 --max-num-seqs 1 --max-model-len 448 \
  --max-num-batched-tokens 1536 --dtype bfloat16 --no-enable-prefix-caching --async-scheduling \
  --mm-processor-cache-gb 0 --limit-mm-per-prompt '{"audio": 1}' \
  --additional-config '{
    "neuron_config": {"quantization": "bf16", "num_batched_tokens_buckets": [16, 448],
                      "on_device_sampling_config": {"all_greedy": true}},
    "encoder_neuron_config": {"num_encoder_input_buckets": [1500], "encoder_attention_block_size": 1500,
                              "tp_size": 4, "dp_size": 1, "encoder_cache_num_blocks": 3}}' \
  --port 8000
```

**Speculative decoding (`lat-spec`):** add
```bash
  --speculative-config '{"method": "draft_model", "model": "distil-whisper/distil-large-v3", "num_speculative_tokens": 10}'
```
Keep `num_speculative_tokens` below the smallest `num_batched_tokens_buckets` value minus 1 (16 here, so at most 14).

**DFlash speculative decoding (`lat-dflash`):** add, instead of the `draft_model` config,
```bash
  --speculative-config '{"method": "dflash", "model": "jburtoft/whisper-large-v3-DFlash", "num_speculative_tokens": 15}'
```
`num_speculative_tokens` must be 15 (the draft's block size 16 minus 1); a verify step is 16 tokens, which the 16-token bucket
covers. The draft also attends to the encoder output (its "audio context"), computed once per request after the encoder.

## 5. Transcribe

```bash
curl -s localhost:8000/v1/audio/transcriptions -F model=whisper -F file=@clip.wav \
  -F language=en -F temperature=0
```
Or with the OpenAI client:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
with open("clip.wav", "rb") as f:
    print(client.audio.transcriptions.create(model="whisper", file=f, language="en", temperature=0).text)
```
Pass `language`: without it, Whisper first runs a language-detection step for each request.
Audio longer than 30 s is not split by the server: send 30 s chunks.

## 6. Reproduce the numbers

The scripts are in [`scripts/`](scripts/). They are clients only and run against the server on port 8000. Run them
in a second shell on the same instance, in the DLAMI vLLM environment, from the directory that contains `scripts/`:
```bash
git clone https://github.com/jimburtoft/NeuronStuff.git ~/NeuronStuff && cd ~/NeuronStuff/whisper-vllm-neuron
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
```
Install the client packages in a separate directory, without dependencies (the venv already has the rest; letting pip
resolve dependencies here pulls a newer numpy and huggingface-hub that break transformers):
```bash
pip install --target ~/vn_client --no-deps datasets pyarrow xxhash multiprocess kaldialign num2words docopt duckdb more_itertools
export PYTHONPATH=~/vn_audio:~/vn_client
```

**Latency set and identity check.** `build_pub18.py fetch` downloads the 18 clips by dataset ID into the manifest's
directory; `ref_pub18.json` is the Hugging Face transformers FP32 greedy output for them (`hf_reference.py` re-creates it).
```bash
mkdir -p ~/pub18 && cp scripts/manifest.csv ~/pub18/
python scripts/build_pub18.py fetch --manifest ~/pub18/manifest.csv
python scripts/lat_bench.py --manifest ~/pub18/manifest.csv --reps 3 --warm 1 --out lat.json
python scripts/cmp_ref.py scripts/ref_pub18.json lat.json            # token-level identity, per clip
python scripts/conc_check.py --manifest ~/pub18/manifest.csv --ref lat.json --conc 128 --reps 4   # same text under load
```
`conc_check.py` works against any of the configs; keep `lat.json` from the `lat` server as the reference.
`lat_stats.py ~/pub18/manifest.csv lat.json` prints the mean / p50 / p90 per length bucket.

**Throughput corpora.** `build_perf_corpus.py` builds the 512 x 30 s Earnings22 windows, and `build_short_corpus.py` the 8 s
clips from them:
```bash
python scripts/build_perf_corpus.py && python scripts/build_short_corpus.py
python scripts/tput_bench.py --max-tok 444 --conc 256 --num 2048 \
  --manifest ~/.cache/vllm-neuron/datasets/asr30s-earnings22-v1/manifest_512.jsonl --out tput30.json
python scripts/tput_bench.py --max-tok 444 --conc 256 --num 2048 \
  --manifest ~/.cache/vllm-neuron/datasets/asr8s-earnings22-approx/manifest_512.jsonl --out tput8.json
```

**WER.** `wer_public.py` uses the open_asr_leaderboard data loading and normalizer:
```bash
git clone https://github.com/huggingface/open_asr_leaderboard ~/oal && git -C ~/oal checkout 48e6f0d75ad7
for d in earnings22 libri-clean libri-other; do python scripts/wer_public.py 8000 128 wer_$d $d; done
```

## Notes

- **Software:** Neuron SDK 2.33 DLAMI (`20261020`): vllm 0.29.0, torch / torch-xla 2.13.0, neuronx-cc 2.28.7631.0,
  nki 0.7.0; branch `whisper/0.29-public` @ `aa283ad906` on top of vllm-neuron `v0.29.0.1.2.0` (the `tput`, `lat` and
  `lat-spec` numbers were measured at `f95e5739b1`; `aa283ad906` adds DFlash only, and `lat` / `lat-spec` were re-measured on it
  with the same results within run-to-run noise).
- **`max_model_len 448`** is Whisper's decoder limit. The prompt is 4 tokens, so a request can produce up to 444 tokens.
- **Throughput layout:** TP1 x DP8 at LNC=1 is the fastest layout on one trn2.3xlarge. The other layouts we measured
  (TP2 x DP4 and TP4 x DP2 at LNC=1; TP1 x DP4, TP2 x DP2 and TP4 at LNC=2) are 7-55 % slower.
- **FP8 decoder MLP weights** (`WHISPER_FP8=decoder_mlp`) exist on the branch but are slower; do not use them.
- **Startup:** the first request after a start can take longer while the runtime warms up; send one warm-up request.
