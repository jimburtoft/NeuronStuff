# Moshi / PersonaPlex full-duplex speech on Trainium2 with vLLM-Omni (Neuron SDK 2.33)

Serve kyutai's [Moshi](https://github.com/kyutai-labs/moshi) (`kyutai/moshiko-pytorch-bf16` or
`kyutai/moshika-pytorch-bf16`) and NVIDIA's [PersonaPlex](https://huggingface.co/nvidia/personaplex-7b-v1)
(`nvidia/personaplex-7b-v1`: the same architecture with a voice prompt and a text persona) on a trn2.3xlarge as a
**full-duplex, real-time voice model** behind vLLM-Omni's OpenAI-Realtime-style WebSocket API
(`/v1/realtime?duplex=1`). The model listens and talks at the same time in 80 ms frames: there is no turn detection,
no ASR -> LLM -> TTS cascade.

- **Environment:** the public Neuron SDK 2.33 DLAMI, its vLLM-Omni venv (vllm 0.29.0, vllm-omni 0.29.0rc1,
  vllm-omni-neuron 0.29.0.0.2.0) unchanged, plus the `contrib/moshi` directory of
  [jimburtoft/vllm-omni-neuron](https://github.com/jimburtoft/vllm-omni-neuron/tree/contrib/moshi/contrib/moshi)
  on `PYTHONPATH`.
- **Model:** 7.7B (Helium-7B temporal transformer + depth transformer, BF16) and the Mimi codec (FP32). Moshi
  weights CC-BY 4.0; PersonaPlex under NVIDIA's license (gated on Hugging Face).
- **Endpoint:** `ws://<host>:8000/v1/realtime?duplex=1` (24 kHz float32 PCM in, 24 kHz 16-bit PCM out).

## Results (one trn2.3xlarge: 1 Trainium2 device, 12 vCPUs)

**PersonaPlex, throughput layout** (TP4 over the device's 4 NeuronCores, FP8 KV cache, 1500-frame attention window as
in the PersonaPlex reference; voice + persona prompt per session; sessions arriving 1 s apart):

| Session length | Concurrent sessions | Every frame on time | Per-frame response latency, p50 / p99 |
|---|---|---|---|
| 4 minutes | 36 | yes | 36-47 / 248 ms |
| 4 minutes | **40** | yes | **56-64 / 112 ms** |

Above 40 sessions the limit is the vLLM-Omni API server process on the host, not the Trainium device (which keeps
up with 44-48 sessions at this point).

**With upcoming vLLM-Omni** (its unified full-duplex framework, pending duplex CPU fixes and the option to run the
duplex orchestrator in its own process) and the same Neuron workers, on a trn2.3xlarge, benchmark client on the same
instance, API server and orchestrator process each pinned to their own CPU core:

| Session length | Concurrent sessions | Every frame on time | Per-frame response latency, p50 / p99 |
|---|---|---|---|
| 4 minutes | 56 | yes (2 runs) | 67-95 / 546-743 ms |
| 4 minutes | 60 | yes | 63-79 / 140 ms |
| 4 minutes | 64 | yes (3 of 3 runs) | 59-109 / 221-451 ms |
| 4 minutes | 72 | yes (3 of 3 runs) | 53-114 / 106-777 ms |
| 4 minutes | 76 | yes (2 of 2 runs) | 45-59 / 88-466 ms |
| 4 minutes | **80** | **yes (4 of 4 runs)** | **54-97 / 123-515 ms** |
| 4 minutes | 84 | 2 of 5 runs (the edge) | 55-72 / 518-988 ms on the clean runs |

How the rows were reached:
- **60 and 64 sessions:** host-side changes in our stage-0 path.
  - Per-step messages to the Neuron workers no longer grow with conversation length.
  - User audio is sent as one packed tensor per step.
  - Hand-off between stages is batched.
- **72 sessions:** PersonaPlex runs as a single pipeline stage. The Neuron workers decode the audio and hand it
  straight to the realtime front end; user audio arrives as raw samples.
- **76-84 sessions:** two device-side settings cut the model step at full load by about 7 ms.
  - `MOSHI_DEP_NKI_MINB=1`: the NKI depth-transformer kernel.
  - `MOSHI_KV_BLOCK=12` with context buckets `768` (+ the full window): the attention kernel keeps 12-position KV
    blocks at the 1536-slot window instead of splitting 32-slot blocks into 4-position pieces, so its KV reads are 3x
    larger.

Above 80 the limit is the device step: about 66-68 ms of each 80 ms frame at 84 sessions, plus the host work around
it. Quality at 84 sessions:
- Teacher-forced against the fp32 reference: text 99.3 %, audio codes 97.8 %.
- Whisper's transcript of the generated speech vs the model's own text: WER 0.110 mean / 0.105 median (0.118 / 0.114
  at 40 sessions with a BF16 KV cache).

With a BF16 KV cache instead of FP8, the same two settings give **64 sessions** (2 of 2 runs; 68 does not keep up).

**Moshi (moshiko / moshika)**, 3000-frame window, 28.8 s conversation streamed at real-time pace:

| Layout | Concurrent sessions | Every frame on time | Per-frame response latency, p50 / p99 |
|---|---|---|---|
| Latency: 3 language-model cores + 1 decoder core | 15 | yes | 83-118 / 193 ms (1 session: 54 / 60 ms) |
| Throughput: TP4 over all 4 cores | 24 | yes | 194-214 / 401-539 ms (1 session: 44 / 83 ms) |

Per-frame response latency is the time from sending an 80 ms input frame to receiving the 80 ms of audio it
produces; it does not grow over the session.

Quality: with greedy decoding the Trainium output reproduces the kyutai reference implementation's text on our
test clips; teacher-forced against the fp32 reference, text predictions match 100% and audio-code predictions
99.5% (bf16 KV; FP8 KV costs ~1.4 points on audio codes, text unchanged). Whisper's transcript of the generated
speech matches the model's inner-monologue text.

## How it works

vLLM-Omni already runs Moshi-architecture models (its PersonaPlex pipeline) as a two-stage duplex pipeline on GPUs.
vllm-omni-neuron 0.29 has no Neuron worker for those stages, so `contrib/moshi` adds them. In the throughput
layout one stage-0 worker owns the whole device:

| Stage | Where | Work per 80 ms frame, for every live session |
|---|---|---|
| 0 | 4 NeuronCores (TP4) | Mimi-encode the user's 1920 samples, one compiled graph (Helium step + depth-transformer steps + sampling), Mimi-decode the reply |
| 1 | host | pass the PCM through to the vLLM-Omni duplex front end |

Every session's streaming state (Mimi convolution buffers, transformer KV caches) stays in device memory, so only
audio and a few tokens cross the host per frame. The temporal transformer's attention runs on the fused decode
kernel from the Neuron Kernel Library (`attention_block_tkg`, shipped in the DLAMI) over a block-paged ring KV cache;
everything else is PyTorch compiled by the Neuron compiler.

## 1. Launch an instance

- **Instance:** `trn2.3xlarge` (1 Trainium2 device, 4 logical NeuronCores at the default LNC=2).
- **AMI:** *Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0* (multi-framework). Latest ID:
  ```bash
  aws ssm get-parameter --region <region> \
    --name /aws/service/neuron/dlami/multi-framework/ubuntu-24.04/latest/image_id \
    --query Parameter.Value --output text
  ```
- **Disk:** 300 GB or more.

## 2. Add swap (once per instance)

```bash
sudo fallocate -l 64G /swapfile && sudo chmod 600 /swapfile
sudo mkswap --force /swapfile && sudo swapon /swapfile
```

## 3. Get the workers

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_omni_0_29_0_0_2_0/bin/activate
git clone -b contrib/moshi https://github.com/jimburtoft/vllm-omni-neuron.git
export PYTHONPATH=$PWD/vllm-omni-neuron/contrib/moshi NEURON_SKIP_EFA_AFFINITY=1
```

Activate the venv (don't just call its `python`): the Neuron compiler is found through `PATH`.

## 4a. Serve PersonaPlex (40 sessions)

`nvidia/personaplex-7b-v1` is gated on Hugging Face: request access and log in (`huggingface-cli login`) first.

```bash
PP=$(python -c "from huggingface_hub import snapshot_download as s; print(s('nvidia/personaplex-7b-v1'))")
export PYTHONPATH=$PYTHONPATH:$PWD/vllm-omni-neuron/contrib/moshi/scripts/fs_site
export MOSHI_CONTEXT=1500 MOSHI_KV_SLOTS=1536 MOSHI_CTX_BUCKETS=512,1024 MOSHI_BUCKETS=24,32,40,48 \
  MOSHI_FP8_KV=1 MOSHI_FP8_PACKED=1 MOSHI_ENC_SHARD=1 MOSHI_DEP_RUN=8 \
  MOSHI_EMB_INPUTS=1 MOSHI_PP_PREFILL=1 \
  MOSHI_FRAMESYNC=1 MOSHI_FS_PERIOD_MS=60 MOSHI_WORKER_THREADS=1 MOSHI_GC_FULL=0 \
  MOSHI_FAST_APPEND=1 MOSHI_CONS_FIX=1 MOSHI_MARKS_FIX=1 MOSHI_WS_NO_DEFLATE=1 \
  MOSHI_STAGE_EVENTS_CAP=64 MOSHI_MARK_COALESCE=1 MOSHI_SNAP_FIX=1 \
  MOSHI_PCM_FAST=1 MOSHI_SHM_PROBE=1 MOSHI_API_TORCH_THREADS=4 \
  MOSHI_S0_DECODE=1 MOSHI_MLP_XW=1 MOSHI_NC_COUNT=4 \
  NEURON_RT_XU_WORKER_ENABLE_LOW_LATENCY_COMPLETION=0 MOSHI_RESP_SLEEP_MS=0.5
MOSHI_B=48 MOSHI_TP0=4 MOSHI_DEV0=0,1,2,3 MOSHI_DEV1=0 python -m moshi_neuron.serving.make_deploy /tmp/pp_tp4.yaml
python -m moshi_neuron.serving.serve_launcher serve $PP --deploy-config /tmp/pp_tp4.yaml --port 8000 --omni \
    --init-timeout 10800 --stage-init-timeout 10800
```

The first start compiles every batch-size x context-length graph (about 2 hours); wait for
`Application startup complete`. Later starts load them from the compile cache. What the main switches do is
documented in the [contrib README](https://github.com/jimburtoft/vllm-omni-neuron/tree/contrib/moshi/contrib/moshi).

## 4b. Serve Moshi (15 sessions, low latency)

```bash
MOSHI_B=5 MOSHI_REP0=3 MOSHI_DEV0=0,2,3 MOSHI_DEV1=1 python -m moshi_neuron.serving.make_deploy /tmp/moshi15.yaml
MOSHI_MLP_XW=1 MOSHI_BUCKETS=1,2,3,4,5 MOSHI_DEC_BUCKETS=1,2,4,8,15 MOSHI_CTX_BUCKETS=512,1024,2048 MOSHI_NC_COUNT=4 \
python -m moshi_neuron.serving.serve_launcher serve kyutai/moshiko-pytorch-bf16 \
    --deploy-config /tmp/moshi15.yaml --port 8000 --omni --init-timeout 3600 --stage-init-timeout 3600
```

The first start compiles the three language-model replicas one after another (about 20 minutes). The 24-session
throughput layout for Moshi (TP4) is in the contrib README.

## 5. Talk to it

Stream a 24 kHz (or any rate; it is resampled) wav file at real-time pace and record the reply:

```bash
python -m moshi_neuron.serving.duplex_client --wav my_question.wav --sessions 1 --out out/
# PersonaPlex: add a voice prompt and a persona
python -m moshi_neuron.serving.duplex_client --wav my_question.wav --sessions 1 --out out/ \
    --voice NATM1.pt --instructions "You work at a restaurant. Take the diner's order and answer questions about the menu."
```

`out/session0.wav` is the model's side of the conversation, time-aligned with your input, and `out/session0.txt`
is its inner-monologue text (sent as `text` deltas on the `response.output_audio.delta` events). `--sessions N`
opens N concurrent conversations (`--stagger-s 1` spreads their start). Any client of the vLLM-Omni realtime duplex
protocol works, for example `vllm_omni.clients.duplex.DuplexClient` with
`SessionConfig(input_audio=AudioFormat("pcm_f32le", 24000), output_audio=AudioFormat("pcm16", 24000))`.
Moshi takes no `voice` or `instructions`.

## Notes

- The installed vllm-omni 0.29 builds a PersonaPlex voice/persona prompt for every session, which Moshi
  checkpoints don't have. The launcher applies a small runtime patch (the same rule as upstream vllm-omni
  pull request [#9319](https://github.com/vllm-project/vllm-omni/pull/9319)): Moshi sessions start with no prompt.
- Sessions can be longer than the attention window (the KV caches are rings); the server caps them at 1 hour
  (`MOSHI_MAX_MODEL_LEN`, in 80 ms frames). Moshi itself tends to repeat or go quiet in very long one-sided
  monologues (the kyutai reference does the same).
- The capacity numbers were measured with the benchmark client on the same instance.
