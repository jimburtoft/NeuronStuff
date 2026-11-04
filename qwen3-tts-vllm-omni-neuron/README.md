# Qwen3-TTS on Trainium2 with vLLM-Omni (Neuron SDK 2.33)

Serve Alibaba's [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 12Hz 1.7B text-to-speech models on a trn2.3xlarge
behind vLLM-Omni's OpenAI-compatible speech API (`/v1/audio/speech`, with streaming). All three checkpoints work:

| Checkpoint | `task_type` | What you send |
|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `VoiceDesign` | text + a natural-language voice description (`instructions`) |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | `CustomVoice` | text + a preset `voice` (aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian), optional `instructions` |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `Base` | text + a reference clip (`ref_audio`, plus `ref_text` for in-context cloning) |

- **Environment:** the public Neuron SDK 2.33 DLAMI with its vLLM-Omni venv (vllm 0.29.0, vllm-omni 0.29.0rc1,
  vllm-omni-neuron 0.29.0.0.2.0) unchanged. Add the `contrib/qwen3_tts` directory of
  [jimburtoft/vllm-omni-neuron](https://github.com/jimburtoft/vllm-omni-neuron/tree/contrib/qwen3-tts/contrib/qwen3_tts)
  to `PYTHONPATH`.
- **Model:** Qwen3-TTS 12Hz 1.7B. It has a 28-layer talker that predicts the first codebook, a 5-layer code predictor
  for the other 15 codebooks, and a 12.5 Hz -> 24 kHz codec decoder. BF16. Weights Apache-2.0.
- **Output:** 24 kHz mono. WAV, or 16-bit PCM when streaming.

## How it works

vLLM-Omni runs Qwen3-TTS as a two-stage pipeline: a talker that generates codec frames, and code2wav that turns
frames into audio. vllm-omni-neuron 0.29 has no Neuron worker for those stages, so `contrib/qwen3_tts` adds two:

| Stage | Per NeuronCore | What runs on Trainium2 |
|---|---|---|
| 0 talker | up to 48 concurrent requests | one compiled graph per 80 ms audio frame: talker decode, the 15-step code predictor, and per-request sampling, for all requests at once |
| 1 code2wav | shared by many requests | the codec decoder; its high-rate upsampling blocks run as NKI kernels |

- The vLLM-Omni scheduler, connector and OpenAI server are used unchanged. Prompt construction reuses vLLM-Omni's
  own prompt builder, so request fields are the same as on any other vLLM-Omni deployment.
- Every request's KV cache and decoder state stay in device memory.

## Results (trn2.3xlarge, whole instance)

VoiceDesign, default sampling, ~4.4 s utterances. Measured with vLLM-Omni's own TTS benchmark
(`vllm bench serve --omni --backend openai-audio-speech --dataset-name seed-tts-design`, the bundled prompt set).
Throughput is seconds of audio generated per second.

| Setup | Concurrent requests | Audio throughput | Request latency p50 | First audio p50 |
|---|---|---|---|---|
| 1 server, LNC=2 | 1 | 2.7x real time | 1.6 s | 60 ms |
| 1 server, LNC=2 | 128 | 176x | 2.8 s | -- |
| 1 server, LNC=2 | 256 | 186x | 5.7 s | -- |
| 2 servers, LNC=1 | 200 (100 each) | 262x | 3.0 s | 260 ms |
| **2 servers, LNC=1** | **400 (200 each)** | **367-373x** | **4.2-4.3 s** | 1.0 s |
| 2 servers, LNC=1 | 600 (300 each) | 349x | 6.3 s | -- |

- **Sizing by latency** (2 servers at LNC=1, random arrivals at a fixed rate): up to about 80 requests/s (341x real
  time) first audio stays under 400 ms at p99 and request latency under 5.3 s at p90. The instance saturates at about
  85 requests/s (~385x); beyond that requests queue.
- **Lowest single-request latency:** a server with a batch size of 1 takes 0.58 s for a ~4 s utterance; streaming
  first audio arrives in 38 ms.
- **Quality:** on a 100-prompt VoiceDesign set in six languages, Whisper large-v3 transcripts give 0.5-1.2 % WER/CER
  (sampled, varies run to run). The official `qwen_tts` reference package gives 3.15 % on the same set. Run in fp32 on
  CPU, the Neuron implementation reproduces the reference package's tokens exactly.

## 1. Launch an instance

- **Instance:** `trn2.3xlarge` (1 Trainium2 device, 4 logical NeuronCores at the default LNC=2, or 8 at LNC=1).
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

## 3. Get the Qwen3-TTS workers

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_omni_0_29_0_0_2_0/bin/activate
git clone -b contrib/qwen3-tts https://github.com/jimburtoft/vllm-omni-neuron.git
export PYTHONPATH=$PWD/vllm-omni-neuron/contrib/qwen3_tts NEURON_SKIP_EFA_AFFINITY=1
```

Activate the venv; don't just call its `python`. The Neuron compiler is found through `PATH`.

## 4. Start a server

**One server on the whole device (LNC=2): 2 talker cores + 2 code2wav cores, 48 requests per talker core.**

```bash
QTTS_B=48 QTTS_REP0=2 QTTS_DEV0=0,2 QTTS_REP1=2 QTTS_DEV1=1,3 QTTS_B1=96 \
  python -m qwen3_tts_neuron.serving.make_deploy /tmp/qtts.yaml
QTTS_NC_COUNT=4 python -m qwen3_tts_neuron.serving.serve_launcher serve Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --deploy-config /tmp/qtts.yaml --port 8000 --trust-remote-code --omni \
    --init-timeout 3600 --stage-init-timeout 3600
```

- **First start:** it downloads the weights and compiles every graph, which takes about 30-40 minutes. Wait for
  `Application startup complete`. Later starts load the graphs from the compile cache in a few minutes.
- **Other checkpoints:** use `-CustomVoice` or `-Base` in place of `-VoiceDesign`. The same YAML works for all three.

**Highest throughput: LNC=1, two independent servers, 80 requests per talker core.**
- At LNC=1 the device exposes 8 half-size NeuronCores. A talker step takes about as long on a half-size core as on a
  full one, so splitting the device doubles talker capacity.
- Run two servers, each with 2 talker cores + 2 code2wav cores, and put a load balancer in front of them:

```bash
export NEURON_LOGICAL_NC_CONFIG=1
QTTS_SLOT_LEN=1024 QTTS_B=80 QTTS_REP0=2 QTTS_DEV0=0,2 QTTS_REP1=2 QTTS_DEV1=1,3 QTTS_B1=160 \
  python -m qwen3_tts_neuron.serving.make_deploy /tmp/a.yaml
QTTS_SLOT_LEN=1024 QTTS_B=80 QTTS_REP0=2 QTTS_DEV0=4,6 QTTS_REP1=2 QTTS_DEV1=5,7 QTTS_B1=160 \
  python -m qwen3_tts_neuron.serving.make_deploy /tmp/b.yaml
QTTS_NC_COUNT=8 python -m qwen3_tts_neuron.serving.serve_launcher serve Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --deploy-config /tmp/a.yaml --port 8101 --trust-remote-code --omni --init-timeout 3600 --stage-init-timeout 3600 &
QTTS_NC_COUNT=8 python -m qwen3_tts_neuron.serving.serve_launcher serve Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --deploy-config /tmp/b.yaml --port 8102 --trust-remote-code --omni --init-timeout 3600 --stage-init-timeout 3600 &
```

LNC=1 graphs are compiled separately from LNC=2 ones, so the first start compiles again.
`QTTS_SLOT_LEN=1024` limits a request to 1024 positions: the prompt plus about 80 s of audio. The smaller KV cache
is what leaves room for 80 requests per half-size core.

**Lowest latency:** `QTTS_B=1 python -m qwen3_tts_neuron.serving.make_deploy /tmp/b1.yaml` (one talker core, one
code2wav core).

`make_deploy` options:

| Option | Meaning |
|---|---|
| `QTTS_B` | Requests per talker core (default 8; 48 at LNC=2, 80 at LNC=1 with `QTTS_SLOT_LEN=1024`). |
| `QTTS_SLOT_LEN` | Positions per request (prompt + 12.5 per second of audio; default 2048). |
| `QTTS_REP0` / `QTTS_DEV0` | Number of talker cores and which ones. |
| `QTTS_REP1` / `QTTS_DEV1` | Number of code2wav cores and which ones. Each replica needs its own core. |
| `QTTS_B1` | code2wav concurrency. |
| `QTTS_GREEDY=1` | Greedy decoding by default. The default is the checkpoint's sampling: temperature 0.9, top-k 50. |

## 5. Generate speech

```bash
curl -s localhost:8000/v1/audio/speech -H 'Content-Type: application/json' -o out.wav -d '{
  "model": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "task_type": "VoiceDesign", "response_format": "wav",
  "input": "She said she would be here by noon, and she always keeps her word.", "language": "English",
  "instructions": "Calm, warm adult female voice, steady pace."}'
```

**Request variants:**
- **CustomVoice:** `"task_type": "CustomVoice", "voice": "vivian"`.
- **Base (voice clone):** `"task_type": "Base", "ref_audio": "<url, base64 data URI or path>", "ref_text": "<what the clip says>"`.
  Use `"x_vector_only_mode": true` to clone the voice from the clip without its transcript.
- **Streaming:** add `"stream": true, "stream_format": "audio", "response_format": "pcm"`. The response is 16-bit
  PCM at 24 kHz.

**Sampling is per request.** For example `"seed": 7, "extra_params": {"temperature": 0.9, "top_k": 50}`.
`"temperature": 0` gives greedy. A request with a seed is reproducible whatever else is in the batch.

**Load testing:** use vLLM-Omni's benchmark from a vllm-omni v0.29.0rc1 source checkout, for its bundled datasets.
Set `VLLM_PLUGINS=none` so the client process does not try to open the Neuron device.
```bash
git clone -b v0.29.0rc1 --depth 1 https://github.com/vllm-project/vllm-omni.git
VLLM_PLUGINS=none vllm bench serve --omni --host 127.0.0.1 --port 8000 \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --backend openai-audio-speech --endpoint /v1/audio/speech \
  --dataset-name seed-tts-design --dataset-path vllm-omni/benchmarks/build_dataset/seed_tts_design --seed-tts-locale en \
  --num-prompts 640 --num-warmups 4 --extra-body '{"task_type":"VoiceDesign","language":"English"}' \
  --max-concurrency 128 --request-rate inf --percentile-metrics e2el,audio_rtf,audio_ttfp,audio_duration
```

## Notes

- **Request length:** with `QTTS_SLOT_LEN=1024` a request is capped at its prompt plus about 80 s of audio (the
  LNC=1 recipe); it ends cleanly at the cap. The LNC=2 default (2048) allows about 2.5 minutes.
- **Long text:** split it into sentences or paragraphs. Prompts above about 1,000 characters of English are rejected
  (the server answers "did not produce audio"). The checkpoint itself degrades on very long single utterances.
- **Base voice cloning:** the speaker and speech-tokenizer encoders run on the host CPU, once per new reference clip
  (0.1-0.5 s). Results are cached by clip.
- **Sampling:** `top_p` is ignored and top-k is capped at 64.
- **Greedy decoding:** with CustomVoice and `instructions`, greedy can loop until `max_new_tokens`; the reference
  package does the same. Use the default sampling.
- **Backported fix:** vllm-omni 0.29.0rc1 re-sends the final audio chunk when an utterance ends exactly on a 2 s
  boundary. The workers apply the upstream fix at start-up.
- **Load-test clients:** many HTTP clients cap open connections at 100 by default (httpx, aiohttp). Raise the limit
  when testing above 100 concurrent requests.
