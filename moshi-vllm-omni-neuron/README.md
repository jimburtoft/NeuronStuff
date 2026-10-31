# Moshi full-duplex speech on Trainium2 with vLLM-Omni (Neuron SDK 2.33)

Serve kyutai's [Moshi](https://github.com/kyutai-labs/moshi) (`kyutai/moshiko-pytorch-bf16` or
`kyutai/moshika-pytorch-bf16`) on a trn2.3xlarge as a **full-duplex, real-time voice model** behind
vLLM-Omni's OpenAI-Realtime-style WebSocket API (`/v1/realtime?duplex=1`). Moshi listens and talks at the same
time in 80 ms frames: there is no turn detection, no ASR -> LLM -> TTS cascade.

- **Environment:** the public Neuron SDK 2.33 DLAMI, its vLLM-Omni venv (vllm 0.29.0, vllm-omni 0.29.0rc1,
  vllm-omni-neuron 0.29.0.0.2.0) unchanged, plus the `contrib/moshi` directory of
  [jimburtoft/vllm-omni-neuron](https://github.com/jimburtoft/vllm-omni-neuron/tree/contrib/moshi/contrib/moshi)
  on `PYTHONPATH`.
- **Model:** Moshi 7.7B (Helium-7B temporal transformer + 8-step depth transformer, BF16) and the Mimi codec
  (FP32). Weights CC-BY 4.0.
- **Endpoint:** `ws://<host>:8000/v1/realtime?duplex=1` (24 kHz float32 PCM in, 24 kHz 16-bit PCM out).

## How it works

vLLM-Omni already runs Moshi-architecture models (its PersonaPlex pipeline, a Moshi fine-tune) as a two-stage
duplex pipeline on GPUs. vllm-omni-neuron 0.29 has no Neuron worker for those stages, so `contrib/moshi` adds two:

| Stage | NeuronCore | Work per 80 ms frame, for every live session |
|---|---|---|
| 0 | 1 core | Mimi-encode the user's 1920 samples, then one compiled graph: Helium step + 8 depth-transformer steps + sampling |
| 1 | 1 core | Mimi-decode Moshi's 8 audio codes to 1920 samples |

Every session's streaming state (Mimi convolution buffers, transformer KV caches, the 3000-frame Helium KV)
stays in device memory, so only audio and a few tokens cross the host per frame.

## Results (trn2.3xlarge, one server = 2 NeuronCores)

| Concurrent sessions | Every frame in real time | First audio | Per-frame response latency, p50 / p99 |
|---|---|---|---|
| 1 | yes | 175 ms | 93 / 96 ms |
| 4 | yes | 220-230 ms | 112-129 / 151-167 ms |
| 5 | yes | 242 ms | 135-147 / 176-186 ms |

A trn2.3xlarge has 4 NeuronCores, so two servers run **10 real-time conversations per instance** (measured: all 10
sessions concurrent, every frame on time, p99 latency at most 180 ms).

Per-frame response latency is the time from sending an 80 ms input frame to receiving the 80 ms of Moshi audio it
produces (measured on a 28.8 s conversation streamed at real-time pace; it does not grow over the session).

The temporal transformer's attention runs on the fused decode kernel from the Neuron Kernel Library
(`attention_block_tkg`, shipped in the DLAMI); everything else is plain PyTorch compiled by the Neuron compiler.

Quality: with greedy decoding the Trainium output reproduces the kyutai reference implementation's text on our
test clips, and Whisper's transcript of the generated speech matches Moshi's inner-monologue text.

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

## 3. Get the Moshi workers

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_omni_0_29_0_0_2_0/bin/activate
git clone -b contrib/moshi https://github.com/jimburtoft/vllm-omni-neuron.git
export PYTHONPATH=$PWD/vllm-omni-neuron/contrib/moshi NEURON_SKIP_EFA_AFFINITY=1
```

Activate the venv (don't just call its `python`): the Neuron compiler is found through `PATH`.

## 4. Start the server

```bash
MOSHI_B=5 python -m moshi_neuron.serving.make_deploy /tmp/moshi.yaml   # 5 sessions; stage 0 on core 0, stage 1 on core 1
MOSHI_NC_COUNT=4 python -m moshi_neuron.serving.serve_launcher serve kyutai/moshiko-pytorch-bf16 \
    --deploy-config /tmp/moshi.yaml --port 8000 --omni --init-timeout 3600 --stage-init-timeout 3600
```

The first start downloads the weights (15 GB) and compiles the graphs, which takes about 8 minutes; wait for
`Application startup complete`. Later starts load the graphs from the compile cache.

For a second server on the other two cores (10 sessions per instance), run the same commands in another shell with
`MOSHI_DEV0=2 MOSHI_DEV1=3` for `make_deploy`, a different YAML path and `--port 8001`.

`make_deploy` options: `MOSHI_B` (sessions per server, default 1, at most 5: six sessions' KV caches do not fit next to the weights on one core), `MOSHI_DEV0` / `MOSHI_DEV1` (cores),
`MOSHI_GREEDY=1` in the server environment for greedy decoding (default: Moshi's sampling, text temperature 0.7
top-k 25, audio 0.8 top-k 250).

## 5. Talk to it

Stream a 24 kHz (or any rate; it is resampled) wav file at real-time pace and record the reply:

```bash
python -m moshi_neuron.serving.duplex_client --wav my_question.wav --sessions 1 --out out/
```

`out/session0.wav` is Moshi's side of the conversation, time-aligned with your input. `--sessions 5` opens five
concurrent conversations. Any client of the vLLM-Omni realtime duplex protocol works, for example
`vllm_omni.clients.duplex.DuplexClient` with
`SessionConfig(input_audio=AudioFormat("pcm_f32le", 24000), output_audio=AudioFormat("pcm16", 24000))`.
Moshi takes no `voice` or `instructions`.

## Notes

- The installed vllm-omni 0.29 builds a PersonaPlex voice/persona prompt for every session, which Moshi
  checkpoints don't have. The launcher applies a small runtime patch (the same rule as upstream vllm-omni
  pull request [#9319](https://github.com/vllm-project/vllm-omni/pull/9319)): Moshi sessions start with no prompt.
- Sessions longer than 240 s (3000 frames) are not supported yet.
- The text side channel (Moshi's inner monologue) is not yet forwarded to the client.
