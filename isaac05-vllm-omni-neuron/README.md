# Isaac-0.5 robot policy on Trainium2 with vLLM-Omni + vLLM-Neuron (Neuron SDK 2.33)

Serve PerceptronAI's [Isaac-0.5](https://huggingface.co/PerceptronAI/Isaac-0.5) on a trn2.3xlarge in two ways:

- **Robot actions:** a vision-language-action policy behind vLLM-Omni's **OpenPI** websocket
  (`ws://<host>:8000/v1/realtime/robot/openpi`). Send camera frames, joint state and an instruction; get back a
  50 x 7 action chunk. Concurrent clients are batched on the device.
- **Text / vision-language:** the same 36B model behind the OpenAI-compatible vLLM API.

The model is a 36B Qwen3.5-style hybrid MoE: Gated DeltaNet + gated attention, 256 experts top-8 with a null-expert
router, and a Qwen3.5 vision tower. The action path adds a 0.58B DiT flow expert that runs 10 Euler steps.

- **Environment:**
  - the public Neuron SDK 2.33 DLAMI and its vLLM (`vllm 0.29.0`, `vllm-neuron 0.29.0.1.2.0`) and vLLM-Omni
    (`vllm-omni 0.29.0rc1`, `vllm-omni-neuron 0.29.0.0.2.0`) venvs, unchanged;
  - the [`isaac05-action`](https://github.com/jimburtoft/vllm-neuron/tree/isaac05-action) branch of
    jimburtoft/vllm-neuron (based on tag `v0.29.0.1.2.0`) on `PYTHONPATH`;
  - Perceptron's LeRobot fork and mharmony from GitHub, installed into a side directory.
- **Model:** BF16, converted once from the published FP32 checkpoint.
- **Details:** setup, serving switches, the kernels and the full measurements are in the
  [branch README](https://github.com/jimburtoft/vllm-neuron/blob/isaac05-action/contrib/isaac05_omni/README.md).

## Results (one trn2.3xlarge: 1 Trainium2 device, TP=4)

**Robot actions:**
- Setup: OpenPI websocket, 2 cameras at 256 x 256 (3-step observation window), joint state and instruction
  (462 prompt tokens), 50 x 7 chunk.
- Measurement: 60 s windows, latency = full websocket round trip.

| Concurrent clients | 1 | 2 | 4 | 8 | 16 | 32 | 48 |
|---|---|---|---|---|---|---|---|
| Plans per second | 4.50 | 4.88 | 6.40 | 8.00 | 9.62 | 11.47 | 11.48 |
| p50 latency (ms) | 222 | 408 | 624 | 993 | 1641 | 2766 | 4143 |

Accuracy: action chunks match the reference implementation run with the same initial noise. Cosine similarity is
0.998 / 0.999 / 0.999 on the three test observations, one at a time and in batches of 16.

**Text and images** (`vllm bench serve`, output tokens/s at the throughput plateau):

| Input / output tokens | 256 / 128 | 256 / 1024 | 1024 / 128 | 2048 / 1024 | 7168 / 1024 | 2 images (384²) + 128 / 128 |
|---|---|---|---|---|---|---|
| Peak output tok/s | 1531 | 2296 | 650 | 762 | 396 | 700 |

Single-request decode: 6.1 ms per token at 2K context, 7.0 ms at 8K.

## How it works

| Part | Where | What |
|---|---|---|
| Observation render | host (rank 0) | LeRobot + mHarmony build the prompt stream. Raw frames go straight to the image processor (no PNG round trip). |
| Vision tower | NeuronCores | One compiled graph per image grid. Images in a batch are split across the 4 cores. |
| VLM prefill | NeuronCores (TP4) | One packed prefill of B x 512 tokens per batch, with per-observation attention and state resets. The MoE uses a **job-list NKI kernel** that loads each used expert once per layer, sized for the sparse routing at short prompts. |
| DiT flow expert | NeuronCores | All 10 Euler steps in one graph. Per-step conditioning is precomputed and kept in device memory. |

It plugs in as a vLLM-Omni pipeline (`IsaacActionPipeline`, request batching on) through the package's entry point.
No SDK package is modified.

## 1. Launch an instance

- **Instance:** `trn2.3xlarge` (1 Trainium2 device, 4 logical NeuronCores at the default LNC=2).
- **AMI:** *Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0* (multi-framework). Latest ID:
  ```bash
  aws ssm get-parameter --region <region> \
    --name /aws/service/neuron/dlami/multi-framework/ubuntu-24.04/latest/image_id \
    --query Parameter.Value --output text
  ```
- **Disk:** 500 GB. It holds the FP32 download (143 GB), the BF16 copy (67 GB) and the compile caches.

## 2. Add swap (once per instance)

```bash
sudo fallocate -l 64G /swapfile && sudo chmod 600 /swapfile
sudo mkswap --force /swapfile && sudo swapon /swapfile
```

## 3. Get the code, the checkpoint and the extras

```bash
git clone -b isaac05-action https://github.com/jimburtoft/vllm-neuron.git ~/vllm-neuron
C=~/vllm-neuron/contrib/isaac05_omni
source /opt/aws_neuronx_venv_pytorch_inference_vllm_omni_0_29_0_0_2_0/bin/activate

# BF16 checkpoint (the published one is FP32)
hf download PerceptronAI/Isaac-0.5 --revision 6e397429f0ab91528ebac915004eba5e763fcdd4 --local-dir ~/models/Isaac-0.5-fp32
python $C/scripts/convert_bf16.py ~/models/Isaac-0.5-fp32 ~/models/Isaac-0.5-bf16

# LeRobot (Perceptron fork) + mharmony (Rust extension, built from source) into a side directory
git clone https://github.com/perceptron-ai-inc/lerobot.git ~/lerobot && git -C ~/lerobot checkout e12389c1f8f591ad05dced4e284d4e92e48c5df4
git clone https://github.com/perceptron-ai-inc/mharmony.git ~/mharmony && git -C ~/mharmony checkout 9a57efbf8ada2d73641fc40081f4e4fa6ff2ab92
curl -sSf https://sh.rustup.rs | sh -s -- -y && source ~/.cargo/env
python3 -m venv ~/maturin && ~/maturin/bin/pip install maturin
(cd ~/mharmony && ~/maturin/bin/maturin build --release -o /tmp/mhwheel)
mkdir -p ~/isaac05_overlay
pip install --no-deps --target ~/isaac05_overlay /tmp/mhwheel/*.whl draccus gymnasium Farama-Notifications \
    mergedeep termcolor toml typing-inspect
ln -s ~/lerobot/src/lerobot ~/isaac05_overlay/lerobot
pip install --no-deps --target ~/isaac05_overlay $C
python -c "from huggingface_hub import hf_hub_download as d; [d('Qwen/Qwen3.5-0.8B', f) for f in ['vocab.json','merges.txt','tokenizer.json','tokenizer_config.json','config.json','chat_template.jinja']]"
```

Activate the venv (don't just call its `python`): the Neuron compiler is found through `PATH`.

## 4a. Serve robot actions (OpenPI)

```bash
ISAAC05_MODEL=~/models/Isaac-0.5-bf16 DEPLOY=isaac05_action_tp4_batch.yaml \
  XENV="ISAAC05_MAXB=16 ISAAC05_BATCH_BUCKETS=1,2,4,8,16" bash $C/serve_isaac.sh
```

Wait for `Application startup complete`. The **first request at each batch size compiles that size's graphs**, which
takes a long time. Warm up every bucket (1, 2, 4, 8, 16 concurrent requests) before you measure. Later starts load the
graphs from the compile cache. For one request at a time, use `DEPLOY=isaac05_action_tp4.yaml` and drop `XENV`.

## 4b. Serve text or images (OpenAI API)

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
python $C/scripts/make_vllm_dir.py ~/models/Isaac-0.5-bf16 ~/models/Isaac-0.5-vllm text
python $C/scripts/make_vllm_dir.py ~/models/Isaac-0.5-bf16 ~/models/Isaac-0.5-vllm-vl vl
MODEL=~/models/Isaac-0.5-vllm    PROFILE=t2k bash $C/scripts/serve_text.sh   # text, up to 2K context
MODEL=~/models/Isaac-0.5-vllm    PROFILE=t8k bash $C/scripts/serve_text.sh   # text, up to 8K context
MODEL=~/models/Isaac-0.5-vllm-vl PROFILE=img bash $C/scripts/serve_text.sh   # up to 2 images per prompt
```

## 5. Send an observation

The endpoint speaks the standard OpenPI protocol: msgpack with numpy arrays, LeRobot keys. Any OpenPI client works.
Minimal Python:

```python
import asyncio, msgspec, numpy as np, websockets

def enc(o):  # openpi msgpack-numpy
    if isinstance(o, np.ndarray):
        return {b"__ndarray__": True, b"data": o.tobytes(), b"dtype": o.dtype.str, b"shape": list(o.shape)}
    raise TypeError(type(o))

async def main():
    async with websockets.connect("ws://localhost:8000/v1/realtime/robot/openpi", max_size=None) as ws:
        print("server:", msgspec.msgpack.decode(await ws.recv()))  # policy metadata
        obs = {
            "observation.images.image": np.zeros((3, 256, 256), np.uint8),        # external camera, uint8
            "observation.images.wrist_image": np.zeros((3, 256, 256), np.uint8),  # wrist camera
            "observation.state": np.zeros(8, np.float32),
            "task": "put the red block in the bowl",
        }
        await ws.send(msgspec.msgpack.encode(obs, enc_hook=enc))
        reply = msgspec.msgpack.decode(await ws.recv())  # {"actions": msgpack-numpy array [50, 7]}
        a = reply.get(b"actions", reply.get("actions"))
        g = lambda k: a.get(k.encode(), a.get(k))
        actions = np.frombuffer(g("data"), np.dtype(g("dtype"))).reshape(g("shape"))
        print(actions.shape)

asyncio.run(main())
```

Each request is rendered on its own: the server keeps no observation history between requests, and the single frame per camera fills the policy's 3-step window (`n_obs_steps`). Send frames as **uint8**. Float frames work, but they add about 30 ms per request in transfer. The camera keys must
match the policy's `camera_order`: see `lerobot_policy/config.json` in the checkpoint, or the server's metadata
message. `$C/openpi_gate.py` and `$C/openpi_load.py` are a full accuracy client and a load generator.

## Notes
- One observation layout per compiled graph set: 2 cameras at 256 x 256 with a 3-step window, and a prompt of 512
  tokens or fewer. A different layout compiles new graphs on its first request.
- Up to 16 observations per device step. A step of 32 fails to compile with this SDK's neuronx-cc (NCC_INLA001).
- Each request is stateless: the 3-step observation window is filled from the current frame. Multi-frame history
  across requests and real-time chunking (action-prefix conditioning) are not wired in yet.
