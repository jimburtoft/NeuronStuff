# Evo 2 7B on Trainium2 (TorchNeuron, Neuron SDK 2.33)

Batched autoregressive decode for [Evo 2](https://github.com/ArcInstitute/evo2)
([`arcinstitute/evo2_7b_base`](https://huggingface.co/arcinstitute/evo2_7b_base)), the Arc Institute DNA language
model, on one Trainium2 device with PyTorch Native / TorchNeuron (`torch.device("neuron")`).

The `evo2_neuron_decode` package routes the upstream [`vortex`](https://github.com/Zymrael/vortex) model code through
fused NKI kernels for the Hyena blocks, the NKI Library `attention_tkg` kernel for decode attention, and tensor
parallelism. It carries no model definition of its own: it patches vortex at runtime (pinned commit in
`package/VENDORED.md`).

Evo 2 / StripedHyena 2 has 32 blocks: 9 HCM (Hyena, 128-tap inner filter), 9 HCS (7-tap), 9 HCL (long IIR) and 5
attention blocks; 8,192-token trained context; 512-token byte vocabulary (DNA bases A/C/G/T).

## Results

trn2.3xlarge (1 Trainium2 device = 4 logical NeuronCores at LNC=2), bf16, public Neuron SDK 2.33 DLAMI, 128-nt prompt:

| configuration | ms/step | **tok/s per device** |
|---|---|---|
| TP=4, B=88, `max_seqlen=8192`, KV fill 128 | 37.7 | **2,333** |
| TP=4, B=88, KV fill 1,024 | 40.3 | **2,181** |
| TP=4, B=88, KV fill 4,096 | 56.1 | **1,569** |
| TP=4, B=80, KV fill 7,680 | 49.4 | **1,618** |
| DP=4 (4 processes), B=128 per core, `max_seqlen=256` | 152.6 | **3,356** |

Prefill (TP=4, B=8): 11,600-16,000 prompt tokens/s for prompts of 256 to 8,000 nt. Correctness: logits within
`cos_sim 0.99999930` of an independent fp32 CPU implementation; rows inside a batch bit-identical; no CPU fallback
ops. Details, method and caveats: [BENCHMARK.md](BENCHMARK.md).

This package is built for batched throughput. Single-request (B=1) decode is ~30 ms/token on one core.

## 1. Launch the instance

| Setting | Value |
|---|---|
| Instance type | `trn2.3xlarge` |
| AMI | **`Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0 20261020`** (or *Deep Learning AMI Neuron torch-neuronx 2.14.0*) |
| Disk | 300 GB gp3 |

```bash
aws ssm get-parameter --region <region> \
  --name /aws/service/neuron/dlami/multi-framework/ubuntu-24.04/latest/image_id --query Parameter.Value --output text
```

`trn2` instances are available through EC2 Capacity Blocks.

## 2. Set up

The DLAMI has no swap, and `neuronx-cc` needs up to ~40 GB of host memory on the larger graphs:

```bash
sudo dd if=/dev/zero of=/swapfile bs=1M count=65536 status=none && sudo chmod 600 /swapfile
sudo mkswap --force /swapfile && sudo swapon /swapfile

source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
git clone https://github.com/jimburtoft/NeuronStuff.git
cd NeuronStuff/evo2
bash run.sh setup        # package (pip install -e package --no-deps), vortex at the pinned commit, 13 GB weights
```

`--no-deps` keeps pip away from the DLAMI's torch stack. The only extra packages are vortex's own imports (`rich`,
`einops`, `pyyaml`), which `run.sh` installs if they are missing.

## 3. Run

```bash
bash run.sh fill         # TP=4 8K sweep (rows 1-4 of the table above) -> results/fill.jsonl
bash run.sh dp4-256      # DP=4 @ 256 (row 5)
bash run.sh test-sim     # CPU + NKI simulator tests
bash run.sh test-hw      # on-device tests: fp32 logit parity, batch invariance, TP sharding, prefill
```

The first run compiles every graph; later runs reuse the compile cache (`TORCH_NEURONX_NEFF_CACHE_DIR`, default
`~/.cache/evo2_neff`). On SDK 2.33 a first compile at TP=4 with B >= 48 occasionally fails with `NCC_INLA001` or
`NCC_IBIR229`; the same graph compiles on the next attempt, and `run.sh` retries those two errors automatically
(an arm can need several attempts on a fresh cache).
A warm cache removes compilation but not tracing: a fresh process still needs a few minutes before it serves at full
speed, so load once and serve many requests. A process keeps its NeuronCores until it exits.

## 4. Use it in your own code

```python
import torch, yaml
import evo2_neuron_decode as evo2d
from evo2_neuron_decode.patches import neuron_shim
neuron_shim.install()                                  # makes vortex importable without CUDA extensions
from vortex.model.utils import dotdict
from vortex.model.model import StripedHyena

MAX_SEQLEN = 8192                                      # multiple of 256
evo2d.enable()                                         # patch vortex (all measured defaults on)
cfg = evo2d.prepare_config(dotdict(yaml.safe_load(open("vortex/configs/evo2-7b-8k.yml"))), MAX_SEQLEN)
cfg.max_batch_size = 88

model = StripedHyena(cfg)
model.load_state_dict(torch.load("evo2_7b_base.pt", weights_only=False), strict=False)
# TP=4: under torchrun, dist.init_process_group(backend="neuron") and
#       evo2_neuron_decode.tp.shard_model(model, rank, 4) here, before to_device
model = evo2d.to_device(model, dtype=torch.bfloat16).eval()   # NOT bare .to()
evo2d.compile_blocks(model)                                   # all 32 blocks; not optional

ids = torch.tensor([[ord(c) for c in "ACGT..."]] * 88, device="neuron")
ipd, last_logits = evo2d.prefill(model, ids, chunk_rows=8,
                                 buckets=(128, 256, 512, 1024, 2048, 4096, 8192))
# decode: model(next_tokens, inference_params_dict=ipd), then advance ipd[n].seqlen_offset by 1
print(evo2d.config_summary())
```

`harness/bench_fill.py` is a complete TP=4 and single-core decode loop (`--fill-mode prompt` prefills a real
prompt of the fill length instead of setting the KV offset). `package/README.md` documents every
option and the measurement behind it.

## Constraints

- `max_seqlen` must be a multiple of 256 (`attention_tkg` shards the KV length by 2 into 128-row tiles).
- TP=4 at `max_seqlen=8192` fits B=88 (B=80 near full fill); B=96 runs out of device memory.
- One batch size per compiled model; each new prompt length is a new prefill graph unless you use `buckets=`.
- FP8 is not supported (31 of 32 MLPs use an identity activation that the FP8 MLP kernels cannot express).

## License

Apache-2.0 (`package/LICENSE`, `package/NOTICE`). The model definition ([`Zymrael/vortex`](https://github.com/Zymrael/vortex),
[`ArcInstitute/evo2`](https://github.com/ArcInstitute/evo2), Apache-2.0) and the weights are not included and are
downloaded separately.
