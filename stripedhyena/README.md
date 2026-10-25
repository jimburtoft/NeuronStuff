# StripedHyena-Hessian-7B on Trainium2 (TorchNeuron, Neuron SDK 2.33)

Autoregressive decode for [togethercomputer/StripedHyena-Hessian-7B](https://huggingface.co/togethercomputer/StripedHyena-Hessian-7B)
(Together AI, the StripedHyena 1 architecture) on one Trainium2 device. The package runs the upstream
`stripedhyena` model code (vendored, unmodified) on PyTorch Native / TorchNeuron with runtime patches,
a compiled segmented prefill, a compiled decode step, a fused NKI Hyena decode kernel, the NKI Library
`attention_tkg` and `mlp` kernels, and Megatron-style tensor parallelism. **0 CPU fallback ops** in prefill and
decode.

The architecture: 32 blocks alternating gated-convolution Hyena blocks (short FIR k=3 + a long convolution realised
as a 2-state complex modal SSM) and grouped-query attention blocks (32 Q / 8 KV heads, d=128, RoPE), SiLU MLP,
vocab 32,000, **32K context**. The Hyena half keeps a constant ~0.4 MB decode state per sequence; only the attention
half has a KV cache. The same package runs the chat fine-tune
[StripedHyena-Nous-7B](https://huggingface.co/togethercomputer/StripedHyena-Nous-7B) (identical architecture).

## Results

trn2.3xlarge (1 Trainium2 device = 4 logical NeuronCores at LNC=2), TP=4, bf16 weights with fp32 poles/residues,
public Neuron SDK 2.33 DLAMI. From `bash run.sh` (`results/results_reference.json`):

| | 2K context | 8K context | 32K context |
|---|---|---|---|
| decode, best batch | **5,774 tok/s @ B=512** | **1,992 tok/s @ B=128** | **645 tok/s @ B=30** |
| decode, B=1 (ms/token) | **12.8** | 13.2 | 13.3 |
| time to first token, B=1 | 142 ms | 743 ms | 8.5 s (32,512-token prompt) |

More shapes, serving-style workloads and accuracy: [BENCHMARK.md](BENCHMARK.md).

**Accuracy** vs the upstream reference implementation on the same tokens: WikiText-2 perplexity 6.1966 / 5.5418 at
2,048 / 8,192 (reference 6.1945 / 5.5423); ARC-Easy, PIQA and HellaSwag (n=500) equal to the reference within one
standard error; teacher-forced decode logits top-1 agreement >= 0.98, mean KL <= 0.0007.

## 1. Launch the instance

| Setting | Value |
|---|---|
| Instance type | `trn2.3xlarge` |
| AMI | **`Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0 20261020`** (or *Deep Learning AMI Neuron torch-neuronx 2.14.0*) |
| Disk | 300 GB gp3 |

Look up the AMI in your region:

```bash
aws ssm get-parameter --region <region> \
  --name /aws/service/neuron/dlami/multi-framework/ubuntu-24.04/latest/image_id --query Parameter.Value --output text
```

`trn2` instances are available through EC2 Capacity Blocks.

## 2. Set up

The DLAMI has no swap, and `neuronx-cc` needs tens of GB of host memory for the 32K graphs:

```bash
sudo dd if=/dev/zero of=/swapfile bs=1M count=65536 status=none && sudo chmod 600 /swapfile
sudo mkswap --force /swapfile && sudo swapon /swapfile

source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
git clone https://github.com/jimburtoft/NeuronStuff.git
cd NeuronStuff/stripedhyena
```

The venv already has every dependency (`einops`, `safetensors`, `pyyaml`, `tokenizers`, `huggingface_hub`, `pytest`).
`run.sh` installs the package with `pip install -e package --no-deps` (`--no-deps` keeps pip away from the DLAMI's
torch stack) and downloads the weights (15.3 GB, Hugging Face revision `59e474c241746a77ff2439ed39dd5469381d3f92`)
into `$SHN_WEIGHTS` (default `~/weights/hessian-7b`).

## 3. Run

```bash
bash run.sh            # CPU + simulator tests, then the TP=4 benchmark above -> results/results.json
bash run.sh test-hw    # on-device parity gates vs the reference + batched generation
bash run.sh quick      # one decode point + one TTFT point
```

Each benchmark point is its own process (one compiled shape set per process). The first run compiles every graph;
later runs reuse the compile cache. At the batch ceilings (B=128 @ 8K, B=30 @ 32K) the runtime prints
`Failed to allocate DEVICE memory` blocks during warm-up; they are non-fatal. A process keeps its NeuronCores until it
exits, so do not run two device jobs at once.

## 4. Use it in your own code

```python
import torch, striped_hyena_neuron as shn
from tokenizers import Tokenizer

model, info = shn.load("~/weights/hessian-7b")              # meta build + strict load + device placement
gen = shn.compile(model, B=1, window=4096)                  # compiled prefill + decode (not optional)
tok = Tokenizer.from_file(shn.TOKENIZER_JSON)
ids = torch.tensor([tok.encode("The striped hyena is").ids])
print(tok.decode(gen.generate(ids, max_new_tokens=64)[0].tolist()))
```

For TP=4, run your script with `torchrun --standalone --nproc_per_node=4` and use
`dist.init_process_group(backend="neuron")`, `shn.load(weights, tp=(rank, 4))` and
`shn.compile(model, B=..., window=..., tp=4)`. `package/README.md` documents every default and the measurement
behind it; `package/examples/generate.py` is a complete example.

## Constraints

- KV window: a multiple of 256, at most 32,768. The TP degree must divide 8 (the KV head count).
- One (batch, window) configuration per process.
- One logical core at a 32K window serves B=1 only; use TP=4 for long contexts and large batches.

## Provenance and license

Apache-2.0 (`package/LICENSE`, `package/NOTICE`). Vendored upstream:
[togethercomputer/stripedhyena](https://github.com/togethercomputer/stripedhyena) at
`7e13f618027fea9625be1f2d2d94f9a361f6bd02` (Apache-2.0). The weights are not included (Apache-2.0, downloaded from
Hugging Face). Adapted code is listed in `package/PROVENANCE.md`: the fused cascade kernel, the `attention_tkg`
integration and the TP sharding come from the [Evo 2 package](../evo2/) in this repository.
