# PhyloGPN on Trainium2 (TorchNeuron, Neuron SDK 2.33)

Genome-scale scoring with [songlab/PhyloGPN](https://huggingface.co/songlab/PhyloGPN) on one Trainium2 device.
PhyloGPN is an 83M-parameter genomic language model: a dilated convolutional network that outputs
[F81](https://en.wikipedia.org/wiki/Models_of_DNA_evolution#F81_model_(Felsenstein_1981)) rate parameters
(log-rates for A, C, G, T) and 960-dimensional embeddings for every site of a DNA sequence
([preprint](https://arxiv.org/abs/2503.03773)).

The package runs the model on PyTorch Native / TorchNeuron with a fused NKI kernel per network block. It loads the
official Hugging Face checkpoint and reproduces the Hub model's outputs (max difference 1e-5 on CPU in fp32).

## Results

trn2.3xlarge (1 Trainium2 device = 4 logical NeuronCores at LNC=2), bf16, public Neuron SDK 2.33 DLAMI, one process
per core. Whole pipeline: bytes → token ids → overlapping windows → device → stitched per-site output, 30 M sites per
core, sustained. From `bash run.sh` (`results/results_reference.json`):

| | sites/s per device | human genome, both strands (6.2 G sites) |
|---|---|---|
| **NKI kernel** (default) | **1.73 M** | **60 min** |
| stock TorchNeuron graph (`kernel=False`) | 0.98 M | 105 min |

Single-window latency (batch 1, one core, ids in → logits back on the host): 11 ms for a 4,096-position window,
33 ms for 16,384. More shapes and the method are in
[BENCHMARK.md](BENCHMARK.md).

**Accuracy** vs the Hugging Face model in fp32 on CPU, same sequence: log-rate cosine similarity 0.99985, and the
highest-probability allele agrees at 99.90 % of sites (bf16 on the device). The CPU path in this package
(`phylogpn_neuron.load()`) matches the Hugging Face model to 1e-5 in fp32.

## 1. Launch the instance

| Setting | Value |
|---|---|
| Instance type | `trn2.3xlarge` |
| AMI | **`Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0 20261020`** (or *Deep Learning AMI Neuron torch-neuronx 2.14.0*) |
| Disk | 100 GB gp3 is enough (the model is 332 MB) |

Look up the AMI in your region:

```bash
aws ssm get-parameter --region <region> \
  --name /aws/service/neuron/dlami/multi-framework/ubuntu-24.04/latest/image_id --query Parameter.Value --output text
```

`trn2` instances are available through EC2 Capacity Blocks.

## 2. Set up

```bash
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
git clone https://github.com/jimburtoft/NeuronStuff.git
cd NeuronStuff/phylogpn-torchneuron
python -m pip install -e package --no-deps
```

The DLAMI venv already has every dependency (`torch`, `numpy`, `safetensors`, `huggingface_hub`). `--no-deps` keeps pip
away from the DLAMI's torch stack. The weights (332 MB) download from the Hugging Face Hub on first use, pinned to
revision `a763739cfaa653fc8305ec908a0b24d6f03c7f05`. Set `HF_HOME` to choose the cache location.

## 3. Run

```bash
bash run.sh            # CPU parity tests, on-device check, then the benchmark above -> results/results.json
bash run.sh quick      # short benchmark (3 M sites per core)
```

The first run compiles the model for its window and batch size; later runs reuse the compile cache. A process keeps its NeuronCores until it exits, so do not run two device jobs on the same cores at once.

## 4. Use it in your own code

```python
import phylogpn_neuron as pg

scorer = pg.Scorer.build(L=16384, B=4)      # load + compile for one (window, batch) shape, on one NeuronCore
logits = scorer.score("ACGT...")            # float32 [len(seq), 4]: ACGT log-rates for every site
probs = logits - logits.max(1, keepdims=True)   # stationary distribution of the F81 model:
probs = np.exp(probs) / np.exp(probs).sum(1, keepdims=True)   # softmax over ACGT
```

- `score` takes a `str`, `bytes` or uint8 token array of any length. Positions near the ends are padded the way the
  model card does (pad token `-`, 240 positions each side), so you get one output per input site.
- Input letters: A, C, G, T in any case; anything else is treated as N.
- One `Scorer` uses one NeuronCore (`NEURON_RT_VISIBLE_CORES`). For full-device throughput run one process per core:
  `harness/bench.py` shows the pattern (4 processes at LNC=2).
- Embeddings: call the model directly, `e, logits = model(ids)` with `ids` a `[B, L]` tensor, `L >= 481`; `e` is
  `[B, L-480, 960]`.
- CPU only (no Neuron device): `model = pg.load()` returns the fp32 PyTorch model, which matches the Hub model.

Window size `L` (default 16,384) trades compile time for throughput: each window carries 480 positions of overlap.
`L=4096` is about 15 % slower per site, `L=8192` about 4 %.

## How it works

- **Exact graph rewrite.** The Hub model crops the residual after every block. That chain of crops does not compile as
  one Neuron graph. Each block here pads its dilated convolution instead ("same" padding), and one 240-position crop per
  side happens before the output head. The central outputs are identical.
- **Weights folded once.** The Hub code symmetrizes the convolution weights (reverse-complement equivariance) on every
  forward pass. `load()` applies that once.
- **Fused block kernel** (`package/phylogpn_neuron/kernel.py`): one NKI kernel per block keeps all intermediates on
  chip, computes the LayerNorm sums on the Tensor engine in bf16, and interleaves two sequence tiles so the Tensor
  engine always has work queued. A one-time exact weight centring (`model.center_()`) means each LayerNorm needs a single
  reduction pass.

## Constraints

- bf16 on the device. The kernel path does not run in fp32. For exact fp32 results use `kernel=False` with an fp32
  model on CPU.
- One `(L, B)` shape per `Scorer`; each new shape compiles once.
- `L` must be at least 482 (the model's receptive field is 481).

## Provenance and license

Package code: Apache-2.0 (`package/LICENSE`). The model code is a re-implementation of the Hub `modeling_phylogpn.py`
(no upstream code is vendored). The weights are not included; they download from
[songlab/PhyloGPN](https://huggingface.co/songlab/PhyloGPN) under its own license (Creative Commons, see the model card).
Cite the PhyloGPN preprint when you use the model.
