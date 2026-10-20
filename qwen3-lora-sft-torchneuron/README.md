# LoRA fine-tuning of Qwen3 on AWS Trainium2 with TorchNeuron and Hugging Face TRL

Supervised fine-tuning with a LoRA adapter, using only public software:
- the Neuron SDK 2.33 **TorchNeuron** backend (native PyTorch on Trainium, `torch.device("neuron")`);
- Hugging Face **TRL** `SFTTrainer` with **PEFT** LoRA;
- standard data parallelism with `torchrun`, or FSDP2 for larger models.

No custom kernels, patches or private images.

## 1. Launch an instance
- **Instance:** `trn2.3xlarge` (1 Trainium2 device, 4 NeuronCores, 96 GB device memory). `trn2.48xlarge` works
  the same way with more cores.
- **AMI:** *Deep Learning AMI Neuron torch-neuronx 2.14.0 (Ubuntu 26.04)* or *(Ubuntu 24.04)*, Neuron SDK 2.33.0.
  Both were tested. Find the latest ID with SSM:
  ```bash
  aws ssm get-parameter --region <region> \
    --name /aws/service/neuron/dlami/torch-neuronx-2.14.0/ubuntu-26.04/latest/image_id \
    --query Parameter.Value --output text
  ```
  or launch with `--image-id resolve:ssm:/aws/service/neuron/dlami/torch-neuronx-2.14.0/ubuntu-26.04/latest/image_id`.
- **Disk:** 512 GB root volume (models, datasets and the compile cache).

## 2. Install (about 15 seconds)
```bash
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
pip install -r requirements.txt
```
Recommended once per instance, because the compiler can use a lot of host memory on multi-process runs:
```bash
sudo dd if=/dev/zero of=/swapfile bs=1M count=65536 status=none && sudo chmod 600 /swapfile \
  && sudo mkswap /swapfile && sudo swapon /swapfile
```
<details><summary>No DLAMI available? Manual install on stock Ubuntu Server 26.04</summary>

`bash setup_manual_ubuntu2604.sh` follows the
[Neuron manual install guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/pytorch/manual.html)
and adds two things: `libarchive13t64`, which the Neuron compiler needs and the stock image lacks, and a swap file.
Use Ubuntu 26.04 for this path. The pip wheels need glibc 2.43, so they do not import on 24.04.
</details>

## 3. Train
```bash
# Qwen3-0.6B, LoRA r=16 on all linear layers, trl-lib/Capybara, max_length 2048, all 4 NeuronCores (data parallel)
torchrun --nproc_per_node=4 sft_lora.py --max_steps 60

# Qwen3-8B: the frozen base model is sharded across the 4 cores with FSDP2
torchrun --nproc_per_node=4 sft_lora.py --model Qwen/Qwen3-8B --fsdp --max_length 1024 --max_steps 20

# Single NeuronCore
python sft_lora.py --max_steps 30
```
The LoRA adapter is saved to `qwen3-lora-trainium/` (`adapter_model.safetensors`). Load it with
`peft.PeftModel.from_pretrained(base_model, "qwen3-lora-trainium")`. Run `python sft_lora.py --help` for all options.

## What to expect (measured on trn2.3xlarge, 2026-10-20)
| Run | First run on a fresh instance | Once compiled (re-run) |
|---|---|---|
| Qwen3-0.6B, 4 cores, data parallel, max_length 2048, 60 steps | 25-28 min | **2.9 min, ~1.75 s/step** (4 sequences/step) |
| Qwen3-8B, 4 cores, FSDP2, max_length 1024, 20 steps | 22 min | **~4.8 s/step** |

These numbers are the same on the Ubuntu 26.04 and Ubuntu 24.04 DLAMIs, within 3%. Loss decreases as expected:
Qwen3-0.6B goes 1.95 -> 1.35 over 60 steps, and the curve is identical on both DLAMIs.

**Why the first run is slow.** Neuron compiles one graph per input shape, and the first time each shape is seen is
slow. TRL pads sequences to a multiple of 128 on Neuron by default (`pad_to_multiple_of="auto"`), so there are at
most `max_length / 128` shapes. Compiled graphs are cached on disk (`/var/tmp/neuron-compile-cache`), so re-runs
and later epochs are fast. Expect most of the first run to be compilation.

## Limits of this example
- **Sequence length:** tested up to 2048 tokens (0.6B) and 1024 tokens (8B). Attention here is standard PyTorch
  SDPA, which materializes the full attention matrix. Much longer sequences need more memory than one device has,
  and sequence parallelism is not yet supported through the Hugging Face Trainer on Trainium.
- **Model size:** for models above ~4B parameters, use `--fsdp`. Without it, every core holds a full copy of the
  model and runs out of device memory.
- **`--compile`** (`torch.compile` with the Neuron backend) works on a single core but is not recommended here.
  The first run is longer and the steady-state gain is small at this size. Do not combine it with `--fsdp`.
