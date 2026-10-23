# NeuronStuff
Trainium and Inferentia examples using the Neuron SDK.  This is more a place for me to store reminders and collections of useful code.

- [gpt-oss](./gpt-oss/): Serve GPT-OSS-20B with vLLM on Trainium2 (trn2.3xlarge) on the public Neuron SDK 2.33 DLAMI, with our vllm-neuron and NKI Library branches: FP8 expert weights, 5.5x lower time to first token and up to 2x faster decode than the stock plugin.
- [boltz-neuron](https://github.com/jimburtoft/boltz-neuron) (separate repo): Boltz-2 protein structure prediction on Inferentia2 (inf2.xlarge) with TorchNeuron, on the public Neuron SDK 2.33 torch-neuronx 2.14.0 DLAMI. Drop-in `boltz predict` CLI plus a prebuilt compile cache.
- [voxtral-dflash-vllm-neuron](./voxtral-dflash-vllm-neuron/): Serve Voxtral-Mini-3B with vLLM on Trainium2 (trn2.3xlarge), with optional DFlash speculative decoding (2.3x lower latency on short clips with the public draft). Public Neuron SDK 2.33 DLAMI plus the open vllm-neuron Voxtral PR.
- [qwen3-lora-sft-torchneuron](./qwen3-lora-sft-torchneuron/): LoRA fine-tuning of Qwen3 (0.6B and 8B) on Trainium2 with TorchNeuron and Hugging Face TRL `SFTTrainer` + PEFT, on the public Neuron SDK 2.33 torch-neuronx 2.14.0 DLAMI. Data parallel or FSDP2 with `torchrun`.
- [disaggregated-trainium-nvidia](./disaggregated-trainium-nvidia/): Disaggregated inference between Trainium2 and NVIDIA GPUs. The KV cache moves device to device over NIXL/EFA, in both directions, with vllm-neuron 0.24 and vLLM 0.24.
