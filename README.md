# NeuronStuff
Trainium and Inferentia examples using the Neuron SDK.  This is more a place for me to store reminders and collections of useful code.

- [qwen3-lora-sft-torchneuron](./qwen3-lora-sft-torchneuron/): LoRA fine-tuning of Qwen3 (0.6B and 8B) on Trainium2 with TorchNeuron and Hugging Face TRL `SFTTrainer` + PEFT, on the public Neuron SDK 2.33 torch-neuronx 2.14.0 DLAMI. Data parallel or FSDP2 with `torchrun`.
- [disaggregated-trainium-nvidia](./disaggregated-trainium-nvidia/): Disaggregated inference between Trainium2 and NVIDIA GPUs. The KV cache moves device to device over NIXL/EFA, in both directions, with vllm-neuron 0.24 and vLLM 0.24.
