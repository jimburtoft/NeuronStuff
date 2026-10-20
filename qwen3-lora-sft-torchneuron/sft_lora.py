"""LoRA supervised fine-tuning of Qwen3 on AWS Trainium2 with TorchNeuron + Hugging Face TRL.

Runs on a single NeuronCore or data-parallel across NeuronCores with torchrun, e.g.:
    python sft_lora.py                                   # 1 core
    torchrun --nproc_per_node=4 sft_lora.py              # trn2.3xlarge, all 4 cores (data parallel)
"""
import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--dataset", default="trl-lib/Capybara")
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=50)
    p.add_argument("--output_dir", default="qwen3-lora-trainium")
    p.add_argument("--fsdp", action="store_true", help="Shard the frozen base model with FSDP2 (needed for ~4B+ models)")
    p.add_argument("--compile", action="store_true", help="torch.compile with the neuron backend (single process)")
    args = p.parse_args()

    if args.compile:
        # Neuron compiles one graph per shape; keep shapes static (see huggingface/trl#7660).
        torch._dynamo.config.automatic_dynamic_shapes = False

    extra = {}
    if args.fsdp:
        extra = dict(
            fsdp=True,
            fsdp_config={"fsdp_version": 2, "reshard_after_forward": True, "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                         "transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"]},
        )

    config = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        bf16=True,
        max_length=args.max_length,
        # pad_to_multiple_of defaults to "auto" -> 128 on Neuron, which bounds the number of compiled shapes.
        torch_compile=args.compile,
        model_init_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"},
        **extra,
    )
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=load_dataset(args.dataset, split="train"),
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(args.output_dir)  # saves the LoRA adapter


if __name__ == "__main__":
    main()
