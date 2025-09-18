# inference on Neuron for the https://github.com/IBM/molformer model

import torch
import torch_neuronx
from transformers import AutoModel, AutoTokenizer
import concurrent.futures
import argparse
import os
import time
import numpy as np
import shutil
import pandas as pd

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = "True"

def encode_smiles(tokenizer, smiles_list, max_length=202, batch_size=1):
    """Encode SMILES strings for MoLFormer input"""
    tokens = tokenizer(
        smiles_list,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    return (
        torch.repeat_interleave(tokens['input_ids'], batch_size, 0),
        torch.repeat_interleave(tokens['attention_mask'], batch_size, 0),
    )

def display(metrics):
    """Display metrics in evenly spaced columns"""
    pad = max(map(len, metrics)) + 1
    for key, value in metrics.items():
        parts = key.split('_')
        parts = list(map(str.title, parts))
        title = ' '.join(parts) + ":"
        
        if isinstance(value, float):
            value = f'{value:0.3f}'
        
        print(f'{title :<{pad}} {value}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=1, help="batch size")
    parser.add_argument("--n_workers", type=int, default=1, help="total threads")
    parser.add_argument("--n_models", type=int, default=1, help="A model for each NeuronCore")
    parser.add_argument("--max_length", type=int, default=202, help="SMILES sequence length (max padding)")
    parser.add_argument("--model", type=str, default="ibm/MoLFormer-XL-both-10pct", help="MoLFormer model name")
    parser.add_argument("--saved_dir", type=str, default="saved_models", help="Directory to save compiled models")
    parser.add_argument("--auto_cast", type=str, default="none", help="compiler args for auto cast")
    parser.add_argument("--auto_cast_type", type=str, default="", help="compiler args for auto cast type")
    parser.add_argument("--verify", action="store_true", help="verify neuron model outputs against CPU reference")
    args = parser.parse_args()

    print("model name: ", args.model)
    print("batch size: ", args.bs)
    print("max_length: ", args.max_length)
    print("verify: ", args.verify)

    name = args.model
    batch_size = args.bs
    n_workers = args.n_workers
    n_models = args.n_models
    max_length = args.max_length
    saved_dir = args.saved_dir
    auto_cast = args.auto_cast
    auto_cast_type = args.auto_cast_type
    
    if auto_cast == "none" or auto_cast == "default":
        auto_cast_type = ""
    
    filename = "neuron_molformer_s{}_b{}_{}_{}.pt".format(max_length, batch_size, auto_cast, auto_cast_type)
    batches_per_thread = 1

    # Build tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    
    # Setup example SMILES inputs (from molformer.py)
    smiles_examples = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)Oc1ccccc1C(=O)O"]
    
    # For tracing and verification, use single example
    single_smiles = [smiles_examples[0]]  # Use only first SMILES
    
    # Encode examples for tracing
    example_inputs = encode_smiles(tokenizer, single_smiles, max_length=max_length, batch_size=batch_size)

    model_file = os.path.join(saved_dir, filename)
    if os.path.isfile(model_file):
        print("Load model from ", model_file)
    else:
        print("Trace model and save it to", model_file)
        if auto_cast == "default":
            compiler_args = []
        elif auto_cast == "none":
            compiler_args = "--auto-cast none"
        else:
            compiler_args = "--auto-cast {} --auto-cast-type {}".format(auto_cast, auto_cast_type)
        
        print("compiler_args: ", compiler_args)
        compiler_workdir = "workdir_molformer_s{}_b{}_{}_{}".format(max_length, batch_size, auto_cast, auto_cast_type)
        
        os.makedirs(saved_dir, exist_ok=True)
        shutil.rmtree(compiler_workdir, ignore_errors=True)
        shutil.rmtree(model_file, ignore_errors=True)
        
        # Load MoLFormer model
        model = AutoModel.from_pretrained(name, deterministic_eval=True, trust_remote_code=True, torchscript=True)
        
        # Trace model for Neuron
        neuron_model = torch_neuronx.trace(model, example_inputs, compiler_args=compiler_args, compiler_workdir=compiler_workdir)
        
        # Save the compiled model
        neuron_model.save(model_file)

    print("== input shape ==")
    print(example_inputs[0].shape)

    # Verification against CPU reference
    if args.verify:
        print("\n== Verification ==")
        # Load CPU reference model
        cpu_model = AutoModel.from_pretrained(name, deterministic_eval=True, trust_remote_code=True)
        cpu_model.eval()
        
        # Load neuron model
        neuron_model = torch.jit.load(model_file)
        
        # Run both models
        with torch.no_grad():
            cpu_outputs = cpu_model(*example_inputs)
            neuron_outputs = neuron_model(*example_inputs)
        
        print(f"CPU output type: {type(cpu_outputs)}")
        print(f"Neuron output type: {type(neuron_outputs)}")
        
        # Compare pooler outputs
        cpu_pooler = cpu_outputs.pooler_output
        print(f"CPU pooler shape: {cpu_pooler.shape}")
        
        # Debug neuron outputs
        if isinstance(neuron_outputs, tuple):
            print(f"Neuron outputs tuple length: {len(neuron_outputs)}")
            for i, out in enumerate(neuron_outputs):
                print(f"Neuron output[{i}] type: {type(out)}, shape: {out.shape if hasattr(out, 'shape') else 'no shape'}")
            neuron_pooler = neuron_outputs[1]
        else:
            print(f"Neuron output shape: {neuron_outputs.shape if hasattr(neuron_outputs, 'shape') else 'no shape'}")
            neuron_pooler = neuron_outputs
        
        print(f"Final comparison - CPU: {cpu_pooler.shape}, Neuron: {neuron_pooler.shape}")
        
        # Handle tensor shape differences
        if cpu_pooler.shape != neuron_pooler.shape:
            print(f"Shape mismatch detected")
            if neuron_pooler.numel() == cpu_pooler.numel():
                print(f"Reshaping neuron output from {neuron_pooler.shape} to {cpu_pooler.shape}")
                neuron_pooler = neuron_pooler.view(cpu_pooler.shape)
            else:
                print(f"Cannot reshape - different number of elements: {neuron_pooler.numel()} vs {cpu_pooler.numel()}")
                exit(1)
        
        max_diff = torch.max(torch.abs(cpu_pooler - neuron_pooler)).item()
        mean_diff = torch.mean(torch.abs(cpu_pooler - neuron_pooler)).item()
        
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        print(f"Outputs match: {torch.allclose(cpu_pooler, neuron_pooler, atol=1e-3)}")
        
        if args.verify:
            exit(0)

    # Load models for benchmarking
    models = [torch.jit.load(model_file) for _ in range(n_models)]

    # Warmup
    for _ in range(8):
        for model in models:
            model(*example_inputs)

    latencies = []

    def task(model):
        print("batch per thread: ", batches_per_thread)
        for _ in range(batches_per_thread):
            start = time.time()
            model(*example_inputs)
            finish = time.time()
            latencies.append(finish - start)

    # Submit tasks
    begin = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        fut_list = []
        for i in range(n_workers):
            fut = pool.submit(task, models[i % len(models)])
            fut_list.append(fut)
        for fut in fut_list:
            fut.result()
    end = time.time()

    # Compute metrics
    duration = end - begin
    inferences = len(latencies) * batch_size
    throughput = inferences / duration
    avg_latency = np.mean(latencies) * 1000.0
    p50_latency = np.percentile(latencies, 50) * 1000.0
    p90_latency = np.percentile(latencies, 90) * 1000.0
    p99_latency = np.percentile(latencies, 99) * 1000.0

    # Display metrics
    metrics = {
        'filename': str(filename),
        'batch_size': batch_size,
        'batches': len(latencies),
        'inferences': inferences,
        'workers': n_workers,
        'models': n_models,
        'duration': duration,
        'throughput': throughput,
        'Avg. latency (ms)': avg_latency,
        'P50 latency (ms)': p50_latency,
        'P90 latency (ms)': p90_latency,
        'P99 latency (ms)': p99_latency,
    }
    display(metrics)

    # Save results to CSV
    metrics_csv = {
        'Filename': [str(filename)],
        'batch size': [batch_size],
        'n_workers': [n_workers],
        'n_models': [n_models],
        'Throughput': [throughput],
        'Avg. latency (ms)': [avg_latency],
        'P50 latency (ms)': [p50_latency],
        'P90 latency (ms)': [p90_latency],
        'P99 latency (ms)': [p99_latency],
    }
    df = pd.DataFrame(metrics_csv)
    csv_file = "neuron_molformer_s{}_b{}_{}_{}_w{}_m{}.csv".format(max_length, batch_size, auto_cast, auto_cast_type, n_workers, n_models)
    df.to_csv(csv_file)
    print("Save results to {}".format(csv_file))
