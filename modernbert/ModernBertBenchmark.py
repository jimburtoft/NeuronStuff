import torch
import torch_neuronx
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import transformers
import concurrent.futures
import argparse
import os
import time
import numpy as np
import shutil
import pandas as pd
#import torch._dynamo

#torch._dynamo.config.suppress_errors = True

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = "True"

def encode(tokenizer, *inputs, max_length=2048, batch_size=1):
    tokens = tokenizer.encode_plus(
        *inputs,
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
    """
    Display a dictionary of metrics in evenly spaced columns.

    Args:
        metrics: A dictionary of performance statisics.
    """
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
    parser.add_argument("--n_models", type=int, default=1, help="A model for each NeuronCore on inf2 instance")
    parser.add_argument("--max_length", type=int, default=2048, help="NLP sequence length (max padding)")
    parser.add_argument("--model", type=str, default="clapAI/modernBERT-large-multilingual-sentiment", help="HuggingFace model name or local model directory")
    parser.add_argument("--saved_dir", type=str, default="saved_models", help="HuggingFace model name or local model directory")
    parser.add_argument("--auto_cast", type=str, default="none", help="compiler args for auto cast")
    parser.add_argument("--auto_cast_type", type=str, default="", help="compiler args for auto cast type")
    parser.add_argument("--instance", type=str, default="inf2", help="instance to benchmark, ex: inf2, g5")
    args = parser.parse_args()

    print("model name: ", args.model)
    print("batch size: ", args.bs)
    print("n_workers: ", args.n_workers)
    print("n_models: ", args.n_models)
    print("max_length: ", args.max_length)
    print("saved dir: ", args.saved_dir)
    print("instance: ", args.instance)

    name = args.model
    batch_size = args.bs
    n_workers = args.n_workers
    n_models = args.n_models
    max_length = args.max_length
    saved_dir = args.saved_dir
    auto_cast = args.auto_cast
    auto_cast_type = args.auto_cast_type
    instance = args.instance
    if auto_cast == "none" or auto_cast == "default":
        auto_cast_type = ""
    filename = "neuron_{}_s{}_b{}_{}_{}.pt".format(name, max_length, batch_size, auto_cast, auto_cast_type)
    # batches_per_thread = 10000  # For benchmarking purposes, use a fixed number of inference requests
    batches_per_thread = 1  # For benchmarking purposes, use a fixed number of inference requests

    # Build tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(name)
    
    # Setup some example inputs
    sequence_0 = "The company HuggingFace is based in New York City"
    sequence_1 = "Apples are especially bad for your health"
    sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

    paraphrase = encode(tokenizer, sequence_0, sequence_2, max_length=max_length, batch_size=batch_size)
    not_paraphrase = encode(tokenizer, sequence_0, sequence_1, max_length=max_length, batch_size=batch_size)

    model_file = os.path.join(saved_dir, filename)
    if os.path.isfile(model_file):
        print("Load models from ", model_file)
    else:
        print("Trace model and save it to", model_file)
        if auto_cast == "default":
            compiler_args = []
        elif auto_cast == "none":
            compiler_args = "--auto-cast none"
        else:
            compiler_args = "--auto-cast {} --auto-cast-type {}".format(auto_cast, auto_cast_type)
        print("compiler_args: ", compiler_args)
        # compiler_workdir = "compiler_workdir"
        compiler_workdir = "workdir_{}_s{}_b{}_{}_{}".format(name, max_length, batch_size, auto_cast, auto_cast_type)
        os.makedirs(saved_dir, exist_ok=True)
        shutil.rmtree(compiler_workdir, ignore_errors=True)
        shutil.rmtree(model_file, ignore_errors=True)
        model = AutoModelForSequenceClassification.from_pretrained(name, torchscript=True)
        # Run precompiled TorchScript that is optimized by AWS Neuron Inf2
        neuron_model = torch_neuronx.trace(model, paraphrase, compiler_args=compiler_args, compiler_workdir=compiler_workdir)
        # Save the TorchScript for later use
        neuron_model.save(model_file)

    # Verify the TorchScript works on both example inputs
    # paraphrase_neuron_logits = neuron_model(*paraphrase)
    # not_paraphrase_neuron_logits = neuron_model(*not_paraphrase)

    print("== input shape ==")
    print(paraphrase[0].shape)

    example = paraphrase        # Use the paraphrase example generated during tracing
    # Load models
    models = [torch.jit.load(model_file) for _ in range(n_models)]

    # Warmup
    for _ in range(8):
        for model in models:
            model(*example)

    latencies = []

    # Thread task: Each of these thread tasks executes in a serial loop for a single model.
    #              Multiple of these threads are launched to achieve parallelism.
    def task(model):
        print("batch per thread: ", batches_per_thread)
        for _ in range(batches_per_thread):
            start = time.time()
            model(*example)
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
            infa_rslts = fut.result()
    end = time.time()

    # Compute metrics
    duration = end - begin
    inferences = len(latencies) * batch_size
    throughput = inferences / duration
    # avg_latency = (sum(latencies) * 1000) / len(latencies)
    avg_latency = np.mean(latencies)*1000.0
    p50_latency = np.percentile(latencies, 50)*1000.0
    p90_latency = np.percentile(latencies, 90)*1000.0
    p99_latency = np.percentile(latencies, 99)*1000.0

    # Metrics
    metrics = {
        'filename': str(filename),
        'batch_size': batch_size,
        'batches': len(latencies),
        'inferences': inferences,
        'workers': n_workers,
        'models': n_models,
        'duration': duration,
        'throughput': throughput,
        'Avg. latency': avg_latency,
        'P50 latency': p50_latency,
        'P90 latency': p90_latency,
        'P99 latency': p99_latency,
    }
    display(metrics)

    # Metrics
    metrics_csv = {
        'Filename': [str(filename)],
        'batch size': [batch_size],
        'n_workers': [n_workers],
        'n_models': [n_models],
        'Throughput': [throughput],
        'Avg. latency': [avg_latency],
        'P50 latency': [p50_latency],
        'P90 latency': [p90_latency],
        'P99 latency': [p99_latency],
    }
    df = pd.DataFrame(metrics_csv)
    csv_file = "neuron_{}_s{}_b{}_{}_{}_w{}_m{}.csv".format(name.replace('/', '_'), max_length, batch_size, auto_cast, auto_cast_type, n_workers, n_models)
    df.to_csv(csv_file)
    print("Save results to {}".format(csv_file))

