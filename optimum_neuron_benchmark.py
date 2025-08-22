import torch
from transformers import AutoTokenizer
from optimum.neuron import NeuronModelForSequenceClassification
import concurrent.futures
import argparse
import time
import numpy as np
import pandas as pd
import os

"""
optimum-cli export neuron \
  --model clapAI/modernBERT-base-multilingual-sentiment \
  --sequence_length 2048 \
  --auto_cast none \
  --batch_size 1 \
  modernBERT-base-multilingual-sentiment_neuronS2048BS1

Match seq length and batch size then run with:  

python optimum_neuron_benchmark.py \
    --neuron_model_path modernBERT-base-multilingual-sentiment_neuronS2048BS1 \
    --n_workers 1 --n_models 1 --bs 1 --max_length 2048

"""

def encode(tokenizer, *inputs, max_length=128, batch_size=1):
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
    pad = max(map(len, metrics)) + 1
    for key, value in metrics.items():
        parts = key.split('_')
        parts = list(map(str.title, parts))
        title = ' '.join(parts) + ":"
        if isinstance(value, float):
            value = f'{value:0.3f}'
        print(f'{title :<{pad}} {value}')

if __name__ == '__main__':
    #os.environ["NEURON_CC_FLAGS"] = "--optlevel 3"
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=1, help="batch size")
    parser.add_argument("--n_workers", type=int, default=1, help="total threads")
    parser.add_argument("--n_models", type=int, default=1, help="number of model instances")
    parser.add_argument("--max_length", type=int, default=117, help="sequence length")
    parser.add_argument("--neuron_model_path", type=str, default="modernBERT-base-multilingual-sentiment_neuron", help="compiled Neuron model path")
    parser.add_argument("--original_model", type=str, default="clapAI/modernBERT-base-multilingual-sentiment", help="original HuggingFace model")
    args = parser.parse_args()

    print("neuron model path:", args.neuron_model_path)
    print("batch size:", args.bs)
    print("n_workers:", args.n_workers)
    print("n_models:", args.n_models)
    print("max_length:", args.max_length)

    batch_size = args.bs
    n_workers = args.n_workers
    n_models = args.n_models
    max_length = args.max_length
    batches_per_thread = 100

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.original_model)
    
    # Setup example inputs
    sequence_0 = "Hamilton is considered to be the best musical of past years."
    sequence_1 = "The company HuggingFace is based in New York City"
    
    example = encode(tokenizer, sequence_0, max_length=max_length, batch_size=batch_size)
    
    # Load OptimumNeuron models
    models = [NeuronModelForSequenceClassification.from_pretrained(args.neuron_model_path) for _ in range(n_models)]

    # Warmup
    for _ in range(8):
        for model in models:
            model(input_ids=example[0], attention_mask=example[1])

    latencies = []

    def task(model):
        for _ in range(batches_per_thread):
            start = time.time()
            model(input_ids=example[0], attention_mask=example[1])
            finish = time.time()
            latencies.append(finish - start)

    # Run benchmark
    begin = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        fut_list = [pool.submit(task, models[i % len(models)]) for i in range(n_workers)]
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

    metrics = {
        'batch_size': batch_size,
        'batches': len(latencies),
        'inferences': inferences,
        'workers': n_workers,
        'models': n_models,
        'duration': duration,
        'throughput': throughput,
        'Avg_latency_ms': avg_latency,
        'P50_latency_ms': p50_latency,
        'P90_latency_ms': p90_latency,
        'P99_latency_ms': p99_latency,
    }
    display(metrics)

    # Save to CSV
    df = pd.DataFrame({k: [v] for k, v in metrics.items()})
    csv_file = f"optimum_neuron_benchmark_b{batch_size}_w{n_workers}_m{n_models}.csv"
    df.to_csv(csv_file, index=False)
    print(f"Results saved to {csv_file}")