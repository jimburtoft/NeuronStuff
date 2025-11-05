import torch
import neuronx_distributed
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import transformers
import concurrent.futures
import argparse
import os
import time
import numpy as np
import shutil
import pandas as pd
from neuronx_distributed.parallel_layers import layers, parallel_state
from neuronx_distributed import ModelBuilder, shard_checkpoint
from neuronx_distributed.utils.model_utils import init_on_device


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
    parser.add_argument("--buckets", type=int, nargs='+', default=[128, 256, 512, 1024, 2048], help="Bucket sizes for multi-bucket tracing")
    parser.add_argument("--model", type=str, default="clapAI/modernBERT-large-multilingual-sentiment", help="HuggingFace model name or local model directory")
    parser.add_argument("--saved_dir", type=str, default="saved_models", help="HuggingFace model name or local model directory")
    parser.add_argument("--auto_cast", type=str, default="none", help="compiler args for auto cast")
    parser.add_argument("--auto_cast_type", type=str, default="", help="compiler args for auto cast type")
    parser.add_argument("--instance", type=str, default="inf2", help="instance to benchmark, ex: inf2, g5")
    parser.add_argument("--force_recompile", action="store_true", help="Force recompilation even if model exists")
    args = parser.parse_args()

    print("model name: ", args.model)
    print("batch size: ", args.bs)
    print("n_workers: ", args.n_workers)
    print("n_models: ", args.n_models)
    print("max_length: ", args.max_length)
    print("buckets: ", args.buckets)
    print("saved dir: ", args.saved_dir)
    print("instance: ", args.instance)

    name = args.model
    batch_size = args.bs
    n_workers = args.n_workers
    n_models = args.n_models
    max_length = args.max_length
    buckets = args.buckets
    saved_dir = args.saved_dir
    auto_cast = args.auto_cast
    auto_cast_type = args.auto_cast_type
    instance = args.instance
    if auto_cast == "none" or auto_cast == "default":
        auto_cast_type = ""
    filename = "neuron_{}_s{}_b{}_{}_{}_buckets_{}.pt".format(name, max_length, batch_size, auto_cast, auto_cast_type, '_'.join(map(str, buckets)))
    batches_per_thread = 1

    # Build tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(name)
    
    # Setup some example inputs
    sequence_0 = "The company HuggingFace is based in New York City"
    sequence_1 = "Apples are especially bad for your health"
    sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

    # Create single input for tracing
    example_inputs = encode(tokenizer, sequence_0, sequence_2, max_length=max_length, batch_size=batch_size)

    model_file = os.path.join(saved_dir, filename)
    if os.path.isfile(model_file) and not args.force_recompile:
        print("Load models from ", model_file)
    else:
        if args.force_recompile:
            print("Force recompile enabled")
        print("Trace model and save it to", model_file)
        os.makedirs(saved_dir, exist_ok=True)
        shutil.rmtree(model_file, ignore_errors=True)
        
        try:
            # Preload base model and weights
            base_model = AutoModelForSequenceClassification.from_pretrained(name, torchscript=True)
            base_model.eval()
            
            class ModelWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, input_ids, attention_mask):
                    return self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            model = ModelWrapper(base_model)
            
            # Run forward pass to initialize all buffers
            with torch.no_grad():
                _ = model(*example_inputs)
            
            # Preload weights before compilation - get complete state dict
            model_checkpoint = model.state_dict()
            
            # Verify all expected weights are present
            config = base_model.config
            head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
            inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
            
            for layer_idx in range(config.num_hidden_layers):
                key = f"model.model.layers.{layer_idx}.attn.rotary_emb.inv_freq"
                if key not in model_checkpoint:
                    model_checkpoint[key] = inv_freq
            
            # Verify classifier weights exist
            assert 'model.classifier.weight' in model_checkpoint, "Missing classifier.weight"
            assert 'model.classifier.bias' in model_checkpoint, "Missing classifier.bias"
            
            print(f"Preloaded {len(model_checkpoint)} weight tensors")
            print(f"Classifier weight shape: {model_checkpoint['model.classifier.weight'].shape}")
            print(f"Classifier bias shape: {model_checkpoint['model.classifier.bias'].shape}")
            
            # Build and compile model
            nxd_model = ModelBuilder(model) \
                .trace(args=example_inputs, tag="main") \
                .compile()
            
            print("Compilation completed successfully")
            
            # Set weights after compilation
            nxd_model.set_weights([model_checkpoint])
            print("Weights set successfully")
            
        except Exception as e:
            print(f"Error during tracing: {str(e)}")
            raise
        
        # Save with weights embedded
        nxd_model.save(model_file, save_weights=True)
        print(f"Model saved with embedded weights to {model_file}")

    print("== input shape ==")
    print(example_inputs[0].shape)
    
    # Load models (weights already embedded)
    models = []
    for _ in range(n_models):
        model = torch.jit.load(model_file)
        model.to_neuron()
        models.append(model)
    
    print(f"Loaded {n_models} model(s) with embedded weights")

    # Warmup
    for _ in range(8):
        for model in models:
            model(list(example_inputs), {})

    latencies = []

    def task(model):
        print("batch per thread: ", batches_per_thread)
        for _ in range(batches_per_thread):
            start = time.time()
            model(list(example_inputs), {})
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
    csv_file = "neuron_{}_s{}_b{}_{}_{}_buckets_{}_w{}_m{}.csv".format(name.replace('/', '_'), max_length, batch_size, auto_cast, auto_cast_type, '_'.join(map(str, buckets)), n_workers, n_models)
    df.to_csv(csv_file)
    print("Save results to {}".format(csv_file))