# DistilBERT Triton Inference Server on AWS Neuron

End-to-end benchmarks of DistilBERT (distilbert-base-uncased, 67M params) served via Triton Inference Server on AWS Neuron instances. Each notebook is fully self-contained -- it compiles models, builds the Docker image, starts Triton, and runs the benchmark.

## Results

| Instance | Config | Peak Throughput | Best Latency-Throughput | Lowest P50 |
|----------|--------|-----------------|------------------------|------------|
| trn2.3xlarge | LNC=2 (4 cores) | 17,316 inf/sec | 11,411 inf/sec @ 1.88ms | 1.88ms |
| trn2.3xlarge | LNC=1 (8 cores) | 21,289 inf/sec | 14,328 inf/sec @ 3.79ms | 3.79ms |
| inf2.xlarge | 2 cores | 4,986 inf/sec | 3,996 inf/sec @ 2.99ms | 2.99ms |

All results use `--auto-cast matmult` (BF16 matrix multiplications, FP32 output). Input: 128 tokens, output: 768-dim CLS embedding.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `distilbert_triton_trainium2.ipynb` | Trainium2 with LNC=2 (source) |
| `distilbert_triton_trainium2_lnc1.ipynb` | Trainium2 with LNC=1 (source) |
| `distilbert_triton_inf2.ipynb` | Inferentia2 (source) |
| `*_executed.ipynb` | Pre-run versions with saved outputs |
