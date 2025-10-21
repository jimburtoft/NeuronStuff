"""
Configuration constants and environment setup for neuronx benchmarking automation.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime


# Batch size configuration
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 96, 128, 256]
BATCH_SIZE_REDUCTION_STEP = 4  # Step size for finding maximum working batch size

# Timeout configuration (in seconds)
SERVER_STARTUP_TIMEOUT = 3600  # 60 minutes
BENCHMARK_TIMEOUT = 1800       # 30 minutes
HEALTH_CHECK_INTERVAL = 5      # 5 seconds between health checks

# Server configuration
SERVER_PORT = 8000
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
TENSOR_PARALLEL_SIZE = 8
MAX_MODEL_LEN = 2048
# BLOCK_SIZE = 2048  # Optional: Should match MAX_MODEL_LEN (sequence length) when used
BLOCK_SIZE = None    # Set to None to disable, or set to MAX_MODEL_LEN if needed
DEVICE = "neuron"

# Paths
VENV_PATH = "/opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate"
RESULTS_BASE_DIR = "results"
LLMPERF_SCRIPT = "token_benchmark_ray.py"

# Environment variables for server
SERVER_ENV_VARS = {
    "NEURON_CONTEXT_LENGTH_BUCKETS": "128, 512, 1024, 2048",
    "VLLM_NEURON_FRAMEWORK": "neuronx-distributed-inference",
}

# Base environment variables for benchmark
BASE_BENCHMARK_ENV_VARS = {
    "OPENAI_API_KEY": "dummy",
    "TP": "8",
    "OPENAI_API_BASE": f"http://localhost:{SERVER_PORT}/v1",
}


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    batch_size: int
    max_num_seqs: int
    port: int = SERVER_PORT
    model_name: str = MODEL_NAME
    tensor_parallel_size: int = TENSOR_PARALLEL_SIZE
    max_model_len: int = MAX_MODEL_LEN
    block_size: Optional[int] = BLOCK_SIZE
    device: str = DEVICE
    
    @classmethod
    def from_batch_size(cls, batch_size: int) -> 'BenchmarkConfig':
        """Create configuration from batch size."""
        return cls(
            batch_size=batch_size,
            max_num_seqs=2 * batch_size
        )


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    batch_size: int
    max_num_seqs: int
    throughput_tok_per_sec: Optional[float] = None
    mean_inter_token_latency_s: Optional[float] = None
    mean_ttft_s: Optional[float] = None
    mean_end_to_end_latency_s: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class ProcessStatus:
    """Status of a running process."""
    pid: Optional[int] = None
    status: str = 'starting'  # 'starting', 'running', 'ready', 'failed', 'completed'
    start_time: Optional[datetime] = None
    error_message: Optional[str] = None


def get_server_environment() -> Dict[str, str]:
    """
    Get environment variables for server process.
    
    Returns:
        Dictionary of environment variables for server startup
    """
    env = os.environ.copy()
    env.update(SERVER_ENV_VARS)
    return env


def get_benchmark_environment(batch_size: int) -> Dict[str, str]:
    """
    Get environment variables for benchmark process.
    
    Args:
        batch_size: The batch size for this benchmark run
        
    Returns:
        Dictionary of environment variables for benchmark execution
    """
    env = os.environ.copy()
    env.update(BASE_BENCHMARK_ENV_VARS)
    
    # Set batch-specific environment variables
    env["LLM_PERF_CONCURRENT"] = str(batch_size)
    env["LLM_PERF_MAX_REQUESTS"] = str(100 * batch_size)
    
    # Add timestamp for result identification
    date_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    env["date_str"] = date_str
    
    return env


def get_results_directory(batch_size: int) -> str:
    """
    Get the results directory path for a specific batch size.
    
    Args:
        batch_size: The batch size
        
    Returns:
        Path to the results directory for this batch size
    """
    return os.path.join(RESULTS_BASE_DIR, f"batch_{batch_size}")


def ensure_results_directory(batch_size: int) -> str:
    """
    Ensure the results directory exists for a specific batch size.
    
    Args:
        batch_size: The batch size
        
    Returns:
        Path to the results directory
    """
    results_dir = get_results_directory(batch_size)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def get_csv_filename() -> str:
    """
    Generate a timestamped CSV filename for consolidated results.
    
    Returns:
        Filename for the consolidated results CSV
    """
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    return f"consolidated_results_{timestamp}.csv"


def get_server_command(config: BenchmarkConfig) -> List[str]:
    """
    Generate the server startup command.
    
    Args:
        config: Benchmark configuration
        
    Returns:
        List of command arguments for server startup
    """
    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", config.model_name,
        "--tensor-parallel-size", str(config.tensor_parallel_size),
        "--max-model-len", str(config.max_model_len),
    ]
    
    # Only add block-size if it's configured
    if config.block_size is not None:
        command.extend(["--block-size", str(config.block_size)])
    
    command.extend([
        "--device", config.device,
        "--port", str(config.port),
        "--max-num-seqs", str(config.max_num_seqs)
    ])
    
    return command


def get_benchmark_command(batch_size: int, results_dir: str) -> List[str]:
    """
    Generate the benchmark execution command.
    
    Args:
        batch_size: The batch size for this run
        results_dir: Directory to store results
        
    Returns:
        List of command arguments for benchmark execution
    """
    return [
        "python", LLMPERF_SCRIPT,
        "--model", MODEL_NAME,
        "--results-dir", results_dir,
        "--num-concurrent-requests", str(batch_size),
        "--max-num-completed-requests", str(100 * batch_size)
    ]


def setup_environment() -> None:
    """
    Set up the basic environment for the benchmarking system.
    This function ensures all necessary directories exist and
    basic environment variables are set.
    """
    # Ensure base results directory exists
    os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
    
    # Set any global environment variables if needed
    # (Most environment setup is done per-process)
    pass


def validate_environment() -> bool:
    """
    Validate that the required environment is available.
    
    Returns:
        True if environment is valid, False otherwise
    """
    # Check if virtual environment path exists
    if not os.path.exists(VENV_PATH):
        print(f"Error: Virtual environment not found at {VENV_PATH}")
        return False
    
    # Check if we can create results directory
    try:
        os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
    except Exception as e:
        print(f"Error: Cannot create results directory: {e}")
        return False
    
    return True