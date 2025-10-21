"""
Benchmark process management for neuronx benchmarking automation.

This module handles the lifecycle of benchmark processes including:
- Benchmark execution with batch-specific environment variables
- Process monitoring and timeout handling
- Output collection and file management
- Benchmark cleanup functions
"""

import os
import time
import signal
import subprocess
import logging
import glob
import shutil
import atexit
import threading
from typing import Optional, Dict, List
from datetime import datetime, timedelta

# Optional dependency - psutil for advanced process management
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

from config import (
    BenchmarkConfig, 
    BenchmarkResult,
    ProcessStatus,
    BENCHMARK_TIMEOUT,
    get_benchmark_command,
    LLMPERF_SCRIPT
)
from environment_utils import (
    prepare_benchmark_environment,
    create_shell_command,
    setup_process_directories
)


class BenchmarkManager:
    """
    Manages the lifecycle of a benchmark process.
    
    This class handles benchmark execution, monitoring, and result collection
    for a single batch size run.
    """
    
    def __init__(self, config: BenchmarkConfig):
        """
        Initialize the benchmark manager.
        
        Args:
            config: Benchmark configuration for this run
        """
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.status = ProcessStatus()
        self.logger = logging.getLogger(f'neuronx_benchmark.benchmark.batch_{config.batch_size}')
        self.directories: Dict[str, str] = {}
        self.result: Optional[BenchmarkResult] = None
        self._cleanup_lock = threading.Lock()
        self._cleanup_completed = False
        
        # Register cleanup function for this instance
        atexit.register(self._emergency_cleanup)
        
    def execute_benchmark(self) -> bool:
        """
        Execute the benchmark process with batch-specific environment variables.
        
        Sets up the environment with LLM_PERF_CONCURRENT and LLM_PERF_MAX_REQUESTS
        based on the batch size, then starts the benchmark process.
        
        Returns:
            True if benchmark started successfully, False otherwise
        """
        self.logger.info(f"Starting benchmark for batch size {self.config.batch_size}")
        
        try:
            # Set up directories for this batch size
            self.directories = setup_process_directories(self.config.batch_size)
            
            # Prepare batch-specific environment variables
            env = prepare_benchmark_environment(self.config.batch_size)
            
            # Get benchmark command
            command = get_benchmark_command(self.config.batch_size, self.directories['results'])
            
            # Create shell command with virtual environment activation
            shell_command = create_shell_command(command, env)
            
            # Change to the directory containing the benchmark script
            # This ensures the script can find its dependencies
            benchmark_cwd = self._find_benchmark_script_directory()
            if not benchmark_cwd:
                self.logger.error(f"Could not find benchmark script {LLMPERF_SCRIPT}")
                return False
            
            self.logger.info(f"Benchmark command: {' '.join(command)}")
            self.logger.debug(f"Full shell command: {' '.join(shell_command)}")
            self.logger.info(f"LLM_PERF_CONCURRENT: {env.get('LLM_PERF_CONCURRENT')}")
            self.logger.info(f"LLM_PERF_MAX_REQUESTS: {env.get('LLM_PERF_MAX_REQUESTS')}")
            
            # Set up log files
            benchmark_log_path = os.path.join(self.directories['logs'], 'benchmark.log')
            benchmark_error_path = os.path.join(self.directories['logs'], 'benchmark_error.log')
            
            # Log the exact command for replication
            with open(benchmark_log_path, 'w') as f:
                f.write("=== BENCHMARK COMMAND FOR REPLICATION ===\n")
                f.write(f"# Batch size: {self.config.batch_size}\n")
                f.write(f"# Environment variables:\n")
                for key, value in env.items():
                    if key.startswith(('LLM_PERF_', 'OPENAI_', 'TP')):
                        f.write(f"export {key}={value}\n")
                f.write(f"# Working directory: {benchmark_cwd}\n")
                f.write(f"# Full command to replicate this benchmark:\n")
                f.write(f"cd {benchmark_cwd}\n")
                f.write(f"{' '.join(shell_command)}\n")
                f.write("=" * 50 + "\n\n")
            
            # Start the benchmark process
            with open(benchmark_log_path, 'a') as stdout_file, \
                 open(benchmark_error_path, 'w') as stderr_file:
                
                self.process = subprocess.Popen(
                    shell_command,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    env=env,
                    cwd=benchmark_cwd,
                    preexec_fn=os.setsid  # Create new process group for easier cleanup
                )
            
            # Update status
            self.status.pid = self.process.pid
            self.status.status = 'running'
            self.status.start_time = datetime.now()
            
            self.logger.info(f"Benchmark process started with PID {self.process.pid}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start benchmark: {e}")
            self.status.status = 'failed'
            self.status.error_message = str(e)
            return False
    
    def monitor_benchmark(self) -> bool:
        """
        Monitor the benchmark process and handle timeout.
        
        Waits for the benchmark to complete within the timeout period (1800s).
        Monitors process status and handles both successful completion and timeouts.
        
        Returns:
            True if benchmark completed successfully, False if failed or timed out
        """
        if not self.process:
            self.logger.error("No benchmark process to monitor")
            return False
        
        self.logger.info(f"Monitoring benchmark process (timeout: {BENCHMARK_TIMEOUT}s)")
        
        start_time = datetime.now()
        timeout = timedelta(seconds=BENCHMARK_TIMEOUT)
        
        while datetime.now() - start_time < timeout:
            # Check if process has completed
            poll_result = self.process.poll()
            if poll_result is not None:
                if poll_result == 0:
                    self.status.status = 'completed'
                    elapsed = datetime.now() - start_time
                    self.logger.info(f"Benchmark completed successfully after {elapsed.total_seconds():.1f} seconds")
                    return True
                else:
                    self.status.status = 'failed'
                    self.status.error_message = f"Process exited with code {poll_result}"
                    self.logger.error(f"Benchmark failed with exit code {poll_result}")
                    return False
            
            # Wait before next check
            time.sleep(10)  # Check every 10 seconds
            
            # Log progress every 2 minutes
            elapsed = datetime.now() - start_time
            if elapsed.total_seconds() % 120 < 10:
                self.logger.info(f"Benchmark still running... ({elapsed.total_seconds():.0f}s elapsed)")
        
        # Timeout reached
        self.logger.error(f"Benchmark timeout after {BENCHMARK_TIMEOUT} seconds")
        self.status.status = 'failed'
        self.status.error_message = "Benchmark timeout"
        
        # Terminate the timed-out process
        self._terminate_benchmark_process()
        return False
    
    def collect_benchmark_output(self) -> BenchmarkResult:
        """
        Collect and parse benchmark output files.
        
        Searches for result files in the results directory and extracts
        performance metrics from the benchmark output.
        
        Returns:
            BenchmarkResult with extracted metrics or error information
        """
        self.logger.info("Collecting benchmark output files")
        
        result = BenchmarkResult(
            batch_size=self.config.batch_size,
            max_num_seqs=self.config.max_num_seqs
        )
        
        try:
            # Find result files in the results directory
            result_files = self._find_result_files()
            
            if not result_files:
                self.logger.warning("No result files found")
                result.error_message = "No result files found"
                return result
            
            # Parse the most recent result file
            latest_file = max(result_files, key=os.path.getctime)
            self.logger.info(f"Parsing result file: {latest_file}")
            
            metrics = self._parse_result_file(latest_file)
            if metrics:
                result.throughput_tok_per_sec = metrics.get('throughput_tok_per_sec')
                result.mean_inter_token_latency_s = metrics.get('mean_inter_token_latency_s')
                result.mean_ttft_s = metrics.get('mean_ttft_s')
                result.mean_end_to_end_latency_s = metrics.get('mean_end_to_end_latency_s')
                result.success = True
                
                self.logger.info(f"Successfully extracted metrics: throughput={result.throughput_tok_per_sec}")
            else:
                result.error_message = "Failed to parse result file"
                self.logger.error("Failed to parse result file")
        
        except Exception as e:
            self.logger.error(f"Error collecting benchmark output: {e}")
            result.error_message = f"Output collection error: {e}"
        
        self.result = result
        return result
    
    def cleanup_benchmark(self) -> None:
        """
        Clean up benchmark process and temporary files.
        
        Ensures the benchmark process is terminated and cleans up
        any temporary files or resources.
        """
        with self._cleanup_lock:
            if self._cleanup_completed:
                return
            
            self.logger.info("Cleaning up benchmark resources")
            
            try:
                # Ensure process is terminated
                if self.process and self.process.poll() is None:
                    self._terminate_benchmark_process()
                
                # Clean up any child processes
                self._cleanup_child_processes()
                
                # Close process handles
                if self.process:
                    try:
                        if self.process.stdout and not self.process.stdout.closed:
                            self.process.stdout.close()
                        if self.process.stderr and not self.process.stderr.closed:
                            self.process.stderr.close()
                    except Exception as e:
                        self.logger.debug(f"Error closing process handles: {e}")
                
                # Clean up temporary files and directories
                self._cleanup_temporary_files()
                
                # Archive result files to prevent conflicts with next run
                self._archive_result_files()
                
                # Mark cleanup as completed
                self._cleanup_completed = True
                
                self.logger.info("Benchmark resource cleanup completed")
                
            except Exception as e:
                self.logger.error(f"Error during benchmark cleanup: {e}")
    
    def _emergency_cleanup(self) -> None:
        """
        Emergency cleanup function called on process exit.
        
        This ensures resources are cleaned up even if the normal
        cleanup process fails or is not called.
        """
        try:
            if not self._cleanup_completed and self.process:
                # Force terminate any remaining processes
                if self.process.poll() is None:
                    try:
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass
                
                # Clean up child processes
                self._cleanup_child_processes()
                
        except Exception:
            # Avoid exceptions during emergency cleanup
            pass
    
    def _cleanup_temporary_files(self) -> None:
        """
        Clean up temporary files and directories created during benchmark execution.
        """
        try:
            # Clean up any temporary benchmark files
            temp_patterns = [
                '/tmp/benchmark_*',
                '/tmp/llmperf_*',
                '/tmp/vllm_*'
            ]
            
            for pattern in temp_patterns:
                for temp_file in glob.glob(pattern):
                    try:
                        if os.path.isfile(temp_file):
                            os.remove(temp_file)
                            self.logger.debug(f"Removed temporary file: {temp_file}")
                        elif os.path.isdir(temp_file):
                            shutil.rmtree(temp_file)
                            self.logger.debug(f"Removed temporary directory: {temp_file}")
                    except Exception as e:
                        self.logger.debug(f"Could not remove temporary file {temp_file}: {e}")
            
            # Clean up any lock files
            lock_patterns = [
                '*.lock',
                '.benchmark_*.lock'
            ]
            
            for pattern in lock_patterns:
                for lock_file in glob.glob(pattern):
                    try:
                        os.remove(lock_file)
                        self.logger.debug(f"Removed lock file: {lock_file}")
                    except Exception as e:
                        self.logger.debug(f"Could not remove lock file {lock_file}: {e}")
                        
        except Exception as e:
            self.logger.warning(f"Error cleaning up temporary files: {e}")
    
    def _find_benchmark_script_directory(self) -> Optional[str]:
        """
        Find the directory containing the benchmark script.
        
        Returns:
            Path to directory containing the benchmark script, or None if not found
        """
        # Look for the script in common locations
        search_paths = [
            ".",  # Current directory
            "llmperf",  # llmperf subdirectory
            "../llmperf",  # Parent's llmperf directory
            "upstreaming-to-vllm/llmperf",  # In the vllm repo
            "../upstreaming-to-vllm/llmperf",  # Parent's vllm repo
        ]
        
        for path in search_paths:
            script_path = os.path.join(path, LLMPERF_SCRIPT)
            if os.path.exists(script_path):
                self.logger.debug(f"Found benchmark script at: {script_path}")
                return os.path.abspath(path)
        
        return None
    
    def _find_result_files(self) -> List[str]:
        """
        Find result files in the results directory.
        
        Returns:
            List of paths to result files
        """
        if not self.directories.get('results'):
            return []
        
        # First check for JSON result files (if they exist)
        patterns = [
            os.path.join(self.directories['results'], "*.json"),
            os.path.join(self.directories['results'], "results_*.txt"),
            os.path.join(self.directories['results'], "*_results.json"),
            os.path.join(self.directories['results'], "token_benchmark_*.json"),
        ]
        
        result_files = []
        for pattern in patterns:
            result_files.extend(glob.glob(pattern))
        
        # If no JSON files found, check for benchmark log file
        if not result_files:
            log_file = os.path.join(self.directories['logs'], 'benchmark.log')
            if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
                result_files.append(log_file)
        
        return result_files
    
    def _parse_result_file(self, file_path: str) -> Optional[Dict[str, float]]:
        """
        Parse a result file to extract performance metrics.
        
        Args:
            file_path: Path to the result file
            
        Returns:
            Dictionary with extracted metrics, or None if parsing failed
        """
        try:
            # Check if this is a log file (benchmark.log)
            if file_path.endswith('benchmark.log'):
                return self._parse_benchmark_log(file_path)
            
            # Try using the enhanced results processor for JSON files
            try:
                from results_processor import ResultsProcessor
                processor = ResultsProcessor()
                metrics = processor.parse_result_file(file_path)
                return metrics
            except Exception:
                # Fall back to direct log parsing if results processor fails
                return self._parse_benchmark_log(file_path)
            
        except Exception as e:
            self.logger.error(f"Error parsing result file {file_path}: {e}")
            return None
    
    def _parse_benchmark_log(self, log_path: str) -> Optional[Dict[str, float]]:
        """
        Parse benchmark.log file to extract performance metrics.
        
        Args:
            log_path: Path to the benchmark log file
            
        Returns:
            Dictionary with extracted metrics, or None if parsing failed
        """
        try:
            with open(log_path, 'r') as f:
                content = f.read()
            
            metrics = {}
            
            # Parse key metrics from the log output
            # Look for patterns like "mean = 0.025393438751946773"
            import re
            
            # Extract inter-token latency mean
            match = re.search(r'inter_token_latency_s\s*\n.*?mean = ([\d.]+)', content, re.DOTALL)
            if match:
                metrics['mean_inter_token_latency_s'] = float(match.group(1))
            
            # Extract TTFT mean
            match = re.search(r'ttft_s\s*\n.*?mean = ([\d.]+)', content, re.DOTALL)
            if match:
                metrics['mean_ttft_s'] = float(match.group(1))
            
            # Extract end-to-end latency mean
            match = re.search(r'end_to_end_latency_s\s*\n.*?mean = ([\d.]+)', content, re.DOTALL)
            if match:
                metrics['mean_end_to_end_latency_s'] = float(match.group(1))
            
            # Extract overall throughput
            match = re.search(r'Overall Output Throughput: ([\d.]+)', content)
            if match:
                metrics['throughput_tok_per_sec'] = float(match.group(1))
            
            self.logger.info(f"Parsed metrics from log: {metrics}")
            return metrics if metrics else None
            
        except Exception as e:
            self.logger.error(f"Error parsing benchmark log {log_path}: {e}")
            return None
    
    def _parse_text_results(self, content: str) -> Dict[str, str]:
        """
        Parse text-based result content.
        
        Args:
            content: Text content to parse
            
        Returns:
            Dictionary with parsed key-value pairs
        """
        data = {}
        lines = content.strip().split('\n')
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                data[key.strip().lower()] = value.strip()
        
        return data
    
    def _terminate_benchmark_process(self) -> None:
        """
        Terminate the benchmark process gracefully.
        """
        if not self.process:
            return
        
        self.logger.info(f"Terminating benchmark process {self.process.pid}")
        
        try:
            # First, try graceful termination
            if self.process.poll() is None:  # Process is still running
                # Send SIGTERM to the process group
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                
                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=10)
                    self.logger.info("Benchmark terminated gracefully")
                except subprocess.TimeoutExpired:
                    self.logger.warning("Graceful shutdown timed out, forcing termination")
                    
                    # Force termination
                    try:
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                        self.process.wait(timeout=5)
                        self.logger.info("Benchmark terminated forcefully")
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        self.logger.error("Failed to force terminate benchmark")
        
        except Exception as e:
            self.logger.error(f"Error terminating benchmark process: {e}")
    
    def _cleanup_child_processes(self) -> None:
        """
        Clean up any child processes that might still be running.
        """
        if not self.process or not self.process.pid or not HAS_PSUTIL:
            return
        
        try:
            parent = psutil.Process(self.process.pid)
            children = parent.children(recursive=True)
            
            for child in children:
                try:
                    self.logger.debug(f"Terminating child process {child.pid}")
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            
            # Wait for children to terminate
            gone, alive = psutil.wait_procs(children, timeout=5)
            
            # Force kill any remaining children
            for child in alive:
                try:
                    self.logger.debug(f"Force killing child process {child.pid}")
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
                    
        except psutil.NoSuchProcess:
            pass
        except Exception as e:
            self.logger.warning(f"Error cleaning up child processes: {e}")
    
    def _archive_result_files(self) -> None:
        """
        Archive result files to prevent conflicts with subsequent runs.
        """
        if not self.directories.get('results'):
            return
        
        try:
            # Create archive directory
            archive_dir = os.path.join(self.directories['results'], 'archived')
            os.makedirs(archive_dir, exist_ok=True)
            
            # Move result files to archive
            result_files = self._find_result_files()
            for file_path in result_files:
                if os.path.exists(file_path):
                    filename = os.path.basename(file_path)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    archived_name = f"{timestamp}_{filename}"
                    archived_path = os.path.join(archive_dir, archived_name)
                    
                    shutil.move(file_path, archived_path)
                    self.logger.debug(f"Archived result file: {archived_path}")
        
        except Exception as e:
            self.logger.warning(f"Error archiving result files: {e}")
    
    def get_status(self) -> ProcessStatus:
        """
        Get the current status of the benchmark process.
        
        Returns:
            Current process status
        """
        # Update status based on process state
        if self.process:
            poll_result = self.process.poll()
            if poll_result is not None and self.status.status == 'running':
                if poll_result == 0:
                    self.status.status = 'completed'
                else:
                    self.status.status = 'failed'
                    self.status.error_message = f"Process exited with code {poll_result}"
        
        return self.status
    
    def get_result(self) -> Optional[BenchmarkResult]:
        """
        Get the benchmark result.
        
        Returns:
            BenchmarkResult if available, None otherwise
        """
        return self.result
    
    def get_log_paths(self) -> Dict[str, str]:
        """
        Get paths to benchmark log files.
        
        Returns:
            Dictionary with log file paths
        """
        if not self.directories:
            return {}
        
        return {
            'stdout': os.path.join(self.directories['logs'], 'benchmark.log'),
            'stderr': os.path.join(self.directories['logs'], 'benchmark_error.log')
        }
    
    def is_running(self) -> bool:
        """
        Check if the benchmark process is currently running.
        
        Returns:
            True if benchmark is running, False otherwise
        """
        if not self.process:
            return False
        
        return self.process.poll() is None
    
    def get_runtime(self) -> Optional[timedelta]:
        """
        Get the runtime of the benchmark process.
        
        Returns:
            Runtime as timedelta, or None if not started
        """
        if not self.status.start_time:
            return None
        
        return datetime.now() - self.status.start_time


def create_benchmark_manager(batch_size: int) -> BenchmarkManager:
    """
    Create a benchmark manager for the specified batch size.
    
    Args:
        batch_size: The batch size for this benchmark run
        
    Returns:
        Configured BenchmarkManager instance
    """
    config = BenchmarkConfig.from_batch_size(batch_size)
    return BenchmarkManager(config)


def run_benchmark_for_batch(batch_size: int) -> Optional[BenchmarkResult]:
    """
    Run a complete benchmark for the specified batch size.
    
    This is a convenience function that creates a benchmark manager,
    executes the benchmark, monitors it, and collects results.
    
    Args:
        batch_size: The batch size for this benchmark run
        
    Returns:
        BenchmarkResult if successful, None if failed
    """
    logger = logging.getLogger('neuronx_benchmark.benchmark_execution')
    
    try:
        # Create benchmark manager
        benchmark_manager = create_benchmark_manager(batch_size)
        
        # Execute benchmark
        if not benchmark_manager.execute_benchmark():
            logger.error(f"Failed to start benchmark for batch size {batch_size}")
            return None
        
        # Monitor benchmark execution
        success = benchmark_manager.monitor_benchmark()
        
        # Collect results regardless of success/failure
        result = benchmark_manager.collect_benchmark_output()
        
        # Clean up
        benchmark_manager.cleanup_benchmark()
        
        if success:
            logger.info(f"Benchmark for batch size {batch_size} completed successfully")
        else:
            logger.error(f"Benchmark for batch size {batch_size} failed")
        
        return result
        
    except Exception as e:
        logger.error(f"Error running benchmark for batch size {batch_size}: {e}")
        return None