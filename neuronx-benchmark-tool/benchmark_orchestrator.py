"""
Main orchestration logic for neuronx benchmarking automation.

This module implements the main orchestration logic that:
- Iterates through batch sizes [1, 2, 4, 8, 16, 32, 64, 96, 128, 256]
- Calculates max_num_seqs (2 * batch_size) for each iteration
- Implements single-attempt failure handling with logging and skip-to-next logic
- Handles batch size reduction logic for memory failures (256 → 252 → 248 → 244)
- Provides comprehensive error logging and progress tracking
"""

import os
import sys
import logging
import signal
import time
import atexit
import threading
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from config import (
    BATCH_SIZES,
    BATCH_SIZE_REDUCTION_STEP,
    BenchmarkConfig,
    BenchmarkResult,
    validate_environment,
    setup_environment,
    get_csv_filename
)
from server_manager import ServerManager, create_server_manager
from benchmark_manager import BenchmarkManager, create_benchmark_manager


class BenchmarkOrchestrator:
    """
    Main orchestrator for the neuronx benchmarking automation system.
    
    This class manages the complete benchmarking workflow across multiple
    batch sizes, handling failures and collecting results.
    """
    
    def __init__(self):
        """Initialize the benchmark orchestrator."""
        self.logger = logging.getLogger('neuronx_benchmark.orchestrator')
        self.results: List[BenchmarkResult] = []
        self.current_server: Optional[ServerManager] = None
        self.current_benchmark: Optional[BenchmarkManager] = None
        self.shutdown_requested = False
        self.background_mode = False
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.heartbeat_stop_event = threading.Event()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGHUP, self._sighup_handler)  # Handle SSH disconnection
        
        # Register cleanup function to run on exit
        atexit.register(self._cleanup_on_exit)
    
    def run_complete_benchmark_suite(self) -> bool:
        """
        Run the complete benchmark suite across all batch sizes.
        
        Implements the main orchestration logic:
        - Iterates through batch sizes [1, 2, 4, 8, 16, 32, 64, 96, 128, 256]
        - Calculates max_num_seqs (2 * batch_size) for each iteration
        - Implements single-attempt failure handling with skip-to-next logic
        - Handles batch size reduction for memory failures
        
        Returns:
            True if suite completed (regardless of individual failures), False if critical error
        """
        self.logger.info("Starting complete benchmark suite")
        
        try:
            # Validate environment before starting
            if not validate_environment():
                self.logger.error("Environment validation failed")
                return False
            
            # Set up environment
            setup_environment()
            
            # Process each batch size
            batch_sizes_to_process = BATCH_SIZES.copy()
            
            last_successful_batch_size = 0
            
            for batch_size in batch_sizes_to_process:
                if self.shutdown_requested:
                    self.logger.info("Shutdown requested, stopping benchmark suite")
                    break
                
                self.logger.info(f"Processing batch size {batch_size}")
                
                # Try to run benchmark for this batch size
                success = self._run_single_batch_benchmark(batch_size)
                
                if success:
                    last_successful_batch_size = batch_size
                    self.logger.info(f"Batch size {batch_size} succeeded")
                else:
                    self.logger.warning(f"Batch size {batch_size} failed")
                    
                    # If we have a successful batch size, try to find the maximum working size
                    if last_successful_batch_size > 0:
                        self.logger.info(f"Attempting to find maximum batch size starting from last successful: {last_successful_batch_size}")
                        max_working_size = self._find_maximum_batch_size(last_successful_batch_size, batch_size)
                        if max_working_size > last_successful_batch_size:
                            self.logger.info(f"Found maximum working batch size: {max_working_size}")
                        else:
                            self.logger.info(f"Maximum working batch size remains: {last_successful_batch_size}")
                    
                    # Skip remaining larger batch sizes since we found the failure point
                    self.logger.info("Skipping remaining larger batch sizes since failure point found")
                    break
                
                # Clean up between batch sizes
                self._cleanup_current_processes()
                
                # Brief pause between batch sizes to ensure clean state
                if not self.shutdown_requested:
                    time.sleep(5)
            
            # Generate final results
            self._generate_consolidated_results()
            
            self.logger.info("Benchmark suite completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Critical error in benchmark suite: {e}")
            return False
        finally:
            # Ensure cleanup
            self._cleanup_current_processes()
    
    def _run_single_batch_benchmark(self, batch_size: int) -> bool:
        """
        Run benchmark for a single batch size with single-attempt failure handling.
        
        Args:
            batch_size: The batch size to benchmark
            
        Returns:
            True if benchmark completed successfully, False otherwise
        """
        self.logger.info(f"Starting benchmark for batch size {batch_size}")
        
        try:
            # Calculate max_num_seqs (2 * batch_size)
            max_num_seqs = 2 * batch_size
            self.logger.info(f"Batch size: {batch_size}, max_num_seqs: {max_num_seqs}")
            
            # Step 1: Start server
            self.logger.info("Step 1: Starting server")
            server_manager = self._start_server_for_batch(batch_size)
            if not server_manager:
                self.logger.error(f"Failed to start server for batch size {batch_size}")
                self._record_failed_result(batch_size, "Server startup failed")
                return False
            
            self.current_server = server_manager
            
            # Step 2: Run benchmark
            self.logger.info("Step 2: Running benchmark")
            benchmark_result = self._run_benchmark_for_batch(batch_size)
            if not benchmark_result:
                self.logger.error(f"Failed to run benchmark for batch size {batch_size}")
                self._record_failed_result(batch_size, "Benchmark execution failed")
                return False
            
            # Step 3: Record results
            self.logger.info("Step 3: Recording results")
            self.results.append(benchmark_result)
            
            if benchmark_result.success:
                self.logger.info(f"Batch size {batch_size} completed successfully")
                self.logger.info(f"Throughput: {benchmark_result.throughput_tok_per_sec} tok/sec")
                return True
            else:
                self.logger.warning(f"Batch size {batch_size} completed with errors: {benchmark_result.error_message}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in single batch benchmark for {batch_size}: {e}")
            self._record_failed_result(batch_size, f"Exception: {e}")
            return False
    
    def _start_server_for_batch(self, batch_size: int) -> Optional[ServerManager]:
        """
        Start server for the specified batch size.
        
        Args:
            batch_size: The batch size for this server instance
            
        Returns:
            ServerManager instance if successful, None if failed
        """
        try:
            # Create server manager
            server_manager = create_server_manager(batch_size)
            
            # Start server
            if not server_manager.start_server():
                self.logger.error(f"Failed to start server process for batch size {batch_size}")
                return None
            
            # Wait for readiness with timeout handling
            self.logger.info("Waiting for server to become ready...")
            if not server_manager.wait_for_readiness():
                self.logger.error(f"Server for batch size {batch_size} did not become ready")
                server_manager.cleanup_resources()
                return None
            
            self.logger.info(f"Server for batch size {batch_size} is ready")
            return server_manager
            
        except Exception as e:
            self.logger.error(f"Exception starting server for batch size {batch_size}: {e}")
            return None
    
    def _run_benchmark_for_batch(self, batch_size: int) -> Optional[BenchmarkResult]:
        """
        Run benchmark for the specified batch size.
        
        Args:
            batch_size: The batch size for this benchmark run
            
        Returns:
            BenchmarkResult if completed, None if failed to start
        """
        try:
            # Create benchmark manager
            benchmark_manager = create_benchmark_manager(batch_size)
            self.current_benchmark = benchmark_manager
            
            # Execute benchmark
            if not benchmark_manager.execute_benchmark():
                self.logger.error(f"Failed to start benchmark for batch size {batch_size}")
                return None
            
            # Monitor benchmark execution with timeout handling
            self.logger.info("Monitoring benchmark execution...")
            success = benchmark_manager.monitor_benchmark()
            
            # Collect results regardless of success/failure
            result = benchmark_manager.collect_benchmark_output()
            
            # Clean up benchmark resources
            benchmark_manager.cleanup_benchmark()
            self.current_benchmark = None
            
            if success:
                self.logger.info(f"Benchmark for batch size {batch_size} completed successfully")
            else:
                self.logger.error(f"Benchmark for batch size {batch_size} failed or timed out")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Exception running benchmark for batch size {batch_size}: {e}")
            return None
    
    def _find_maximum_batch_size(self, last_successful: int, first_failed: int) -> int:
        """
        Find the maximum working batch size by starting from last successful and working up.
        
        This is more efficient than working down from the failed size.
        
        Args:
            last_successful: The last batch size that worked
            first_failed: The first batch size that failed
            
        Returns:
            The maximum batch size that works
        """
        self.logger.info(f"Finding maximum batch size between {last_successful} and {first_failed}")
        
        # Generate batch sizes to test (increment by BATCH_SIZE_REDUCTION_STEP)
        test_batch_sizes = []
        current_size = last_successful + BATCH_SIZE_REDUCTION_STEP
        
        while current_size < first_failed:
            test_batch_sizes.append(current_size)
            current_size += BATCH_SIZE_REDUCTION_STEP
        
        if not test_batch_sizes:
            self.logger.info("No intermediate batch sizes to test")
            return last_successful
        
        self.logger.info(f"Will test intermediate batch sizes: {test_batch_sizes}")
        
        max_working_size = last_successful
        
        for test_size in test_batch_sizes:
            if self.shutdown_requested:
                break
                
            self.logger.info(f"Testing batch size: {test_size}")
            
            # Clean up any existing processes
            self._cleanup_current_processes()
            
            # Try the test batch size
            success = self._run_single_batch_benchmark(test_size)
            
            if success:
                max_working_size = test_size
                self.logger.info(f"Batch size {test_size} succeeded, continuing upward")
            else:
                self.logger.info(f"Batch size {test_size} failed, maximum found: {max_working_size}")
                break
        
        return max_working_size
        return False
    
    def _record_failed_result(self, batch_size: int, error_message: str) -> None:
        """
        Record a failed benchmark result.
        
        Args:
            batch_size: The batch size that failed
            error_message: Description of the failure
        """
        result = BenchmarkResult(
            batch_size=batch_size,
            max_num_seqs=2 * batch_size,
            success=False,
            error_message=error_message
        )
        self.results.append(result)
        self.logger.info(f"Recorded failed result for batch size {batch_size}: {error_message}")
    
    def _cleanup_current_processes(self) -> None:
        """
        Clean up current server and benchmark processes.
        
        Ensures proper cleanup between batch sizes to prevent resource conflicts.
        """
        self.logger.debug("Cleaning up current processes")
        
        try:
            # Clean up benchmark process
            if self.current_benchmark:
                self.current_benchmark.cleanup_benchmark()
                self.current_benchmark = None
            
            # Clean up server process
            if self.current_server:
                self.current_server.terminate_server()
                self.current_server.cleanup_resources()
                self.current_server = None
            
            # Brief pause to ensure processes are fully cleaned up
            time.sleep(2)
            
        except Exception as e:
            self.logger.error(f"Error during process cleanup: {e}")
    
    def _generate_consolidated_results(self) -> None:
        """
        Generate consolidated CSV results file.
        
        Uses the enhanced results processor for consistent CSV generation.
        """
        self.logger.info("Generating consolidated results CSV")
        
        try:
            from results_processor import ResultsProcessor
            
            # Create results processor
            processor = ResultsProcessor()
            
            # Generate both standard and detailed CSV reports
            standard_csv_path = processor.generate_csv_report(self.results)
            detailed_csv_path = processor.generate_detailed_csv_report(self.results)
            
            self.logger.info(f"Standard CSV report saved to: {standard_csv_path}")
            self.logger.info(f"Detailed CSV report saved to: {detailed_csv_path}")
            
            # Log summary
            successful_runs = sum(1 for r in self.results if r.success)
            total_runs = len(self.results)
            self.logger.info(f"Summary: {successful_runs}/{total_runs} benchmark runs succeeded")
            
        except Exception as e:
            self.logger.error(f"Error generating consolidated results: {e}")
    
    def _signal_handler(self, signum: int, frame) -> None:
        """
        Handle shutdown signals gracefully.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        signal_names = {
            signal.SIGINT: 'SIGINT',
            signal.SIGTERM: 'SIGTERM',
            signal.SIGHUP: 'SIGHUP'
        }
        signal_name = signal_names.get(signum, f'Signal {signum}')
        
        self.logger.info(f"Received {signal_name}, initiating graceful shutdown")
        self.shutdown_requested = True
        
        # Stop heartbeat thread
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_stop_event.set()
        
        # Clean up current processes
        self._cleanup_current_processes()
        
        # If this is SIGINT (Ctrl+C), exit immediately
        if signum == signal.SIGINT:
            self.logger.info("Immediate shutdown requested")
            sys.exit(1)
    
    def _sighup_handler(self, signum: int, frame) -> None:
        """
        Handle SIGHUP signal (SSH disconnection) by switching to background mode.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        self.logger.info("Received SIGHUP (SSH disconnection), switching to background mode")
        self.background_mode = True
        
        # Start heartbeat logging for background monitoring
        self._start_heartbeat_logging()
        
        # Redirect stdout/stderr to log files to prevent broken pipe errors
        self._redirect_output_to_logs()
    
    def get_results(self) -> List[BenchmarkResult]:
        """
        Get all benchmark results.
        
        Returns:
            List of BenchmarkResult objects
        """
        return self.results.copy()
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current progress.
        
        Returns:
            Dictionary with progress information
        """
        total_batches = len(BATCH_SIZES)
        completed_batches = len(self.results)
        successful_batches = sum(1 for r in self.results if r.success)
        
        return {
            'total_batches': total_batches,
            'completed_batches': completed_batches,
            'successful_batches': successful_batches,
            'remaining_batches': total_batches - completed_batches,
            'success_rate': successful_batches / completed_batches if completed_batches > 0 else 0
        }
    
    def _start_heartbeat_logging(self) -> None:
        """
        Start heartbeat logging for background monitoring.
        
        This creates a separate thread that periodically logs progress
        to help monitor the process when running in background mode.
        """
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            return  # Already running
        
        self.logger.info("Starting heartbeat logging for background monitoring")
        self.heartbeat_stop_event.clear()
        
        def heartbeat_worker():
            """Worker function for heartbeat logging."""
            heartbeat_logger = logging.getLogger('neuronx_benchmark.heartbeat')
            
            while not self.heartbeat_stop_event.wait(300):  # Log every 5 minutes
                try:
                    summary = self.get_progress_summary()
                    heartbeat_logger.info(
                        f"HEARTBEAT: {summary['completed_batches']}/{summary['total_batches']} "
                        f"batches completed, {summary['successful_batches']} successful, "
                        f"background_mode={self.background_mode}"
                    )
                    
                    # Log current activity
                    if self.current_server:
                        heartbeat_logger.info("HEARTBEAT: Server process active")
                    if self.current_benchmark:
                        heartbeat_logger.info("HEARTBEAT: Benchmark process active")
                        
                except Exception as e:
                    heartbeat_logger.error(f"Error in heartbeat logging: {e}")
        
        self.heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
        self.heartbeat_thread.start()
    
    def _redirect_output_to_logs(self) -> None:
        """
        Redirect stdout and stderr to log files to prevent broken pipe errors
        when SSH session is disconnected.
        """
        try:
            # Create log files for stdout/stderr redirection
            stdout_log = open('logs/background_stdout.log', 'a')
            stderr_log = open('logs/background_stderr.log', 'a')
            
            # Redirect stdout and stderr
            sys.stdout = stdout_log
            sys.stderr = stderr_log
            
            self.logger.info("Redirected stdout/stderr to log files for background mode")
            
        except Exception as e:
            self.logger.error(f"Failed to redirect output to logs: {e}")
    
    def _cleanup_on_exit(self) -> None:
        """
        Cleanup function called on process exit.
        
        This ensures resources are cleaned up even if the process
        exits unexpectedly.
        """
        try:
            # Stop heartbeat thread
            if self.heartbeat_thread and self.heartbeat_thread.is_alive():
                self.heartbeat_stop_event.set()
                self.heartbeat_thread.join(timeout=5)
            
            # Clean up current processes
            self._cleanup_current_processes()
            
            # Log final status
            if hasattr(self, 'logger'):
                self.logger.info("Process cleanup completed on exit")
                
        except Exception:
            # Avoid exceptions during cleanup
            pass
    
    def enable_background_mode(self) -> None:
        """
        Enable background mode for SSH disconnection resilience.
        
        This method prepares the orchestrator to run in background mode,
        setting up heartbeat logging and output redirection.
        """
        self.logger.info("Enabling background mode")
        self.background_mode = True
        
        # Start heartbeat logging
        self._start_heartbeat_logging()
        
        # Redirect output to prevent broken pipe errors
        self._redirect_output_to_logs()
        
        self.logger.info("Background mode enabled - process will continue if SSH disconnects")


def setup_logging(background_mode: bool = False) -> None:
    """
    Set up comprehensive logging system with separate log files for different components.
    
    Creates separate log files for:
    - Main orchestrator logs
    - Server process logs
    - Benchmark process logs
    - Error logs
    - Heartbeat logs (for background monitoring)
    
    Args:
        background_mode: If True, reduces console output for background execution
    """
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    heartbeat_formatter = logging.Formatter(
        '%(asctime)s - HEARTBEAT - %(message)s'
    )
    
    # Main orchestrator log file
    main_handler = logging.FileHandler('logs/orchestrator.log')
    main_handler.setLevel(logging.DEBUG)
    main_handler.setFormatter(detailed_formatter)
    main_handler.addFilter(lambda record: 'orchestrator' in record.name)
    root_logger.addHandler(main_handler)
    
    # Server process logs
    server_handler = logging.FileHandler('logs/server_processes.log')
    server_handler.setLevel(logging.DEBUG)
    server_handler.setFormatter(detailed_formatter)
    server_handler.addFilter(lambda record: 'server' in record.name)
    root_logger.addHandler(server_handler)
    
    # Benchmark process logs
    benchmark_handler = logging.FileHandler('logs/benchmark_processes.log')
    benchmark_handler.setLevel(logging.DEBUG)
    benchmark_handler.setFormatter(detailed_formatter)
    benchmark_handler.addFilter(lambda record: 'benchmark' in record.name)
    root_logger.addHandler(benchmark_handler)
    
    # Results processing logs
    results_handler = logging.FileHandler('logs/results_processing.log')
    results_handler.setLevel(logging.DEBUG)
    results_handler.setFormatter(detailed_formatter)
    results_handler.addFilter(lambda record: 'results' in record.name)
    root_logger.addHandler(results_handler)
    
    # Heartbeat logs for background monitoring
    heartbeat_handler = logging.FileHandler('logs/heartbeat.log')
    heartbeat_handler.setLevel(logging.INFO)
    heartbeat_handler.setFormatter(heartbeat_formatter)
    heartbeat_handler.addFilter(lambda record: 'heartbeat' in record.name)
    root_logger.addHandler(heartbeat_handler)
    
    # General application log (all messages)
    general_handler = logging.FileHandler('logs/benchmark_automation.log')
    general_handler.setLevel(logging.DEBUG)
    general_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(general_handler)
    
    # Console handler for important messages (reduced in background mode)
    if not background_mode:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
    else:
        # In background mode, only show critical messages on console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.CRITICAL)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
    
    # Separate handler for errors
    error_handler = logging.FileHandler('logs/benchmark_errors.log')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # Create a summary log for easy monitoring
    summary_handler = logging.FileHandler('logs/summary.log')
    summary_handler.setLevel(logging.INFO)
    summary_handler.setFormatter(simple_formatter)
    summary_handler.addFilter(lambda record: any(keyword in record.getMessage().lower() 
                                                for keyword in ['completed', 'failed', 'starting', 'summary', 'heartbeat']))
    root_logger.addHandler(summary_handler)


def main(background_mode: bool = False) -> int:
    """
    Main entry point for the benchmark orchestrator.
    
    Args:
        background_mode: If True, enables background execution mode
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Set up logging with background mode consideration
        setup_logging(background_mode)
        
        logger = logging.getLogger('neuronx_benchmark.main')
        logger.info("Starting neuronx benchmarking automation")
        
        if background_mode:
            logger.info("Running in background mode - SSH disconnection resilient")
        
        # Create and run orchestrator
        orchestrator = BenchmarkOrchestrator()
        
        # Enable background mode if requested
        if background_mode:
            orchestrator.enable_background_mode()
        
        success = orchestrator.run_complete_benchmark_suite()
        
        if success:
            logger.info("Benchmark automation completed successfully")
            
            # Print final summary
            summary = orchestrator.get_progress_summary()
            logger.info(f"Final summary: {summary['successful_batches']}/{summary['total_batches']} batches succeeded")
            
            return 0
        else:
            logger.error("Benchmark automation failed")
            return 1
            
    except KeyboardInterrupt:
        logger = logging.getLogger('neuronx_benchmark.main')
        logger.info("Benchmark automation interrupted by user")
        return 1
    except Exception as e:
        logger = logging.getLogger('neuronx_benchmark.main')
        logger.error(f"Critical error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())