"""
Server process management for neuronx benchmarking automation.

This module handles the lifecycle of vLLM server processes including:
- Server startup with proper environment configuration
- Health checks and readiness detection
- Timeout handling and process cleanup
- Server termination and resource cleanup
"""

import os
import time
import socket
import signal
import subprocess
import logging
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
    ProcessStatus,
    SERVER_STARTUP_TIMEOUT,
    HEALTH_CHECK_INTERVAL,
    SERVER_PORT,
    get_server_command
)
from environment_utils import (
    prepare_server_environment,
    create_shell_command,
    setup_process_directories
)


class ServerManager:
    """
    Manages the lifecycle of a single vLLM server process.
    
    This class handles server startup, health monitoring, and cleanup
    for one server instance at a time.
    """
    
    def __init__(self, config: BenchmarkConfig):
        """
        Initialize the server manager.
        
        Args:
            config: Benchmark configuration for this server instance
        """
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.status = ProcessStatus()
        self.logger = logging.getLogger(f'neuronx_benchmark.server.batch_{config.batch_size}')
        self.directories: Dict[str, str] = {}
        
    def start_server(self) -> bool:
        """
        Start the vLLM server process with proper environment configuration.
        
        Returns:
            True if server started successfully, False otherwise
        """
        self.logger.info(f"Starting server for batch size {self.config.batch_size}")
        
        try:
            # Set up directories for this batch size
            self.directories = setup_process_directories(self.config.batch_size)
            
            # Prepare environment variables
            env = prepare_server_environment(self.config.batch_size)
            
            # Get server command
            command = get_server_command(self.config)
            
            # Create shell command with virtual environment activation
            shell_command = create_shell_command(command, env)
            
            self.logger.info(f"Server command: {' '.join(command)}")
            self.logger.debug(f"Full shell command: {' '.join(shell_command)}")
            
            # Set up log files
            server_log_path = os.path.join(self.directories['logs'], 'server.log')
            server_error_path = os.path.join(self.directories['logs'], 'server_error.log')
            
            # Log the exact command for replication
            with open(server_log_path, 'w') as f:
                f.write("=== SERVER STARTUP COMMAND FOR REPLICATION ===\n")
                f.write(f"# Batch size: {self.config.batch_size}, max_num_seqs: {self.config.max_num_seqs}\n")
                f.write(f"# Full command to replicate this server:\n")
                f.write(f"{' '.join(shell_command)}\n")
                f.write("=" * 50 + "\n\n")
            
            # Start the server process
            with open(server_log_path, 'a') as stdout_file, \
                 open(server_error_path, 'w') as stderr_file:
                
                self.process = subprocess.Popen(
                    shell_command,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    env=env,
                    preexec_fn=os.setsid  # Create new process group for easier cleanup
                )
            
            # Update status
            self.status.pid = self.process.pid
            self.status.status = 'starting'
            self.status.start_time = datetime.now()
            
            self.logger.info(f"Server process started with PID {self.process.pid}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            self.status.status = 'failed'
            self.status.error_message = str(e)
            return False
    
    def wait_for_readiness(self) -> bool:
        """
        Wait for the server to become ready and accept connections.
        
        Implements health check by attempting to connect to the server port
        and checking for successful responses.
        
        Returns:
            True if server becomes ready within timeout, False otherwise
        """
        if not self.process:
            self.logger.error("No server process to check readiness")
            return False
        
        self.logger.info(f"Waiting for server readiness on port {self.config.port}")
        
        start_time = datetime.now()
        timeout = timedelta(seconds=SERVER_STARTUP_TIMEOUT)
        
        while datetime.now() - start_time < timeout:
            # Check if process is still running
            if self.process.poll() is not None:
                self.logger.error(f"Server process exited with code {self.process.returncode}")
                self.status.status = 'failed'
                self.status.error_message = f"Process exited with code {self.process.returncode}"
                return False
            
            # Check port availability
            if self._check_port_ready():
                # Additional health check - try to make a simple HTTP request
                if self._check_server_health():
                    self.status.status = 'ready'
                    elapsed = datetime.now() - start_time
                    self.logger.info(f"Server ready after {elapsed.total_seconds():.1f} seconds")
                    return True
            
            # Wait before next check
            time.sleep(HEALTH_CHECK_INTERVAL)
            
            # Log progress every 30 seconds
            elapsed = datetime.now() - start_time
            if elapsed.total_seconds() % 30 < HEALTH_CHECK_INTERVAL:
                self.logger.info(f"Still waiting for server readiness... ({elapsed.total_seconds():.0f}s elapsed)")
        
        # Timeout reached
        self.logger.error(f"Server readiness timeout after {SERVER_STARTUP_TIMEOUT} seconds")
        self.status.status = 'failed'
        self.status.error_message = "Readiness timeout"
        return False
    
    def _check_port_ready(self) -> bool:
        """
        Check if the server port is accepting connections.
        
        Returns:
            True if port is ready, False otherwise
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(2)  # 2 second timeout for connection attempt
                result = sock.connect_ex(('localhost', self.config.port))
                return result == 0
        except Exception:
            return False
    
    def _check_server_health(self) -> bool:
        """
        Perform a basic health check on the server.
        
        This makes a simple HTTP request to verify the server is responding
        correctly to API requests.
        
        Returns:
            True if server responds to health check, False otherwise
        """
        try:
            import requests
            
            # Try to get the models endpoint (basic health check)
            url = f"http://localhost:{self.config.port}/v1/models"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                self.logger.debug("Server health check passed")
                return True
            else:
                self.logger.debug(f"Server health check failed with status {response.status_code}")
                return False
                
        except ImportError:
            # If requests is not available, just rely on port check
            self.logger.debug("Requests library not available, using port check only")
            return True
        except Exception as e:
            self.logger.debug(f"Server health check failed: {e}")
            return False
    
    def terminate_server(self) -> bool:
        """
        Terminate the server process gracefully.
        
        Attempts graceful shutdown first, then forces termination if needed.
        
        Returns:
            True if server was terminated successfully, False otherwise
        """
        if not self.process:
            self.logger.info("No server process to terminate")
            return True
        
        self.logger.info(f"Terminating server process {self.process.pid}")
        
        try:
            # First, try graceful termination
            if self.process.poll() is None:  # Process is still running
                self.logger.info("Attempting graceful shutdown...")
                
                # Send SIGTERM to the process group
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    # Process might have already exited
                    pass
                
                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=10)  # Wait up to 10 seconds
                    self.logger.info("Server terminated gracefully")
                except subprocess.TimeoutExpired:
                    self.logger.warning("Graceful shutdown timed out, forcing termination")
                    
                    # Force termination
                    try:
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                        self.process.wait(timeout=5)
                        self.logger.info("Server terminated forcefully")
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        self.logger.error("Failed to force terminate server")
                        return False
            
            self.status.status = 'completed'
            return True
            
        except Exception as e:
            self.logger.error(f"Error terminating server: {e}")
            self.status.status = 'failed'
            self.status.error_message = f"Termination error: {e}"
            return False
    
    def cleanup_resources(self) -> None:
        """
        Clean up resources associated with the server process.
        
        This includes cleaning up temporary files, closing file handles,
        and ensuring no zombie processes remain.
        """
        self.logger.info("Cleaning up server resources")
        
        try:
            # Ensure process is terminated
            if self.process and self.process.poll() is None:
                self.terminate_server()
            
            # Clean up any child processes that might still be running
            self._cleanup_child_processes()
            
            # Close process handles
            if self.process:
                try:
                    self.process.stdout.close() if self.process.stdout else None
                    self.process.stderr.close() if self.process.stderr else None
                except Exception:
                    pass
            
            self.logger.info("Server resource cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during resource cleanup: {e}")
    
    def _cleanup_child_processes(self) -> None:
        """
        Clean up any child processes that might still be running.
        
        This ensures no orphaned processes remain after server termination.
        """
        if not self.process or not self.process.pid:
            return
        
        if not HAS_PSUTIL:
            self.logger.debug("psutil not available, skipping advanced child process cleanup")
            return
        
        try:
            # Find and terminate any child processes
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
            # Parent process already gone
            pass
        except Exception as e:
            self.logger.warning(f"Error cleaning up child processes: {e}")
    
    def get_status(self) -> ProcessStatus:
        """
        Get the current status of the server process.
        
        Returns:
            Current process status
        """
        # Update status based on process state
        if self.process:
            poll_result = self.process.poll()
            if poll_result is not None and self.status.status not in ['completed', 'failed']:
                self.status.status = 'failed'
                self.status.error_message = f"Process exited with code {poll_result}"
        
        return self.status
    
    def get_log_paths(self) -> Dict[str, str]:
        """
        Get paths to server log files.
        
        Returns:
            Dictionary with log file paths
        """
        if not self.directories:
            return {}
        
        return {
            'stdout': os.path.join(self.directories['logs'], 'server.log'),
            'stderr': os.path.join(self.directories['logs'], 'server_error.log')
        }
    
    def is_running(self) -> bool:
        """
        Check if the server process is currently running.
        
        Returns:
            True if server is running, False otherwise
        """
        if not self.process:
            return False
        
        return self.process.poll() is None
    
    def get_runtime(self) -> Optional[timedelta]:
        """
        Get the runtime of the server process.
        
        Returns:
            Runtime as timedelta, or None if not started
        """
        if not self.status.start_time:
            return None
        
        return datetime.now() - self.status.start_time


def create_server_manager(batch_size: int) -> ServerManager:
    """
    Create a server manager for the specified batch size.
    
    Args:
        batch_size: The batch size for this server instance
        
    Returns:
        Configured ServerManager instance
    """
    config = BenchmarkConfig.from_batch_size(batch_size)
    return ServerManager(config)


def start_server_for_batch(batch_size: int) -> Optional[ServerManager]:
    """
    Start a server for the specified batch size and wait for readiness.
    
    This is a convenience function that creates a server manager,
    starts the server, and waits for it to become ready.
    
    Args:
        batch_size: The batch size for this server instance
        
    Returns:
        ServerManager instance if successful, None if failed
    """
    logger = logging.getLogger('neuronx_benchmark.server_startup')
    
    try:
        # Create server manager
        server_manager = create_server_manager(batch_size)
        
        # Start server
        if not server_manager.start_server():
            logger.error(f"Failed to start server for batch size {batch_size}")
            return None
        
        # Wait for readiness
        if not server_manager.wait_for_readiness():
            logger.error(f"Server for batch size {batch_size} did not become ready")
            server_manager.cleanup_resources()
            return None
        
        logger.info(f"Server for batch size {batch_size} is ready")
        return server_manager
        
    except Exception as e:
        logger.error(f"Error starting server for batch size {batch_size}: {e}")
        return None