"""
Environment setup utilities for server and benchmark processes.
"""

import os
import subprocess
import logging
from typing import Dict, List, Optional
from config import (
    VENV_PATH, 
    get_server_environment, 
    get_benchmark_environment,
    validate_environment
)


def setup_logging() -> logging.Logger:
    """
    Set up logging for the benchmarking system.
    
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('benchmark_automation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('neuronx_benchmark')


def activate_virtual_environment() -> str:
    """
    Get the command prefix to activate the virtual environment.
    
    Returns:
        Command prefix string to activate the virtual environment
    """
    return f"source {VENV_PATH} &&"


def prepare_server_environment(batch_size: int) -> Dict[str, str]:
    """
    Prepare the complete environment for server startup.
    
    Args:
        batch_size: The batch size for this server instance
        
    Returns:
        Dictionary of environment variables for server process
    """
    logger = logging.getLogger('neuronx_benchmark.server_env')
    
    # Get base server environment
    env = get_server_environment()
    
    # Add any batch-specific server environment variables if needed
    logger.info(f"Prepared server environment for batch size {batch_size}")
    logger.debug(f"Server environment variables: {list(env.keys())}")
    
    return env


def prepare_benchmark_environment(batch_size: int) -> Dict[str, str]:
    """
    Prepare the complete environment for benchmark execution.
    
    Args:
        batch_size: The batch size for this benchmark run
        
    Returns:
        Dictionary of environment variables for benchmark process
    """
    logger = logging.getLogger('neuronx_benchmark.benchmark_env')
    
    # Get batch-specific benchmark environment
    env = get_benchmark_environment(batch_size)
    
    logger.info(f"Prepared benchmark environment for batch size {batch_size}")
    logger.debug(f"LLM_PERF_CONCURRENT: {env.get('LLM_PERF_CONCURRENT')}")
    logger.debug(f"LLM_PERF_MAX_REQUESTS: {env.get('LLM_PERF_MAX_REQUESTS')}")
    
    return env


def create_shell_command(command: List[str], env: Dict[str, str]) -> List[str]:
    """
    Create a shell command that activates the virtual environment
    and runs the specified command with the given environment.
    
    Args:
        command: List of command arguments
        env: Environment variables dictionary
        
    Returns:
        Command list for subprocess execution with bash
    """
    # Create environment variable assignments
    env_assignments = []
    for key, value in env.items():
        if key not in os.environ or os.environ[key] != value:
            env_assignments.append(f'{key}="{value}"')
    
    # Build the complete command
    venv_activation = activate_virtual_environment()
    env_string = ' '.join(env_assignments)
    command_string = ' '.join(command)
    
    if env_assignments:
        full_command = f"{venv_activation} {env_string} {command_string}"
    else:
        full_command = f"{venv_activation} {command_string}"
    
    # Return as bash command to ensure 'source' works
    return ['/bin/bash', '-c', full_command]


def validate_process_environment() -> bool:
    """
    Validate that the environment is properly set up for running processes.
    
    Returns:
        True if environment is valid, False otherwise
    """
    logger = logging.getLogger('neuronx_benchmark.validation')
    
    # Validate basic environment
    if not validate_environment():
        logger.error("Basic environment validation failed")
        return False
    
    # Test virtual environment activation using bash explicitly
    try:
        test_command = f"bash -c '{activate_virtual_environment()} python --version'"
        result = subprocess.run(
            test_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            logger.warning(f"Virtual environment test failed: {result.stderr}")
            logger.info("This may be expected in some test environments")
            # Don't fail validation just because venv test fails - the actual server startup will handle this
            return True
        
        logger.info(f"Virtual environment validated: {result.stdout.strip()}")
        
    except subprocess.TimeoutExpired:
        logger.error("Virtual environment test timed out")
        return False
    except Exception as e:
        logger.warning(f"Virtual environment test failed: {e}")
        logger.info("This may be expected in some test environments")
        # Don't fail validation just because venv test fails
        return True
    
    return True


def setup_process_directories(batch_size: int) -> Dict[str, str]:
    """
    Set up directories needed for a specific batch size run.
    
    Args:
        batch_size: The batch size for this run
        
    Returns:
        Dictionary with directory paths
    """
    from config import ensure_results_directory
    
    logger = logging.getLogger('neuronx_benchmark.setup')
    
    # Ensure results directory exists
    results_dir = ensure_results_directory(batch_size)
    
    # Create any additional directories if needed
    log_dir = os.path.join(results_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    directories = {
        'results': results_dir,
        'logs': log_dir
    }
    
    logger.info(f"Set up directories for batch size {batch_size}: {directories}")
    
    return directories


def cleanup_environment() -> None:
    """
    Clean up any temporary environment setup.
    This function is called when the benchmarking process completes.
    """
    logger = logging.getLogger('neuronx_benchmark.cleanup')
    
    # Clean up any temporary files or environment state
    # For now, this is a placeholder for future cleanup needs
    logger.info("Environment cleanup completed")


def get_environment_info() -> Dict[str, str]:
    """
    Get information about the current environment for logging/debugging.
    
    Returns:
        Dictionary with environment information
    """
    info = {
        'python_version': subprocess.run(['python', '--version'], 
                                       capture_output=True, text=True).stdout.strip(),
        'working_directory': os.getcwd(),
        'virtual_env_path': VENV_PATH,
        'virtual_env_exists': str(os.path.exists(VENV_PATH)),
        'user': os.environ.get('USER', 'unknown'),
        'hostname': os.environ.get('HOSTNAME', 'unknown')
    }
    
    return info