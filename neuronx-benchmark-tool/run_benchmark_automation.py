#!/usr/bin/env python3
"""
Main entry point for running the neuronx benchmarking automation.

This script provides a simple command-line interface to start the
complete benchmark automation process.
"""

import sys
import os
import argparse
import logging
from benchmark_orchestrator import BenchmarkOrchestrator, setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run neuronx benchmarking automation across multiple batch sizes'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set the logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform a dry run without actually starting processes'
    )
    
    parser.add_argument(
        '--background',
        action='store_true',
        help='Run in background mode with SSH disconnection resilience'
    )
    
    parser.add_argument(
        '--nohup',
        action='store_true',
        help='Run with nohup for complete SSH disconnection resilience'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Handle nohup mode by re-executing with nohup
    if args.nohup and not os.environ.get('NOHUP_ACTIVE'):
        return run_with_nohup(args)
    
    # Determine background mode
    background_mode = args.background or args.nohup or os.environ.get('NOHUP_ACTIVE')
    
    # Set up logging
    setup_logging(background_mode)
    
    # Adjust log level if specified
    if args.log_level != 'INFO':
        logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger = logging.getLogger('neuronx_benchmark.main')
    
    try:
        logger.info("Starting neuronx benchmarking automation")
        
        if background_mode:
            logger.info("Running in background mode - SSH disconnection resilient")
        
        if args.dry_run:
            logger.info("DRY RUN MODE - No processes will be started")
            # TODO: Implement dry run logic
            logger.info("Dry run completed")
            return 0
        
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
            if not background_mode:
                print(f"\n=== FINAL SUMMARY ===")
                print(f"Total batches: {summary['total_batches']}")
                print(f"Completed batches: {summary['completed_batches']}")
                print(f"Successful batches: {summary['successful_batches']}")
                print(f"Success rate: {summary['success_rate']:.1%}")
            
            logger.info(f"Final summary: {summary['successful_batches']}/{summary['total_batches']} batches succeeded")
            
            return 0
        else:
            logger.error("Benchmark automation failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Benchmark automation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Critical error: {e}")
        return 1


def run_with_nohup(args):
    """
    Re-execute the script with nohup for complete SSH disconnection resilience.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Exit code from nohup execution
    """
    import subprocess
    
    # Build command arguments without --nohup flag
    cmd_args = [sys.executable, __file__]
    
    if args.log_level != 'INFO':
        cmd_args.extend(['--log-level', args.log_level])
    
    if args.dry_run:
        cmd_args.append('--dry-run')
    
    if args.background:
        cmd_args.append('--background')
    
    # Set environment variable to indicate nohup is active
    env = os.environ.copy()
    env['NOHUP_ACTIVE'] = '1'
    
    # Create nohup command
    nohup_cmd = ['nohup'] + cmd_args
    
    print("Starting benchmark automation with nohup...")
    print(f"Command: {' '.join(nohup_cmd)}")
    print("Output will be written to nohup.out and log files in logs/ directory")
    print("You can safely disconnect from SSH - the process will continue running")
    
    try:
        # Execute with nohup
        result = subprocess.run(nohup_cmd, env=env)
        return result.returncode
        
    except Exception as e:
        print(f"Failed to start with nohup: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())