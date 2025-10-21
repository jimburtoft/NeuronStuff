"""
Results processing and CSV generation for neuronx benchmarking automation.

This module handles:
- Parsing token_benchmark_ray.py output files
- Extracting required metrics: throughput, inter-token-latency, TTFT, end-to-end latency
- CSV generation with proper column headers and timestamped filename
- Handling failed benchmarks with placeholder values
"""

import os
import json
import csv
import glob
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

from config import BenchmarkResult, get_csv_filename, RESULTS_BASE_DIR


class ResultsProcessor:
    """
    Processes benchmark results and generates consolidated CSV output.
    
    This class handles parsing of token_benchmark_ray.py output files,
    extracting performance metrics, and generating CSV reports.
    """
    
    def __init__(self, results_base_dir: str = RESULTS_BASE_DIR):
        """
        Initialize the results processor.
        
        Args:
            results_base_dir: Base directory containing batch result folders
        """
        self.results_base_dir = results_base_dir
        self.logger = logging.getLogger('neuronx_benchmark.results_processor')
    
    def parse_result_file(self, file_path: str) -> Optional[Dict[str, float]]:
        """
        Parse a token_benchmark_ray.py result file to extract performance metrics.
        
        Args:
            file_path: Path to the result file (JSON or log format)
            
        Returns:
            Dictionary with extracted metrics, or None if parsing failed
        """
        try:
            self.logger.debug(f"Parsing result file: {file_path}")
            
            # Check if this is a log file (benchmark.log)
            if file_path.endswith('benchmark.log'):
                return self._parse_benchmark_log(file_path)
            
            # Try parsing as JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract metrics from the JSON structure
            metrics = {}
            
            # The token_benchmark_ray.py output has a 'results' section with metrics
            results_data = data.get('results', data)  # Fallback to root if no 'results' key
            
            # Extract throughput (tok/sec) - this is the overall output throughput
            if 'mean_output_throughput_token_per_s' in results_data:
                metrics['throughput_tok_per_sec'] = float(results_data['mean_output_throughput_token_per_s'])
            
            # Extract mean inter-token latency (s)
            inter_token_data = results_data.get('inter_token_latency_s', {})
            if isinstance(inter_token_data, dict) and 'mean' in inter_token_data:
                metrics['mean_inter_token_latency_s'] = float(inter_token_data['mean'])
            elif isinstance(inter_token_data, (int, float)):
                metrics['mean_inter_token_latency_s'] = float(inter_token_data)
            
            # Extract mean TTFT (s)
            ttft_data = results_data.get('ttft_s', {})
            if isinstance(ttft_data, dict) and 'mean' in ttft_data:
                metrics['mean_ttft_s'] = float(ttft_data['mean'])
            elif isinstance(ttft_data, (int, float)):
                metrics['mean_ttft_s'] = float(ttft_data)
            
            # Extract mean end-to-end latency (s)
            e2e_data = results_data.get('end_to_end_latency_s', {})
            if isinstance(e2e_data, dict) and 'mean' in e2e_data:
                metrics['mean_end_to_end_latency_s'] = float(e2e_data['mean'])
            elif isinstance(e2e_data, (int, float)):
                metrics['mean_end_to_end_latency_s'] = float(e2e_data)
            
            # Log extracted metrics
            if metrics:
                self.logger.info(f"Extracted metrics from {file_path}:")
                for key, value in metrics.items():
                    self.logger.info(f"  {key}: {value}")
            else:
                self.logger.warning(f"No metrics extracted from {file_path}")
            
            return metrics if metrics else None
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in {file_path}: {e}")
            # If JSON parsing fails, try parsing as log file
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
            
            # Log extracted metrics
            if metrics:
                self.logger.info(f"Extracted metrics from log {log_path}:")
                for key, value in metrics.items():
                    self.logger.info(f"  {key}: {value}")
            else:
                self.logger.warning(f"No metrics extracted from log {log_path}")
            
            return metrics if metrics else None
            
        except Exception as e:
            self.logger.error(f"Error parsing benchmark log {log_path}: {e}")
            return None
    
    def find_result_files_for_batch(self, batch_size: int) -> List[str]:
        """
        Find result files for a specific batch size.
        
        Args:
            batch_size: The batch size to find results for
            
        Returns:
            List of paths to result files for this batch size
        """
        batch_dir = os.path.join(self.results_base_dir, f"batch_{batch_size}")
        
        if not os.path.exists(batch_dir):
            self.logger.debug(f"Batch directory does not exist: {batch_dir}")
            return []
        
        # Look for JSON result files with common patterns
        patterns = [
            os.path.join(batch_dir, "*_summary.json"),
            os.path.join(batch_dir, "*.json"),
            os.path.join(batch_dir, "results_*.json"),
            os.path.join(batch_dir, "*_results.json"),
        ]
        
        result_files = []
        for pattern in patterns:
            files = glob.glob(pattern)
            result_files.extend(files)
        
        # If no JSON files found, check for benchmark log file
        if not result_files:
            log_file = os.path.join(batch_dir, 'logs', 'benchmark.log')
            if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
                result_files.append(log_file)
        
        # Remove duplicates and sort by modification time (newest first)
        result_files = list(set(result_files))
        result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        self.logger.debug(f"Found {len(result_files)} result files for batch {batch_size}")
        return result_files
    
    def process_batch_results(self, batch_size: int) -> BenchmarkResult:
        """
        Process results for a specific batch size.
        
        Args:
            batch_size: The batch size to process results for
            
        Returns:
            BenchmarkResult with extracted metrics or failure information
        """
        self.logger.info(f"Processing results for batch size {batch_size}")
        
        result = BenchmarkResult(
            batch_size=batch_size,
            max_num_seqs=2 * batch_size
        )
        
        try:
            # Find result files for this batch
            result_files = self.find_result_files_for_batch(batch_size)
            
            if not result_files:
                self.logger.warning(f"No result files found for batch size {batch_size}")
                result.error_message = "No result files found"
                return result
            
            # Try to parse the most recent result file
            for result_file in result_files:
                self.logger.debug(f"Attempting to parse: {result_file}")
                metrics = self.parse_result_file(result_file)
                
                if metrics:
                    # Successfully extracted metrics
                    result.throughput_tok_per_sec = metrics.get('throughput_tok_per_sec')
                    result.mean_inter_token_latency_s = metrics.get('mean_inter_token_latency_s')
                    result.mean_ttft_s = metrics.get('mean_ttft_s')
                    result.mean_end_to_end_latency_s = metrics.get('mean_end_to_end_latency_s')
                    result.success = True
                    
                    self.logger.info(f"Successfully processed results for batch size {batch_size}")
                    self.logger.info(f"  Throughput: {result.throughput_tok_per_sec} tok/sec")
                    return result
            
            # If we get here, no files could be parsed successfully
            result.error_message = "Failed to parse any result files"
            self.logger.error(f"Failed to parse any result files for batch size {batch_size}")
            
        except Exception as e:
            self.logger.error(f"Error processing results for batch size {batch_size}: {e}")
            result.error_message = f"Processing error: {e}"
        
        return result
    
    def process_all_batch_results(self, batch_sizes: List[int]) -> List[BenchmarkResult]:
        """
        Process results for all specified batch sizes.
        
        Args:
            batch_sizes: List of batch sizes to process
            
        Returns:
            List of BenchmarkResult objects
        """
        self.logger.info(f"Processing results for {len(batch_sizes)} batch sizes")
        
        results = []
        for batch_size in batch_sizes:
            result = self.process_batch_results(batch_size)
            results.append(result)
        
        successful_results = sum(1 for r in results if r.success)
        self.logger.info(f"Successfully processed {successful_results}/{len(batch_sizes)} batch results")
        
        return results
    
    def generate_csv_report(self, results: List[BenchmarkResult], output_path: Optional[str] = None) -> str:
        """
        Generate a CSV report from benchmark results.
        
        Args:
            results: List of BenchmarkResult objects
            output_path: Optional custom output path. If None, uses timestamped filename
            
        Returns:
            Path to the generated CSV file
        """
        if output_path is None:
            csv_filename = get_csv_filename()
            output_path = os.path.join(self.results_base_dir, csv_filename)
        
        self.logger.info(f"Generating CSV report: {output_path}")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create directory if path has a directory component
            os.makedirs(output_dir, exist_ok=True)
        
        # Define CSV columns as specified in requirements
        fieldnames = [
            'bb',  # batch_size
            'max_num_seqs',
            'throughput (tok/sec)',
            'mean inter-token-latency (s)',
            'Mean TTFT (s)',
            'Mean End to end latency (s)'
        ]
        
        try:
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    # Handle failed benchmarks with placeholder values
                    row = {
                        'bb': result.batch_size,
                        'max_num_seqs': result.max_num_seqs,
                        'throughput (tok/sec)': result.throughput_tok_per_sec if result.success else 'N/A',
                        'mean inter-token-latency (s)': result.mean_inter_token_latency_s if result.success else 'N/A',
                        'Mean TTFT (s)': result.mean_ttft_s if result.success else 'N/A',
                        'Mean End to end latency (s)': result.mean_end_to_end_latency_s if result.success else 'N/A'
                    }
                    writer.writerow(row)
            
            self.logger.info(f"CSV report generated successfully: {output_path}")
            
            # Log summary statistics
            successful_runs = sum(1 for r in results if r.success)
            total_runs = len(results)
            self.logger.info(f"CSV contains {successful_runs}/{total_runs} successful benchmark runs")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating CSV report: {e}")
            raise
    
    def generate_detailed_csv_report(self, results: List[BenchmarkResult], output_path: Optional[str] = None) -> str:
        """
        Generate a detailed CSV report including error information.
        
        Args:
            results: List of BenchmarkResult objects
            output_path: Optional custom output path. If None, uses timestamped filename
            
        Returns:
            Path to the generated CSV file
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            csv_filename = f"detailed_results_{timestamp}.csv"
            output_path = os.path.join(self.results_base_dir, csv_filename)
        
        self.logger.info(f"Generating detailed CSV report: {output_path}")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create directory if path has a directory component
            os.makedirs(output_dir, exist_ok=True)
        
        # Define detailed CSV columns
        fieldnames = [
            'bb',  # batch_size
            'max_num_seqs',
            'throughput (tok/sec)',
            'mean inter-token-latency (s)',
            'Mean TTFT (s)',
            'Mean End to end latency (s)',
            'success',
            'error_message'
        ]
        
        try:
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    row = {
                        'bb': result.batch_size,
                        'max_num_seqs': result.max_num_seqs,
                        'throughput (tok/sec)': result.throughput_tok_per_sec or 'N/A',
                        'mean inter-token-latency (s)': result.mean_inter_token_latency_s or 'N/A',
                        'Mean TTFT (s)': result.mean_ttft_s or 'N/A',
                        'Mean End to end latency (s)': result.mean_end_to_end_latency_s or 'N/A',
                        'success': result.success,
                        'error_message': result.error_message or ''
                    }
                    writer.writerow(row)
            
            self.logger.info(f"Detailed CSV report generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating detailed CSV report: {e}")
            raise


def create_results_processor(results_base_dir: str = RESULTS_BASE_DIR) -> ResultsProcessor:
    """
    Create a results processor instance.
    
    Args:
        results_base_dir: Base directory containing batch result folders
        
    Returns:
        Configured ResultsProcessor instance
    """
    return ResultsProcessor(results_base_dir)


def process_results_directory(results_dir: str, output_csv: Optional[str] = None) -> str:
    """
    Process all results in a directory and generate CSV report.
    
    This is a convenience function for processing existing results.
    
    Args:
        results_dir: Directory containing batch result folders
        output_csv: Optional output CSV path
        
    Returns:
        Path to generated CSV file
    """
    processor = ResultsProcessor(results_dir)
    
    # Find all batch directories
    batch_sizes = []
    for item in os.listdir(results_dir):
        if item.startswith('batch_') and os.path.isdir(os.path.join(results_dir, item)):
            try:
                batch_size = int(item.split('_')[1])
                batch_sizes.append(batch_size)
            except (IndexError, ValueError):
                continue
    
    batch_sizes.sort()
    
    if not batch_sizes:
        raise ValueError(f"No batch directories found in {results_dir}")
    
    # Process all batch results
    results = processor.process_all_batch_results(batch_sizes)
    
    # Generate CSV report
    return processor.generate_csv_report(results, output_csv)


if __name__ == "__main__":
    """
    Standalone script for processing existing results.
    """
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description='Process neuronx benchmark results and generate CSV report'
    )
    parser.add_argument(
        '--results-dir',
        default=RESULTS_BASE_DIR,
        help=f'Directory containing batch result folders (default: {RESULTS_BASE_DIR})'
    )
    parser.add_argument(
        '--output-csv',
        help='Output CSV file path (default: timestamped filename in results dir)'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Generate detailed CSV with error information'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        if args.detailed:
            processor = ResultsProcessor(args.results_dir)
            
            # Find batch sizes
            batch_sizes = []
            for item in os.listdir(args.results_dir):
                if item.startswith('batch_') and os.path.isdir(os.path.join(args.results_dir, item)):
                    try:
                        batch_size = int(item.split('_')[1])
                        batch_sizes.append(batch_size)
                    except (IndexError, ValueError):
                        continue
            
            batch_sizes.sort()
            results = processor.process_all_batch_results(batch_sizes)
            csv_path = processor.generate_detailed_csv_report(results, args.output_csv)
        else:
            csv_path = process_results_directory(args.results_dir, args.output_csv)
        
        print(f"CSV report generated: {csv_path}")
        sys.exit(0)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)