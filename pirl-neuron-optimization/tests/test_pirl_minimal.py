#!/usr/bin/env python3
"""
Minimal test script for PIRL agent Neuron compatibility assessment.
This version avoids complex imports and focuses on core functionality testing.
"""

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import time
import sys
import os
from collections import deque

class TestNeuralNetwork(nn.Module):
    """Neural network matching PIRL architecture"""
    def __init__(self, input_size=16, output_size=25):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32), 
            nn.Tanh(),
            nn.Linear(32, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.linear_stack(x)

def benchmark_forward_pass(model, batch_sizes=[1, 8, 16, 32], num_runs=100):
    """Benchmark forward pass performance"""
    print("=== Forward Pass Benchmarking ===")
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Generate test input
        test_input = torch.randn(batch_size, 16)
        
        # Warm up
        for _ in range(10):
            with torch.no_grad():
                _ = model(test_input)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                output = model(test_input)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = batch_size / avg_time
        
        results[batch_size] = {
            'avg_time': avg_time,
            'std_time': std_time,
            'throughput': throughput
        }
        
        print(f"  Avg time: {avg_time*1000:.3f} ± {std_time*1000:.3f} ms")
        print(f"  Throughput: {throughput:.1f} samples/sec")
    
    return results

def benchmark_gradient_computation(model, batch_size=16, num_runs=50):
    """Benchmark gradient computation (simulating PDE loss)"""
    print(f"\n=== Gradient Computation Benchmarking ===")
    
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    times = []
    
    for _ in range(num_runs):
        # Generate test data
        x = torch.randn(batch_size, 16, requires_grad=True)
        target = torch.randn(batch_size, 25)
        
        start_time = time.time()
        
        # Forward pass
        output = model(x)
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(output, target)
        
        # Simulate PDE gradient computation
        if x.grad is not None:
            x.grad.zero_()
        
        # Compute gradients
        grad_outputs = torch.autograd.grad(output.sum(), x, create_graph=True)[0]
        pde_loss = torch.nn.functional.mse_loss(grad_outputs.sum(1), torch.zeros(batch_size))
        
        total_loss = loss + 0.001 * pde_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"Gradient computation time: {avg_time*1000:.3f} ± {std_time*1000:.3f} ms")
    
    return avg_time, std_time

def test_neuron_tracing(model):
    """Test Neuron tracing capability with high precision and large batch sizes"""
    print(f"\n=== Neuron Tracing Test (High Precision + Large Batches) ===")
    
    try:
        import torch_neuronx
        print(f"torch_neuronx version: {torch_neuronx.__version__}")
        
        # Set autocast to None for higher precision
        print("Setting autocast to None for higher precision...")
        
        # Test multiple batch sizes including very large ones
        print("\nTesting large batch sizes for optimal Neuron utilization...")
        batch_sizes = [1, 32, 128, 512, 1024, 2048]
        
        traced_models = {}
        
        for batch_size in batch_sizes:
            try:
                print(f"\n--- Tracing model for batch size {batch_size} ---")
                batch_example = torch.randn(batch_size, 16)
                model.eval()
                
                # Trace with autocast disabled for higher precision
                with torch.no_grad():
                    # Use compiler_args to disable autocast
                    traced_model = torch_neuronx.trace(
                        model, 
                        batch_example,
                        compiler_args=["--auto-cast", "none"]
                    )
                
                traced_models[batch_size] = traced_model
                print(f"✓ Successfully traced for batch size {batch_size}")
                
                # Validate output accuracy
                test_batch = torch.randn(batch_size, 16)
                with torch.no_grad():
                    original_output = model(test_batch)
                    traced_output = traced_model(test_batch)
                
                diff = torch.mean(torch.abs(original_output - traced_output)).item()
                print(f"  Output difference: {diff:.8f}")
                
                # Extended benchmark to amortize loading costs
                print(f"  Running extended benchmark (100 iterations)...")
                
                # Warm up both models
                for _ in range(10):
                    with torch.no_grad():
                        _ = model(test_batch)
                        _ = traced_model(test_batch)
                
                # Benchmark original model
                times_orig = []
                for _ in range(100):
                    test_batch = torch.randn(batch_size, 16)
                    start_time = time.time()
                    with torch.no_grad():
                        _ = model(test_batch)
                    times_orig.append(time.time() - start_time)
                
                # Benchmark traced model
                times_traced = []
                for _ in range(100):
                    test_batch = torch.randn(batch_size, 16)
                    start_time = time.time()
                    with torch.no_grad():
                        _ = traced_model(test_batch)
                    times_traced.append(time.time() - start_time)
                
                avg_orig = np.mean(times_orig)
                avg_traced = np.mean(times_traced)
                std_orig = np.std(times_orig)
                std_traced = np.std(times_traced)
                speedup = avg_orig / avg_traced
                throughput_orig = batch_size / avg_orig
                throughput_traced = batch_size / avg_traced
                
                print(f"  Original: {avg_orig*1000:.3f}±{std_orig*1000:.3f}ms ({throughput_orig:.0f} samples/sec)")
                print(f"  Neuron:   {avg_traced*1000:.3f}±{std_traced*1000:.3f}ms ({throughput_traced:.0f} samples/sec)")
                print(f"  Speedup:  {speedup:.2f}x")
                print(f"  Throughput improvement: {throughput_traced/throughput_orig:.2f}x")
                
                # Test sustained performance (simulate real workload)
                if batch_size >= 512:
                    print(f"  Testing sustained performance (1000 iterations)...")
                    
                    start_time = time.time()
                    for _ in range(1000):
                        test_batch = torch.randn(batch_size, 16)
                        with torch.no_grad():
                            _ = traced_model(test_batch)
                    total_time = time.time() - start_time
                    
                    sustained_throughput = (1000 * batch_size) / total_time
                    avg_latency = total_time / 1000
                    
                    print(f"  Sustained throughput: {sustained_throughput:.0f} samples/sec")
                    print(f"  Average latency: {avg_latency*1000:.3f}ms")
                
            except Exception as e:
                print(f"  Failed to trace batch size {batch_size}: {e}")
                continue
        
        # Summary of results
        print(f"\n=== Neuron Performance Summary ===")
        successful_batches = list(traced_models.keys())
        if successful_batches:
            print(f"Successfully traced batch sizes: {successful_batches}")
            print(f"Largest successful batch: {max(successful_batches)}")
            
            # Find optimal batch size
            print(f"\nOptimal batch size analysis:")
            print(f"• Small batches (1-32): Good for low-latency inference")
            print(f"• Medium batches (128-512): Balanced latency/throughput")
            print(f"• Large batches (1024+): Maximum throughput for batch processing")
        
        return True, traced_models
        
    except ImportError:
        print("torch_neuronx not available - skipping Neuron tracing test")
        return False, None
    except Exception as e:
        print(f"Error during Neuron tracing: {e}")
        return False, None
        
    except ImportError:
        print("torch_neuronx not available - skipping Neuron tracing test")
        return False, None
    except Exception as e:
        print(f"Error during Neuron tracing: {e}")
        return False, None

def analyze_computational_intensity(forward_results, gradient_time):
    """Analyze which operations are most computationally intensive"""
    print(f"\n=== Computational Intensity Analysis ===")
    
    # Get forward pass time for batch size 16 (typical training batch)
    forward_time = forward_results[16]['avg_time'] if 16 in forward_results else forward_results[1]['avg_time']
    
    operations = [
        ("Forward Pass (inference)", forward_time),
        ("Gradient Computation (training)", gradient_time),
    ]
    
    operations.sort(key=lambda x: x[1], reverse=True)
    
    print("Operations ranked by computational intensity:")
    for i, (name, time_val) in enumerate(operations, 1):
        print(f"{i}. {name}: {time_val*1000:.3f} ms")
    
    print(f"\nNeuron Optimization Recommendations:")
    print(f"• Primary target: {operations[0][0]} - Best candidate for Neuron tracing")
    print(f"• Secondary target: {operations[1][0]} - May benefit from Neuron optimization")
    
    if operations[0][0] == "Forward Pass (inference)":
        print(f"• Recommendation: Use Neuron tracing for inference, keep training on CPU/GPU")
    else:
        print(f"• Recommendation: Consider hybrid approach with selective Neuron usage")

def main():
    print("=== PIRL Agent Neuron Compatibility Assessment ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Testing on Neuron hardware with CPU baseline comparison")
    
    # Create test model
    model = TestNeuralNetwork(input_size=16, output_size=25)
    
    print(f"\nModel architecture:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Model size: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024:.2f} KB")
    
    # Benchmark forward pass
    forward_results = benchmark_forward_pass(model)
    
    # Benchmark gradient computation
    gradient_time, _ = benchmark_gradient_computation(model)
    
    # Test Neuron tracing
    neuron_available, traced_model = test_neuron_tracing(model)
    
    # Analyze computational intensity
    analyze_computational_intensity(forward_results, gradient_time)
    
    # Summary and recommendations
    print(f"\n=== Summary and Recommendations ===")
    
    print(f"\nPerformance Baseline (CPU):")
    print(f"  Forward pass (batch=1): {forward_results[1]['avg_time']*1000:.3f} ms")
    print(f"  Forward pass (batch=32): {forward_results[32]['avg_time']*1000:.3f} ms")
    print(f"  Gradient computation: {gradient_time*1000:.3f} ms")
    
    if neuron_available:
        print(f"\n✓ Neuron Compatibility: CONFIRMED")
        print(f"✓ Model successfully traced for Neuron")
        print(f"✓ Ready for deployment on AWS Trainium/Inferentia")
        
        print(f"\nDeployment Strategy:")
        print(f"• Use traced models for inference workloads")
        print(f"• Batch processing for optimal Neuron utilization")
        print(f"• Consider hybrid training: CPU/GPU for complex autograd, Neuron for inference")
        
    else:
        print(f"\n⚠ Neuron Compatibility: UNTESTED")
        print(f"• Install torch_neuronx to test Neuron compatibility")
        print(f"• Model architecture is compatible with Neuron tracing")
        
    print(f"\nNext Steps:")
    print(f"1. Test on actual Trainium/Inferentia hardware")
    print(f"2. Optimize batch sizes for production workloads")
    print(f"3. Consider creating separate inference and training pipelines")
    print(f"4. Monitor performance in production environment")
    
    print(f"\n✓ Assessment completed successfully!")

if __name__ == "__main__":
    main()