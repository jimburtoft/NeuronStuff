#!/usr/bin/env python3
"""
Example usage of PIRL Neuron-optimized agent
"""

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

# Import the Neuron-optimized agent
# Note: In your actual project, adjust the import path as needed
try:
    from rl_agent.PIRL_torch_neuron import PIRLagentNeuron, agentOptions, pinnOptions
    NEURON_AVAILABLE = True
except ImportError:
    print("Neuron-optimized agent not found. Make sure to copy PIRL_torch_neuron.py to your project.")
    NEURON_AVAILABLE = False
    exit(1)

# Example neural network (matches PIRL architecture)
class ExampleNeuralNetwork(nn.Module):
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

# Example physics models (simplified for demo)
def example_convection_model(s_and_actIdx):
    """Example convection model"""
    s = s_and_actIdx[:-1]
    x = s[:-1]
    dxdt = np.zeros(15)
    dxdt[0] = 0.1 * x[0]  # Simple dynamics
    dsdt = np.concatenate([dxdt, np.array([-1])])
    return dsdt

def example_diffusion_model(x_and_actIdx):
    """Example diffusion model"""
    diagonals = np.concatenate([0.01*np.ones(5), 0*np.ones(10), np.array([0])])
    sig = np.diag(diagonals)
    return np.matmul(sig, sig.T)

def example_sampling_function(replay_memory):
    """Example sampling function for PINN"""
    n_dim = 16
    nPDE = 32
    nBDini = 32
    nBDsafe = 32
    
    X_PDE = np.random.randn(nPDE, n_dim)
    X_BD_TERM = np.random.randn(nBDini, n_dim)
    X_BD_LAT = np.random.randn(nBDsafe, n_dim)
    
    X_PDE[:, -1] = np.random.uniform(0, 5, nPDE)
    X_BD_TERM[:, -1] = 0
    X_BD_LAT[:, -1] = np.random.uniform(0, 5, nBDsafe)
    
    return X_PDE, X_BD_TERM, X_BD_LAT

def main():
    print("=== PIRL Neuron Optimization Example ===")
    
    # Create model
    model = ExampleNeuralNetwork(input_size=16, output_size=25)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Agent options
    agentOp = agentOptions(
        DISCOUNT=0.99,
        OPTIMIZER=Adam(model.parameters(), lr=5e-4),
        REPLAY_MEMORY_SIZE=1000,
        REPLAY_MEMORY_MIN=100,
        MINIBATCH_SIZE=16,
        EPSILON_INIT=1.0,
        EPSILON_DECAY=0.999,
        EPSILON_MIN=0.01,
    )
    
    # PINN options
    pinnOp = pinnOptions(
        CONVECTION_MODEL=example_convection_model,
        DIFFUSION_MODEL=example_diffusion_model,
        SAMPLING_FUN=example_sampling_function,
        WEIGHT_PDE=1e-4,
        WEIGHT_BOUNDARY=1,
        HESSIAN_CALC=False,
    )
    
    # Create Neuron-optimized agent
    print("Creating Neuron-optimized PIRL agent...")
    agent = PIRLagentNeuron(
        model=model,
        actNum=25,
        agentOp=agentOp,
        pinnOp=pinnOp,
        trace_model=True  # Enable Neuron tracing
    )
    
    print("✓ Agent created successfully!")
    
    # Example single inference
    print("\n=== Single Inference Example ===")
    state = np.random.randn(16)
    q_values = agent.get_qs(state)
    action = agent.get_epsilon_greedy_action(state)
    print(f"State shape: {state.shape}")
    print(f"Q-values shape: {q_values.shape}")
    print(f"Selected action: {action}")
    
    # Example batch inference (optimal for Neuron)
    print("\n=== Batch Inference Example ===")
    batch_sizes = [1, 32, 128, 512]
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        states_batch = np.random.randn(batch_size, 16)
        
        # Time the inference
        import time
        start_time = time.time()
        q_values_batch = agent.get_qs_batch(states_batch)
        end_time = time.time()
        
        throughput = batch_size / (end_time - start_time)
        print(f"  Input shape: {states_batch.shape}")
        print(f"  Output shape: {q_values_batch.shape}")
        print(f"  Time: {(end_time - start_time)*1000:.3f} ms")
        print(f"  Throughput: {throughput:.0f} samples/sec")
    
    # Example performance benchmarking
    print("\n=== Performance Benchmarking ===")
    if hasattr(agent, 'benchmark_inference'):
        print("Running comprehensive benchmark...")
        results = agent.benchmark_inference(
            num_samples=1000,
            batch_sizes=[1, 32, 128, 512, 1024]
        )
        print("✓ Benchmark completed!")
    
    print("\n=== Example Complete ===")
    print("The Neuron-optimized PIRL agent is working correctly!")
    print("For production use:")
    print("  • Use batch sizes ≥512 for optimal performance")
    print("  • Consider hybrid deployment (CPU training, Neuron inference)")
    print("  • Monitor performance with built-in benchmarking tools")

if __name__ == "__main__":
    main()