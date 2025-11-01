# NKI Coding Patterns and Best Practices

## Essential Patterns from NKI Bootcamp and Reference Kernels

This guide provides concrete coding patterns for common NKI kernel operations, derived from the NKI Bootcamp training and reference kernel implementations.

## Basic Kernel Structure

### Minimal Kernel Template

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np

@nki.jit
def my_kernel(input_tensor, output_tensor):
    """
    Brief description of what this kernel does.
    
    Args:
        input_tensor: Input tensor in HBM, shape [M, N]
        output_tensor: Output tensor in HBM, shape [M, K]
    """
    # Get dimensions
    M, N = input_tensor.shape
    K = output_tensor.shape[1]
    
    # Define tile sizes (use NKI utilities, don't calculate manually)
    TILE_M = 128
    TILE_N = 512
    
    # Allocate tiles on SBUF
    input_tile = nl.ndarray((TILE_M, TILE_N), dtype=input_tensor.dtype, buffer=nl.sbuf)
    output_tile = nl.ndarray((TILE_M, TILE_N), dtype=output_tensor.dtype, buffer=nl.sbuf)
    
    # Load data from HBM to SBUF
    nl.load(input_tile[...], input_tensor[0:TILE_M, 0:TILE_N])
    
    # Perform computation
    # ... kernel logic here ...
    
    # Store result from SBUF to HBM
    nl.store(output_tensor[0:TILE_M, 0:TILE_N], output_tile[...])
```

## Indexing Patterns

### Basic Indexing (No Advanced Indexing Needed)

```python
# Simple slicing - when tile fits evenly
input_tile = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
nl.load(input_tile[...], input_hbm[0:128, 0:512])
```

### Advanced Indexing with nl.mgrid

```python
# Use nl.mgrid for advanced indexing patterns
# DO NOT use nl.arange - it's deprecated

# Create index grids
i_p, i_f = nl.mgrid[0:128, 0:512]

# Basic tiling pattern
tile_offset_p = tile_idx * 128
tile_offset_f = tile_idx * 512

# Load with offsets
nl.load(
    input_tile[...],
    input_hbm[i_p + tile_offset_p, i_f + tile_offset_f]
)

# Access specific attributes
# i_p.p gives partition dimension indices
# i_p.x gives free dimension indices (same as i_f.x)
```

### Transpose Pattern with Advanced Indexing

```python
# Transpose during load (may not be efficient - profile!)
i_p, i_f = nl.mgrid[0:M, 0:N]

# Swap dimensions in indexing
nl.load(
    transposed_tile[...],
    input_hbm[i_f, i_p]  # Note: swapped indices
)
```

### Broadcasting Pattern

```python
# Repeat data across dimension
i_p, i_f = nl.mgrid[0:128, 0:512]

# Broadcast by using same index multiple times
nl.load(
    broadcast_tile[...],
    input_hbm[i_p, 0]  # Repeats column 0 across all columns
)

# Or multiply by 0 to repeat
nl.load(
    broadcast_tile[...],
    input_hbm[i_p, i_f * 0]  # All elements from column 0
)
```

## Tiling Patterns

### Simple Tiling (Exact Multiples)

```python
@nki.jit
def tiled_kernel(input_tensor, output_tensor):
    M, N = input_tensor.shape
    TILE_M = 128
    TILE_N = 512
    
    # Allocate tiles
    input_tile = nl.ndarray((TILE_M, TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)
    output_tile = nl.ndarray((TILE_M, TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)
    
    # Iterate over tiles
    for m in range(0, M, TILE_M):
        for n in range(0, N, TILE_N):
            # Load tile
            nl.load(input_tile[...], input_tensor[m:m+TILE_M, n:n+TILE_N])
            
            # Process tile
            # ... computation ...
            
            # Store tile
            nl.store(output_tensor[m:m+TILE_M, n:n+TILE_N], output_tile[...])
```

### Tiling with Masking (Arbitrary Sizes)

```python
@nki.jit
def masked_tiled_kernel(input_tensor, output_tensor):
    M, N = input_tensor.shape
    TILE_M = 128
    TILE_N = 512
    
    input_tile = nl.ndarray((TILE_M, TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)
    output_tile = nl.ndarray((TILE_M, TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)
    
    for m in range(0, M, TILE_M):
        for n in range(0, N, TILE_N):
            # Calculate actual tile size (may be partial)
            actual_m = min(TILE_M, M - m)
            actual_n = min(TILE_N, N - n)
            
            # Create computational mask for partial tiles
            if actual_m < TILE_M or actual_n < TILE_N:
                i_p, i_f = nl.mgrid[0:TILE_M, 0:TILE_N]
                mask = (i_p < actual_m) & (i_f < actual_n)
            else:
                mask = None
            
            # Load with mask
            nl.load(input_tile[...], input_tensor[m:m+TILE_M, n:n+TILE_N])
            
            # Apply mask during computation if needed
            if mask is not None:
                output_tile[...] = nl.where(mask, computation, 0)
            else:
                output_tile[...] = computation
            
            # Store result
            nl.store(output_tensor[m:m+TILE_M, n:n+TILE_N], output_tile[...])
```

## Matrix Multiplication Patterns

### Basic Matrix Multiplication (Single Tile)

```python
@nki.jit
def matmul_single_tile(A, B, C):
    """
    C = A @ B
    A: [M, K], B: [K, N], C: [M, N]
    Assumes M=128, K=128, N=512 (single tile)
    """
    M, K = A.shape
    N = B.shape[1]
    
    # Allocate tiles
    A_tile = nl.ndarray((M, K), dtype=nl.bfloat16, buffer=nl.sbuf)
    B_tile = nl.ndarray((K, N), dtype=nl.bfloat16, buffer=nl.sbuf)
    C_tile = nl.ndarray((M, N), dtype=nl.bfloat16, buffer=nl.psum)
    
    # Load inputs
    nl.load(A_tile[...], A[...])
    nl.load(B_tile[...], B[...])
    
    # Matrix multiplication using ISA
    # Note: A must be transposed for stationary
    A_transpose = nl.ndarray((K, M), dtype=nl.bfloat16, buffer=nl.sbuf)
    A_transpose[...] = nisa.nc_transpose(A_tile[...])
    
    # Perform matmul
    C_tile[...] = nisa.nc_matmul(A_transpose, B_tile)
    
    # Copy from PSUM to SBUF
    C_sbuf = nl.ndarray((M, N), dtype=nl.bfloat16, buffer=nl.sbuf)
    nl.copy(C_sbuf[...], C_tile[...])
    
    # Store result
    nl.store(C[...], C_sbuf[...])
```

### Tiled Matrix Multiplication with Accumulation

```python
@nki.jit
def matmul_tiled(A, B, C):
    """
    C = A @ B with tiling
    A: [M, K], B: [K, N], C: [M, N]
    """
    M, K = A.shape
    N = B.shape[1]
    
    TILE_M = 128
    TILE_K = 128
    TILE_N = 512
    
    # Allocate tiles
    A_tile = nl.ndarray((TILE_M, TILE_K), dtype=nl.bfloat16, buffer=nl.sbuf)
    B_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)
    C_tile = nl.ndarray((TILE_M, TILE_N), dtype=nl.bfloat16, buffer=nl.psum)
    
    # Iterate over output tiles
    for m in range(0, M, TILE_M):
        for n in range(0, N, TILE_N):
            # Initialize accumulator
            C_tile[...] = 0.0
            
            # Iterate over contraction dimension
            for k in range(0, K, TILE_K):
                # Load tiles
                nl.load(A_tile[...], A[m:m+TILE_M, k:k+TILE_K])
                nl.load(B_tile[...], B[k:k+TILE_K, n:n+TILE_N])
                
                # Transpose A
                A_transpose = nisa.nc_transpose(A_tile)
                
                # Accumulate partial result
                C_tile[...] += nisa.nc_matmul(A_transpose, B_tile)
            
            # Copy accumulated result to SBUF
            C_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)
            nl.copy(C_sbuf[...], C_tile[...])
            
            # Store result
            nl.store(C[m:m+TILE_M, n:n+TILE_N], C_sbuf[...])
```

## Reduction Patterns

### Vector Reduction (Sum/Max/Min)

```python
@nki.jit
def reduce_sum(input_tensor, output_tensor):
    """
    Reduce along last dimension
    Input: [M, N], Output: [M, 1]
    """
    M, N = input_tensor.shape
    TILE_M = 128
    TILE_N = 512
    
    input_tile = nl.ndarray((TILE_M, TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)
    output_tile = nl.ndarray((TILE_M, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    
    for m in range(0, M, TILE_M):
        # Initialize accumulator
        partial_sum = nl.ndarray((TILE_M, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        partial_sum[...] = 0.0
        
        for n in range(0, N, TILE_N):
            # Load tile
            nl.load(input_tile[...], input_tensor[m:m+TILE_M, n:n+TILE_N])
            
            # Reduce along free dimension
            tile_sum = nisa.tensor_reduce(
                input_tile,
                axis=1,  # Free dimension
                reduce_op=nl.add
            )
            
            # Accumulate
            partial_sum[...] += tile_sum
        
        # Store result
        nl.store(output_tensor[m:m+TILE_M, 0:1], partial_sum[...])
```

## Activation Patterns

### Element-wise Activation

```python
@nki.jit
def apply_activation(input_tensor, output_tensor):
    """
    Apply activation function element-wise
    """
    M, N = input_tensor.shape
    TILE_M = 128
    TILE_N = 512
    
    input_tile = nl.ndarray((TILE_M, TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)
    output_tile = nl.ndarray((TILE_M, TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)
    
    for m in range(0, M, TILE_M):
        for n in range(0, N, TILE_N):
            # Load tile
            nl.load(input_tile[...], input_tensor[m:m+TILE_M, n:n+TILE_N])
            
            # Apply activation using scalar engine
            output_tile[...] = nisa.activation(
                input_tile,
                op=nl.exp,  # or nl.relu, nl.tanh, etc.
                dtype=nl.bfloat16
            )
            
            # Store result
            nl.store(output_tensor[m:m+TILE_M, n:n+TILE_N], output_tile[...])
```

### Combined Scalar Operations

```python
# Scalar engine can combine multiple operations
# Example: exp(x - max) for softmax

@nki.jit
def softmax_numerator(input_tensor, max_vals, output_tensor):
    """
    Compute exp(x - max) for softmax
    """
    M, N = input_tensor.shape
    TILE_M = 128
    TILE_N = 512
    
    input_tile = nl.ndarray((TILE_M, TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)
    max_tile = nl.ndarray((TILE_M, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    output_tile = nl.ndarray((TILE_M, TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)
    
    for m in range(0, M, TILE_M):
        # Load max values
        nl.load(max_tile[...], max_vals[m:m+TILE_M, 0:1])
        
        for n in range(0, N, TILE_N):
            # Load input
            nl.load(input_tile[...], input_tensor[m:m+TILE_M, n:n+TILE_N])
            
            # Subtract max (broadcast)
            centered = input_tile - max_tile
            
            # Apply exp
            output_tile[...] = nisa.activation(centered, op=nl.exp, dtype=nl.bfloat16)
            
            # Store result
            nl.store(output_tensor[m:m+TILE_M, n:n+TILE_N], output_tile[...])
```

## Loop Fusion Pattern

### Before Fusion (Inefficient)

```python
# Two separate passes over data
@nki.jit
def unfused_ops(input_tensor, output_tensor):
    M, N = input_tensor.shape
    TILE_M = 128
    TILE_N = 512
    
    # First pass: operation 1
    intermediate = nl.ndarray((M, N), dtype=nl.bfloat16, buffer=nl.hbm)
    for m in range(0, M, TILE_M):
        for n in range(0, N, TILE_N):
            tile = nl.ndarray((TILE_M, TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)
            nl.load(tile[...], input_tensor[m:m+TILE_M, n:n+TILE_N])
            result = operation1(tile)
            nl.store(intermediate[m:m+TILE_M, n:n+TILE_N], result[...])
    
    # Second pass: operation 2
    for m in range(0, M, TILE_M):
        for n in range(0, N, TILE_N):
            tile = nl.ndarray((TILE_M, TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)
            nl.load(tile[...], intermediate[m:m+TILE_M, n:n+TILE_N])
            result = operation2(tile)
            nl.store(output_tensor[m:m+TILE_M, n:n+TILE_N], result[...])
```

### After Fusion (Efficient)

```python
# Single pass with fused operations
@nki.jit
def fused_ops(input_tensor, output_tensor):
    M, N = input_tensor.shape
    TILE_M = 128
    TILE_N = 512
    
    # Single pass: both operations
    for m in range(0, M, TILE_M):
        for n in range(0, N, TILE_N):
            tile = nl.ndarray((TILE_M, TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)
            nl.load(tile[...], input_tensor[m:m+TILE_M, n:n+TILE_N])
            
            # Fuse operations - no intermediate store/load
            intermediate = operation1(tile)
            result = operation2(intermediate)
            
            nl.store(output_tensor[m:m+TILE_M, n:n+TILE_N], result[...])
```

## Common Mistakes to Avoid

### ❌ Using nl.arange Instead of nl.mgrid

```python
# DON'T DO THIS (deprecated)
i = nl.arange(128)
j = nl.arange(512)

# DO THIS INSTEAD
i_p, i_f = nl.mgrid[0:128, 0:512]
```

### ❌ Manually Calculating Tile Sizes

```python
# DON'T DO THIS
TILE_SIZE = (24 * 1024 * 1024) // (128 * 2)  # Manual calculation

# DO THIS INSTEAD
# Review reference kernels for tile size patterns
# Use NKI utilities and hardware constraints
TILE_M = 128  # Partition dimension max
TILE_N = 512  # Common free dimension for tensor engine
```

### ❌ Using Modulo in Indexing

```python
# DON'T DO THIS (not supported)
indices = (nl.arange(1024) % 128)

# DO THIS INSTEAD
# Use nl.mgrid with scales and offsets only
i_p, i_f = nl.mgrid[0:128, 0:512]
scaled_indices = i_p * 2 + 10  # Only scales and offsets allowed
```

### ❌ Forgetting to Transpose for Tensor Engine

```python
# DON'T DO THIS
result = nisa.nc_matmul(A, B)  # A not transposed!

# DO THIS INSTEAD
A_transpose = nisa.nc_transpose(A)
result = nisa.nc_matmul(A_transpose, B)
```

## Performance Optimization Patterns

### Pattern 1: Hoist Loads to Outer Loops

```python
# Before: Load in inner loop (inefficient)
for i in range(outer_iterations):
    for j in range(inner_iterations):
        nl.load(tile, data[...])  # Reloading same data!
        result = compute(tile)

# After: Load in outer loop (efficient)
for i in range(outer_iterations):
    nl.load(tile, data[...])  # Load once
    for j in range(inner_iterations):
        result = compute(tile)  # Reuse loaded data
```

### Pattern 2: Choose Stationary Matrix Wisely

```python
# If A has more tiles than B, make A stationary
# Load stationary is 4x faster than multiply moving

# A: 1024x128 (8 tiles of 128x128)
# B: 128x512 (1 tile)

# Good: A is stationary (8 fast loads)
for tile_a in A_tiles:
    A_transpose = transpose(tile_a)
    result += matmul(A_transpose, B)  # B loaded once

# Bad: B is stationary (8 slow multiply-moving ops)
B_transpose = transpose(B)
for tile_a in A_tiles:
    result += matmul(B_transpose, tile_a)  # A loaded 8 times slowly
```

### Pattern 3: Maximize Partition Dimension Usage

```python
# Bad: Only using 64 of 128 partition lanes
tile = nl.ndarray((64, 512), dtype=nl.bfloat16, buffer=nl.sbuf)

# Good: Using all 128 partition lanes
tile = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)

# Or use partition vectorization to pack multiple operations
```

## Testing Patterns

### Unit Test Template

```python
def test_kernel():
    """Test NKI kernel against reference implementation"""
    # Create test inputs
    input_np = np.random.randn(256, 512).astype(np.float32)
    
    # Run reference implementation
    reference_output = reference_function(input_np)
    
    # Run NKI kernel
    input_nki = torch.from_numpy(input_np).to('xla')
    output_nki = my_kernel(input_nki)
    output_np = output_nki.cpu().numpy()
    
    # Compare results
    np.testing.assert_allclose(
        output_np,
        reference_output,
        rtol=1e-3,  # Relative tolerance
        atol=1e-5   # Absolute tolerance
    )
```

### Benchmark Template

```python
from neuronxcc.nki import benchmark

def benchmark_kernel():
    """Benchmark kernel performance"""
    # Create inputs
    input_tensor = torch.randn(1024, 2048).to('xla')
    output_tensor = torch.zeros(1024, 2048).to('xla')
    
    # Benchmark
    latency_ms = benchmark(
        my_kernel,
        input_tensor,
        output_tensor,
        n_iterations=100
    )
    
    print(f"Average latency: {latency_ms:.3f} ms")
    
    # Calculate throughput
    flops = calculate_flops(operation, input_shape)
    throughput = flops / (latency_ms / 1000)
    print(f"Throughput: {throughput / 1e12:.2f} TFLOPS")
```

## Key Takeaways

1. **Always use nl.mgrid** for advanced indexing (not nl.arange)
2. **Review reference kernels** for tile size patterns
3. **Transpose before tensor engine** matmul
4. **Fuse loops** to reduce memory traffic
5. **Maximize partition dimension** usage (all 128 lanes)
6. **Choose stationary wisely** (more tiles = stationary)
7. **Test against reference** at every step
8. **Benchmark every optimization** to verify improvement

## References

- NKI Bootcamp Sessions 1-2: Programming Model and Indexing
- NKI Samples Repository: Reference implementations
- NKI Documentation: API reference
