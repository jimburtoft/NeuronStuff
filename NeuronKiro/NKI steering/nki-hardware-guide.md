# NKI Hardware Architecture Guide

## NeuronCore Architecture Overview

Understanding the hardware is essential for writing efficient NKI kernels. This guide summarizes key hardware concepts from the NKI Bootcamp.

## Compute Engines

### Tensor Engine (Matrix Multiplication)

**Purpose:** Dedicated engine for matrix multiplication

**Architecture:**
- 128x128 systolic array
- Highest throughput engine in NeuronCore
- ~90% of TFLOPS in typical LLMs

**Tile Constraints:**
- **Stationary matrix:** Max 128x128
- **Moving matrix:** Max 128x512
- **Contraction dimension:** Must be mapped to partition dimension

**Key Operations:**
- Matrix multiplication: `nki.isa.nc_matmul`
- Transpose (with identity): `nki.isa.nc_transpose`
- Broadcast (with ones vector)

**Performance Characteristics:**
- **Load stationary:** Pure data movement, 4x faster than multiply moving
- **Multiply moving:** Data movement + computation
- **Choose stationary wisely:** Larger tensor should be stationary (more tiles = more loads)

**Example:**
```python
# Stationary: 128x128 (transposed)
# Moving: 128x512
# Output: 128x512 (in PSUM, transposed)

nki.isa.nc_matmul(
    stationary=x_transpose,  # 128x128
    moving=y,                 # 128x512
    output=result            # 128x512 in PSUM
)
```

### Vector Engine (Reductions & Vector Ops)

**Purpose:** Vector operations and reductions

**Architecture:**
- 128 parallel lanes
- Maps to partition dimension
- Linear cost along free dimension
- Constant cost across partition dimension

**Tile Constraints:**
- **Partition dimension:** 128 lanes (parallelize here)
- **Free dimension:** 64K from SBUF, 4K from PSUM

**Key Operations:**
- Reductions: `nki.isa.tensor_reduce`
- Vector operations: `nki.isa.vector_*`

**Performance Characteristics:**
- **Parallel over P dimension:** Same cost for 1 or 128 elements
- **Linear over F dimension:** Cost scales with number of elements
- **Maximize P utilization:** Always try to use all 128 lanes

**Example:**
```python
# Reduce along free dimension (F)
# Parallelize across partition dimension (P)
result = nki.isa.tensor_reduce(
    input_tile,      # Shape: [128, 512]
    axis=1,          # Reduce along F dimension
    reduce_op='sum'
)
# Output shape: [128, 1] - one result per partition lane
```

### Scalar Engine (Activations & Element-wise)

**Purpose:** Scalar operations, activations, element-wise operations

**Architecture:**
- 128 parallel lanes
- Maps to partition dimension
- Can combine multiple instructions

**Tile Constraints:**
- **Partition dimension:** 128 lanes
- **Free dimension:** 64K from SBUF, 4K from PSUM

**Key Operations:**
- Activations: `nki.isa.activation` (with op codes)
- Element-wise operations

**Performance Characteristics:**
- **Instruction combination:** Can pack multiple ops in one call
- **Parallel over P dimension:** Use all 128 lanes
- **Overhead per instruction:** ~100 cycles setup time

**Example:**
```python
# Single activation
output = nki.isa.activation(
    input_tile,
    op=nl.exp,  # Exponential activation
    dtype=nl.bfloat16
)

# Combined instructions (advanced)
# Multiple operations in single scalar engine call
```

### GPSIMD (General Purpose)

**Purpose:** Catch-all for operations not suited to other engines

**Architecture:**
- 8 cores, 16 total lanes
- Statically mapped to partition dimension
- Less parallelism than other engines

**When to Use:**
- Operations not supported by other engines
- Triangular masking
- Complex indexing patterns
- Custom operations

**Performance Characteristics:**
- **Lower parallelism:** Only 16 lanes vs 128
- **Static mapping:** Can't stride over partition dimension
- **Use sparingly:** Prefer other engines when possible

## Memory Hierarchy

### HBM (High Bandwidth Memory)

**Characteristics:**
- **Capacity:** 16GB per core (NCV2), 32GB per core (NCV3)
- **Location:** Off-chip, local to core
- **Speed:** Slower than on-chip memory
- **Access:** Via DMA engines

**Usage:**
- Store model weights
- Store input/output tensors
- Spill intermediate results when SBUF full

**Best Practices:**
- Minimize HBM traffic
- Load data once, reuse in SBUF
- Batch operations to reduce transfers

### SBUF (State Buffer)

**Characteristics:**
- **Capacity:** ~24MB on-chip
- **Layout:** 2D memory (Partition x Free)
- **Dimensions:** 128 partitions x variable free dimension
- **Speed:** Fast, on-chip access

**Usage:**
- Primary working memory for kernels
- Input tiles for compute engines
- Intermediate results

**Best Practices:**
- Manage capacity carefully
- Reuse data when possible
- Avoid unnecessary spilling to HBM

**Memory Layout:**
```
Partition Dimension (P): 128 lanes
    ↓
[0][1][2]...[127]  ← Free Dimension (F) →
```

### PSUM (Partial Sum Buffer)

**Characteristics:**
- **Capacity:** Smaller than SBUF
- **Purpose:** Accumulate results from Tensor Engine
- **Layout:** 2D memory (Partition x Free)

**Usage:**
- Tensor Engine output
- Accumulate partial matrix multiplications
- Intermediate storage for multi-tile operations

**Best Practices:**
- Use for accumulation in tiled matrix multiplication
- Copy to SBUF when accumulation complete
- Don't use as general storage

## Data Movement

### DMA Engines

**Purpose:** Move data between HBM and SBUF/PSUM

**Characteristics:**
- Multiple DMA engines per core
- Can operate in parallel with compute
- Prefer large, contiguous transfers

**Best Practices:**
- **Large transfers:** Batch small transfers into larger ones
- **Contiguous access:** Avoid strided access patterns
- **Pipeline:** Overlap DMA with compute

**Anti-patterns:**
- Small, frequent transfers (high overhead)
- Strided access (creates "bubbles")
- Sequential DMA + compute (no overlap)

### DMA Transpose (Trainium 2 Only)

**Purpose:** Hardware-accelerated transpose during DMA

**Characteristics:**
- Uses crossbar switch
- Single descriptor for striped writes
- Faster than tensor engine transpose for some patterns

**When to Use:**
- When loading data that needs transposing
- When memory layout allows efficient crossbar use

**When NOT to Use:**
- Complex access patterns (may have port collisions)
- When tensor engine transpose is faster

## Hardware Constraints

### Tile Size Limits

**Tensor Engine:**
- Stationary: 128 x 128 (max)
- Moving: 128 x 512 (max)

**Vector/Scalar Engines:**
- Partition: 128 (max)
- Free: 64K from SBUF, 4K from PSUM

### Memory Capacity

**SBUF:** ~24MB total
- Must fit: input tiles + intermediate results + output tiles
- Careful management required for large operations

**PSUM:** Smaller than SBUF
- Primarily for tensor engine accumulation
- Limited capacity for intermediate storage

### Indexing Constraints

**Advanced Indexing:**
- Can only use scales and offsets
- Cannot use modulo or division
- Must use `nl.mgrid` as base

**Striding:**
- Cannot stride over partition dimension
- Can stride over free dimension (but may be inefficient)

## Performance Optimization Principles

### 1. Maximize Compute Engine Utilization

**Spatial Domain:**
- Use full tile sizes when possible
- Utilize all 128 partition lanes
- Avoid tiny tiles (high overhead)

**Time Domain:**
- Pipeline operations (overlap compute and DMA)
- Minimize idle time between operations
- Keep engines busy

### 2. Minimize Memory Traffic

**Data Reuse:**
- Load data once, use multiple times
- Keep frequently accessed data in SBUF
- Avoid unnecessary spilling to HBM

**Tiling Strategy:**
- Choose tile sizes that maximize reuse
- Consider data dependencies
- Balance SBUF capacity with reuse opportunities

### 3. Choose the Right Engine

**Tensor Engine:**
- Matrix multiplications
- Transposes (with identity matrix)
- Broadcasts (with ones vector)

**Vector Engine:**
- Reductions (sum, max, min)
- Vector operations

**Scalar Engine:**
- Activations (ReLU, GELU, etc.)
- Element-wise operations
- Combined operations

### 4. Pipeline Operations

**Overlap:**
- DMA load + compute on previous tile
- Multiple engine operations in parallel
- Prepare next tile while processing current

**Software Pipelining:**
- Advanced technique for maximum throughput
- Requires careful orchestration
- Significant performance gains possible

## Roofline Model

### Understanding Performance Bounds

**Arithmetic Intensity = FLOP / Byte**

**Memory Bound:**
- Low arithmetic intensity
- Performance limited by memory bandwidth
- Need to increase compute per byte loaded

**Compute Bound:**
- High arithmetic intensity
- Performance limited by compute throughput
- Already maximizing hardware utilization

**Ridge Point:**
- Transition between memory and compute bound
- Hardware-specific value
- Goal: Move from memory bound to compute bound

### Optimization Strategy

**If Memory Bound:**
- Increase data reuse
- Fuse operations to avoid intermediate storage
- Increase batch size (if applicable)
- Improve tiling strategy

**If Compute Bound:**
- Already near optimal
- Focus on reducing overhead
- Consider algorithmic improvements
- May be at hardware limit

## Hardware Generations

### NeuronCore V2 (Trainium 1, Inferentia 2)

**Key Features:**
- 128x128 tensor engine
- 16GB HBM per core
- FP32, BF16, FP16 support

**Limitations:**
- Vector/Scalar engines cannot access SBUF in parallel at full bandwidth
- No DMA transpose
- No hardware sparsity support

### NeuronCore V3 (Trainium 2)

**Key Features:**
- Improved parallel access to SBUF
- DMA transpose (crossbar switch)
- FP8 support
- Hardware sparsity support (16:4, etc.)
- Descriptor Generation Engine (DGE)

**New Capabilities:**
- Software DGE (backported to V2)
- Improved memory efficiency
- Better pipelining opportunities

## Key Takeaways

1. **Tensor Engine is King:** 90% of TFLOPS - optimize for it
2. **Memory is Precious:** SBUF is limited - manage carefully
3. **Parallelize Over P:** Always use all 128 partition lanes
4. **Pipeline Everything:** Overlap compute and data movement
5. **Profile, Don't Guess:** Use Neuron Profiler to find bottlenecks
6. **Hardware Constraints are Real:** Design within limits, don't fight them

## References

- NKI Bootcamp Session 3: Hardware Deep Dive
- NKI Bootcamp Session 4: Trainium 2 Features
- Neuron Hardware Architecture Documentation
