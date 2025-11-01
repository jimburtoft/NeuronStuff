# NKI Kernel Development Principles

## Core Philosophy

When developing NKI kernels, follow these fundamental principles derived from the NKI Bootcamp training:

### 1. Incremental Development is Mandatory

**Never skip steps.** Each development stage must be fully functional before proceeding:

1. Reference Code → 2. NumPy → 3. NKI Lang → 4. NKI ISA → 5. Tiling (multiples) → 6. Masking (arbitrary) → 7. Optimization

**Why:** Catching errors early is exponentially easier than debugging a complex, optimized kernel. Each step builds confidence and understanding.

### 2. Validate at Every Step

**Always compare against the reference implementation** (not NumPy - that's just for unit tests):
- After NumPy: Does it match reference?
- After NKI Lang: Does it match reference?
- After NKI ISA: Does it match reference?
- After Tiling: Does it match reference?
- After Masking: Does it match reference?
- After each optimization: Does it still match reference AND improve performance?

**Why:** The reference implementation is your ground truth. NumPy is useful for unit testing and intermediate validation, but may not have all features.

### 3. Start Simple, Scale Gradually

**Begin with the smallest possible shapes:**
- Single tile: 128x128 or 128x512
- Simple data types: BF16 or FP32
- No edge cases initially

**Then scale:**
- Multiple tiles (exact multiples)
- Arbitrary sizes (with masking)
- Complex data patterns

**Why:** Simple cases are easy to understand, debug, and validate. Complexity should be added only after basics work.

### 4. Understand Before Optimizing

**Complete the functional implementation before optimizing:**
- Steps 1-6: Focus on correctness
- Step 7: Establish performance baseline
- Steps 8-9: Apply optimizations

**Why:** Premature optimization leads to complex, buggy code. You need a working baseline to measure improvements against.

### 5. Benchmark Every Optimization

**Use `nki.benchmark` after each optimization:**
```python
# Before optimization
baseline_latency = nki.benchmark(kernel_v1, inputs, n_iterations=100)

# After optimization
optimized_latency = nki.benchmark(kernel_v2, inputs, n_iterations=100)

# Verify improvement
assert optimized_latency < baseline_latency, "Optimization should improve performance"
```

**If performance degrades, revert the optimization and document why.**

**Why:** Not all theoretical optimizations help in practice. Hardware behavior is complex and counterintuitive.

### 6. Hardware Constraints Drive Design

**Always design with hardware in mind:**

**Tensor Engine:**
- Stationary matrix: max 128x128
- Moving matrix: max 128x512
- Use for matrix multiplications
- Fast load stationary: 4x faster than multiply moving

**Vector Engine:**
- 128 parallel lanes (partition dimension)
- Use for reductions and vector operations
- Parallelize over partition dimension

**Scalar Engine:**
- 128 parallel lanes (partition dimension)
- Use for activations and element-wise ops
- Can combine multiple instructions

**Memory:**
- SBUF: ~24MB on-chip memory
- HBM: Off-chip, slower access
- Minimize HBM traffic

**Why:** The hardware architecture determines what's fast and what's slow. Fighting the hardware leads to poor performance.

### 7. Profile to Find Bottlenecks

**Use Neuron Profiler to identify issues:**
- Large gaps in engine timelines → poor pipelining
- Low engine utilization → suboptimal tile sizes
- High HBM traffic → need better data reuse

**Don't guess at optimizations - profile first.**

**Why:** Intuition about performance is often wrong. The profiler shows you exactly what's happening.

### 8. Reference Kernels are Your Best Teacher

**Study existing kernels in nki-samples:**
- How do they handle tiling?
- How do they use masking?
- How do they determine tile sizes?
- What optimizations do they apply?

**Especially review:**
- `maxpooling.py` - for masking patterns
- `pipelined_attention.py` - for advanced optimizations
- `matmul.py` - for basic patterns

**Why:** These kernels embody best practices and proven patterns. Don't reinvent the wheel.

### 9. Documentation is Part of Development

**Document as you go:**
- Why did you choose these tile sizes?
- What hardware constraints influenced your design?
- What optimizations did you try?
- What worked and what didn't?

**Why:** Future you (and others) will need to understand and maintain this code.

### 10. When Stuck, Simplify

**If something isn't working:**
1. Reduce input size to smallest possible
2. Remove optimizations
3. Add print statements / logging
4. Compare intermediate results with NumPy
5. Check one operation at a time

**Why:** Complex problems become tractable when broken into simple pieces.

## Common Pitfalls to Avoid

### ❌ Don't: Skip validation steps
**Why:** Errors compound and become impossible to debug

### ❌ Don't: Optimize before it works
**Why:** You'll waste time optimizing broken code

### ❌ Don't: Manually calculate tile sizes
**Why:** NKI provides utilities for this - use them

### ❌ Don't: Assume NumPy is ground truth
**Why:** Reference implementation is ground truth; NumPy is for unit testing

### ❌ Don't: Ignore profiler data
**Why:** Guessing at optimizations wastes time

### ❌ Don't: Keep optimizations that hurt performance
**Why:** Not all optimizations help - revert bad ones

### ❌ Don't: Use attention-style masks for tiling
**Why:** NKI masks are computational (index-based), not data masks

### ❌ Don't: Proceed when tests fail
**Why:** Fix issues immediately while context is fresh

## Development Workflow Summary

```
1. Setup environment (verify neuronx-cc)
2. Establish reference code (ground truth)
3. Implement in NumPy (for unit tests)
4. Implement in NKI Lang (high-level)
5. Convert to NKI ISA (low-level control)
6. Add tiling (for larger inputs)
7. Add masking (for arbitrary sizes)
8. Profile and establish baseline
9. Optimize (loop fusion, operation reordering)
10. Document and finalize

At each step: Validate → Benchmark → Document
```

## Key Mantras

- **"Does it match the reference?"** - Ask this constantly
- **"Start simple, scale gradually"** - Resist the urge to tackle everything at once
- **"Profile before optimizing"** - Don't guess
- **"Benchmark every change"** - Measure, don't assume
- **"When in doubt, check the samples"** - Learn from working code
- **"If it's not working, simplify"** - Break problems down

## Remember

Writing NKI kernels is an iterative process. You won't get it perfect the first time. That's okay. Follow the process, validate at each step, and you'll build working, performant kernels.

The goal is not to write the fastest kernel immediately - it's to write a correct kernel first, then make it fast through measured, validated optimizations.
