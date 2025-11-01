# NKI Bootcamp Key Learnings

## Critical Insights from NKI Bootcamp Training

This document captures the most important lessons and insights from the 8-session NKI Bootcamp, organized by topic.

## Development Philosophy

### "Start at the Whiteboard, Not the Keyboard"

**Before writing any code:**
1. Understand the mathematical operation
2. Identify input/output shapes
3. Map operations to compute engines
4. Calculate theoretical peak performance
5. Estimate roofline performance

**Why:** Without understanding what "good" looks like, you can't tell if your kernel is performing well.

### "Indexing is the Hard Part"

**Quote from Bootcamp:** "If you can get indexing right, everything else is easy."

**Key Points:**
- Advanced indexing is the most challenging aspect of NKI
- Most development time is spent getting indexing patterns correct
- Study reference kernels extensively for indexing patterns
- Use nl.mgrid as the foundation for all advanced indexing

### "Profile, Don't Guess"

**Quote from Bootcamp:** "The answer was maybe, but um. We don't guess at optimizations - profile first."

**Key Points:**
- Intuition about performance is often wrong
- The profiler shows exactly what's happening
- Look for gaps in engine timelines
- Check engine utilization percentages
- Measure arithmetic intensity

## Hardware Understanding

### "90% of TFLOPS are in the Tensor Engine"

**Implication:** If you're not using the tensor engine, you're using 10% of the accelerator's capability.

**Design Principle:**
- Optimize for tensor engine first
- Other engines are important but secondary
- Matrix multiplications should dominate compute time

### "Load Stationary is 4X Faster"

**Critical Decision:** Choose which matrix is stationary based on number of tiles.

**Rule:** The matrix with more tiles should be stationary.

**Why:** Load stationary is pure data movement (fast), multiply moving is data movement + compute (slower).

### "Parallelize Over Partition, Linear Over Free"

**Vector/Scalar Engines:**
- Same cost for 1 or 128 elements in partition dimension
- Linear cost in free dimension
- Always maximize partition dimension usage

**Design Implication:** Structure data to maximize partition dimension utilization.

## Memory Management

### "SBUF is Precious"

**Capacity:** ~24MB on-chip memory

**Challenge:** Must fit input tiles + intermediate results + output tiles

**Strategy:**
- Minimize intermediate storage
- Reuse data when possible
- Fuse operations to avoid spilling

### "HBM Traffic is the Enemy"

**Goal:** Minimize data movement between HBM and SBUF

**Techniques:**
- Load data once, use multiple times
- Fuse operations to avoid intermediate stores
- Choose tile sizes that maximize reuse

### "DMA Descriptors Can Eat Your Memory"

**Pre-Neuron 2.20:** DMA descriptors were pre-generated at compile time, consuming GBs of memory.

**Post-Neuron 2.20:** Descriptor Generation Engine (DGE) generates descriptors at runtime.

**Lesson:** Understand how your Neuron version handles DMAs.

## Indexing Insights

### "Advanced Indexing is Pattern Matching"

**Process:**
1. Visualize the output you want
2. Work backwards to determine coordinate arrays needed
3. Figure out how to generate those coordinates using nl.mgrid
4. Apply scales and offsets (only operations allowed)

**Key Constraint:** Can only use scales and offsets, no modulo or division.

### "Masks are Computational, Not Data"

**Common Misconception:** Thinking of masks like attention masks (data masks).

**Reality:** NKI masks are computational - they control which elements are processed.

**Usage:**
- Handle partial tiles at boundaries
- Prevent out-of-bounds accesses
- Use index comparisons to create masks

**Example from max2dpool:** Study this kernel for masking patterns.

## Performance Optimization

### "Arithmetic Intensity is Key"

**Formula:** Arithmetic Intensity = FLOP / Byte

**Goal:** Move from memory-bound to compute-bound region.

**Strategies:**
- Increase data reuse
- Fuse operations
- Increase batch size (if applicable)
- Improve tiling strategy

### "Not All Optimizations Help"

**Quote from Bootcamp:** "If an optimization degrades performance, revert it and document why."

**Key Points:**
- Theoretical optimizations may not help in practice
- Hardware behavior is complex and counterintuitive
- Always benchmark before and after
- Keep a performance log

### "Optimize in Order"

**12-Step Optimization Flow:**
1. NumPy implementation
2. NKI Lang
3. NKI ISA
4. Tiling (multiples)
5. Masking (arbitrary)
6. Performance baseline
7. Loop fusion
8. Operation reordering
9. Advanced optimizations

**Why:** Each step builds on the previous. Skipping steps leads to complex, buggy code.

## Common Pitfalls

### "Don't Use nl.arange"

**Status:** Deprecated

**Use Instead:** nl.mgrid

**Why:** nl.mgrid is the recommended API for generating index arrays.

### "Don't Calculate Tile Sizes Manually"

**Wrong Approach:** Calculating based on SBUF capacity, data types, etc.

**Right Approach:** Use NKI utilities and review reference kernels.

**Why:** NKI provides the right abstractions - use them.

### "Don't Assume NumPy is Ground Truth"

**Correct:** Reference implementation is ground truth.

**NumPy Role:** Unit testing and intermediate validation.

**Why:** NumPy may not have all features/capacity of reference implementation.

### "Don't Optimize Before It Works"

**Quote from Bootcamp:** "Premature optimization leads to complex, buggy code."

**Process:**
1. Get it working (steps 1-6)
2. Establish baseline (step 7)
3. Then optimize (steps 8-9)

## Real-World Case Study: Autodesk Trilinear Interpolation

### The Problem
- Model used trilinear interpolation operator
- Operator not supported by compiler
- Model wouldn't compile

### The Solution Process
1. **Week 1:** Understand the problem, study NumPy implementation
2. **Week 2:** Implement in NKI, hit indexing challenges
3. **Week 3:** Try general approach with indirect HBM access - too slow
4. **Week 4:** Overfit to specific use case (2x upscaling only)
5. **Week 5:** Add tiling, hit SBUF capacity issues
6. **Week 6:** Add SPMD partitioning, finally works end-to-end

### Key Lessons
- **Overfitting is okay:** Don't need to solve the general problem
- **Indexing is hard:** Most time spent on indexing patterns
- **Iterate on approach:** First approach may not work - be ready to pivot
- **Tiling is essential:** Single-tile implementations don't scale
- **6 weeks is realistic:** For complex kernels with learning curve

### Performance Result
- 200-300x slower than CUDA kernel
- But: Unlocked the model on Neuron
- Customer could proceed with migration

**Lesson:** Sometimes "working" is more important than "optimal."

## Profiling Insights

### "Look for Gaps"

**In Profiler:**
- Gaps between operations = poor pipelining
- Gaps within engine timeline = waiting on dependencies
- Small, repeated operations = overhead dominating

**Solutions:**
- Pipeline operations
- Fuse loops
- Increase tile sizes

### "Check Utilization"

**Key Metrics:**
- Tensor engine utilization
- Vector engine utilization
- Scalar engine utilization
- Memory bandwidth utilization

**Goal:** High utilization on bottleneck engine.

### "Understand Your Bottleneck"

**Memory Bound:**
- Low arithmetic intensity
- High HBM traffic
- Need more compute per byte

**Compute Bound:**
- High arithmetic intensity
- Engines at capacity
- Near optimal (or need algorithmic improvement)

## Advanced Topics

### "Software Pipelining is Powerful"

**Concept:** Overlap DMA load of next tile with compute on current tile.

**Benefit:** Can hide memory latency completely.

**Complexity:** Requires careful orchestration.

**When to Use:** After basic optimizations, when memory-bound.

### "Trainium 2 Has New Tricks"

**New Features:**
- DMA transpose (crossbar switch)
- FP8 support
- Hardware sparsity (16:4, etc.)
- Improved SBUF access parallelism

**Caution:** Some features poorly documented - expect to experiment.

### "Allocators Enable Advanced Optimizations"

**Purpose:** Control physical memory layout on SBUF/PSUM.

**When Needed:**
- Working on multiple tiles simultaneously
- Two copies of same variable active
- Advanced pipelining

**Complexity:** Most kernels don't need custom allocators.

## Testing and Validation

### "Unit Tests Save Time"

**Pattern:**
1. Write NumPy reference
2. Create unit tests
3. Validate each NKI stage against tests
4. Catch errors early

**Why:** Debugging a complex kernel is exponentially harder than debugging simple stages.

### "Test with Multiple Shapes"

**Test Cases:**
- Single tile (128x512)
- Multiple tiles (exact multiples)
- Arbitrary sizes (non-multiples)
- Edge cases (very small, very large)

**Why:** Different shapes expose different bugs.

### "Numerical Accuracy Matters"

**Tolerances:**
- BF16: rtol=1e-2, atol=1e-3
- FP16: rtol=1e-3, atol=1e-4
- FP32: rtol=1e-5, atol=1e-6

**Why:** Lower precision = larger acceptable errors.

## Documentation and Communication

### "Document Your Decisions"

**What to Document:**
- Why these tile sizes?
- What hardware constraints influenced design?
- What optimizations were tried?
- What worked and what didn't?

**Why:** Future you will thank present you.

### "Code Comments are Essential"

**What to Comment:**
- Purpose of each section
- Non-obvious indexing patterns
- Hardware-specific workarounds
- Performance considerations

**Why:** NKI code is complex - make it understandable.

## Final Wisdom

### "You Won't Get It Perfect First Time"

**Quote from Bootcamp:** "Writing NKI kernels is an iterative process. You won't get it perfect the first time. That's okay."

**Mindset:**
- Expect to iterate
- Learn from mistakes
- Build incrementally
- Validate constantly

### "The Goal is Correct First, Fast Second"

**Priority:**
1. Correct implementation
2. Functional at scale
3. Performance optimization

**Why:** A fast, wrong kernel is useless. A slow, correct kernel can be optimized.

### "When Stuck, Simplify"

**Debugging Strategy:**
1. Reduce to smallest input
2. Remove optimizations
3. Add logging
4. Compare with NumPy
5. Check one operation at a time

**Why:** Complex problems become tractable when broken down.

## Resources to Review

### Must-Read Kernels
1. **maxpooling.py** - Masking patterns
2. **pipelined_attention.py** - Advanced optimizations
3. **matmul.py** - Basic patterns
4. **interpolate_trilinear_fwd.py** - Real-world complexity

### Key Documentation
1. NKI API Reference
2. Hardware Architecture Guide
3. Performance Guide
4. Profiler Documentation

### Bootcamp Sessions to Re-watch
- **Session 2:** Advanced Indexing (most important)
- **Session 3:** Hardware Deep Dive
- **Session 6:** Performance Optimization
- **Session 8:** 12-Step Optimization Flow

## Remember

The NKI Bootcamp emphasized that kernel development is a skill that improves with practice. Don't expect to be an expert immediately. Follow the process, learn from reference kernels, profile your code, and iterate.

**Most Important Quote:** "If you can get indexing right, everything else is easy."

Focus on mastering indexing patterns, and the rest will follow.
