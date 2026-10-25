"""Fused RMSNorm for decode (M=1) at D=4096 -- one launch instead of ~7 dispatched ops.

WHY THIS IS THE TOP TARGET NOW (re-profile): with all 32 layers on fused cascade
kernels, the per-layer cost is uniform ~3.44 ms and the SHARED GLUE is ~100% of it. The
largest single glue item is RMSNorm: 0.696 ms per norm, x2 per layer x 32 layers = 64 norms
= ~44 ms of a ~104 ms step. 0.696 ms for a 4096-element reduction is absurd arithmetically
(the FP32 work is nanoseconds) -- it is ~7 dispatched ops at our known ~0.1 ms/launch floor.
So this is pure launch overhead and collapses to a single kernel.

  eager chain: pow -> mean -> add -> rsqrt -> mul -> to(dtype) -> mul(gamma)   (~7 launches)
  fused      : 1 launch

nkilib ships rmsnorm_tkg, but it is an un-decorated SUBKERNEL whose validator demands a
(H0, BxS, H1) partition-split layout that rejected every gamma layout tried
(NCC_INKI016: it infers H0/BxS/H1 from gamma and then requires H %% 128 == 0). Writing our
own 20-line kernel is faster than reverse-engineering that contract, and it lets us keep
the exact eager op order for parity.

LAYOUT: (1, D) -- everything on one partition, so both the reduction and the broadcast of
the resulting scalar are free (no cross-partition reduce, no ones-matmul, no transpose).
We are launch-bound, not compute-bound: 4096 elements on one partition is still ~1000x
faster than the 0.696 ms we are replacing. Simplicity beats a clever multi-partition tiling
here.

ACCUMULATE IN FP32 to match the eager reference.

*** VORTEX'S RMSNorm IS NOT THE STANDARD FORMULA *** (layers.py:195)
    y = x / (x.norm(2, dim=-1, keepdim=True) * hidden_size ** (-1/2) + eps)
    return self.scale * y
eps is added to the RMS *OUTSIDE* the sqrt -- NOT to the mean-square inside it. The standard
`x * rsqrt(mean(x^2) + eps)` is a DIFFERENT function. At eps=1e-6 the numerical gap is tiny,
but "tiny" is how silent drift starts, and the project rule is to match upstream op order
exactly (fp32 arithmetic is not associative). So this kernel computes:
    rms = sqrt(sum(x^2) / D)          [norm(2)*D^-0.5 == sqrt(mean(x^2))]
    out = x / (rms + eps) * scale
"""
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def rmsnorm_fused_rows(x, gamma, eps_e6):
    """Batched RMSNorm: x (R, D) with R rows on R partitions, gamma (1, D) broadcast.

    Each row is an INDEPENDENT normalization, so putting rows on the partition axis makes
    the whole batch one launch: nl.sum(axis=1) reduces along the free (D) axis per partition,
    giving (R,1), which then broadcasts back over the free axis. R <= 128 (partition limit).

    GAMMA IS PASSED AS (R, D), pre-broadcast by the caller. A (1, D) gamma does NOT
    auto-broadcast across partitions on hardware -- nisa.tensor_tensor_arith rejects it
    ("'dst' partition total elements 32 != 'rhs' partition total elements 1"), even though
    THE SIMULATOR ACCEPTS IT SILENTLY. Broadcasting on the free axis (the (R,1) reduction
    result) is fine; broadcasting on the PARTITION axis is not.

    The 1-row kernel shipped first, at BS=1 only; this covers batched decode, where the
    eager path was still costing 2 x 32 = 64 launches per token.
    """
    R, D = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    xt = nl.load(x, dtype=nl.float32)               # (R, D)
    g = nl.load(gamma, dtype=nl.float32)            # (R, D), pre-broadcast by the caller

    sq = nl.multiply(xt, xt)
    ms = nl.sum(sq, axis=1, keepdims=True)          # (R, 1) per-row sum(x^2)
    rms = nl.sqrt(nl.multiply(ms, 1.0 / D))         # (R, 1)
    inv = nl.reciprocal(nl.add(rms, eps_e6 * 1e-6))  # (R, 1)

    nl.store(out, nl.multiply(nl.multiply(xt, inv), g))
    return out


@nki.jit
def rmsnorm_fused(x, gamma, eps_e6):
    """RMSNorm over the last axis. x, gamma: (1, D). eps = eps_e6 * 1e-6.

    eps is passed in units of 1e-6 as an INT so it stays a compile-time constant --
    a float kwarg gets traced as a tensor by the NKI tracer.
    """
    D = x.shape[1]
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    xt = nl.load(x, dtype=nl.float32)           # (1, D) fp32
    g = nl.load(gamma, dtype=nl.float32)

    sq = nl.multiply(xt, xt)                    # x^2
    # keepdims: nl.sum(axis=1) collapses to 1-D, and SBUF tensors need >=2 dims
    ms = nl.sum(sq, axis=1, keepdims=True)      # sum(x^2) -> (1, 1)
    # upstream: norm(2)*D^-0.5 == sqrt(mean(x^2)); eps added OUTSIDE the sqrt
    rms = nl.sqrt(nl.multiply(ms, 1.0 / D))     # (1, 1)
    # NOTE: nl.divide(scalar, tensor) is NOT supported by the ISA layer
    # ("invalid kwarg 'op0': unsupported operator 'divide'"). Use reciprocal.
    inv = nl.reciprocal(nl.add(rms, eps_e6 * 1e-6))    # (1,1), broadcasts on free axis

    nl.store(out, nl.multiply(nl.multiply(xt, inv), g))
    return out
