"""Per-kernel parity on CPU via ``nki.simulate`` -- no Trainium required.

This is the gate that matters for correctness. Each kernel is checked against an fp64
reproduction of the exact upstream vortex op chain, per kernel, so a defect cannot be
diluted the way it can be in an end-to-end logits metric.

Two hard-won conventions are asserted here rather than left as comments:
  * lane order -- upstream unpacks ``x2, x1, v = split(...)``, so interleave lane0 -> x2.
    Getting it backwards produced e2e cos_sim -0.894630 and was NOT caught end to end.
  * gamma must be PRE-BROADCAST to (R,D) for the batched norm: a (1,D) operand broadcasts
    across partitions in the SIMULATOR but hardware rejects it.
"""
import numpy as np
import pytest

nki = pytest.importorskip("nki", reason="nki not installed (needs the AWS PyTorch Native env)")

pytestmark = pytest.mark.simulator


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# --------------------------------------------------------------------------- RMSNorm
@pytest.mark.parametrize("D", [4096, 2048, 512, 128])
def test_rmsnorm_single_row(D):
    from evo2_neuron_decode.kernels.rmsnorm_fused import rmsnorm_fused
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, D)).astype(np.float32)
    g = rng.standard_normal((1, D)).astype(np.float32)
    eps = 1e-6
    xd, gd = x.astype(np.float64), g.astype(np.float64)
    # vortex RMSNorm is NON-STANDARD: eps is added to the RMS OUTSIDE the sqrt
    ref = gd * (xd / (np.linalg.norm(xd, 2, axis=-1, keepdims=True) * D ** -0.5 + eps))
    got = nki.simulate(rmsnorm_fused)(x, g, 1)
    assert _cos(got, ref) > 0.999999


@pytest.mark.parametrize("R,D", [(1, 4096), (4, 4096), (32, 4096), (128, 4096), (4, 2048)])
def test_rmsnorm_batched_rows(R, D):
    from evo2_neuron_decode.kernels.rmsnorm_fused import rmsnorm_fused_rows
    rng = np.random.default_rng(0)
    x = rng.standard_normal((R, D)).astype(np.float32)
    g1 = rng.standard_normal((1, D)).astype(np.float32)
    g = np.repeat(g1, R, axis=0)   # PRE-BROADCAST: hardware rejects a (1,D) operand here
    eps = 1e-6
    xd, gd = x.astype(np.float64), g1.astype(np.float64)
    ref = gd * (xd / (np.linalg.norm(xd, 2, axis=-1, keepdims=True) * D ** -0.5 + eps))
    got = nki.simulate(rmsnorm_fused_rows)(x, g, 1)
    assert _cos(got, ref) > 0.999999
    # every ROW must be right, not just the aggregate -- an aggregate can hide one bad row
    worst = min(_cos(np.asarray(got)[i], ref[i]) for i in range(R))
    assert worst > 0.999999, f"worst row cos_sim {worst}"


# --------------------------------------------------------------------------- cascades
# fp64 reproduction of the exact upstream chain, inlined so the test is self-contained.
from evo2_neuron_decode.kernels.hcm_cascade_fused import hcm_cascade_fused
def ref(u,fs,sw,sb,istate,h_full,Dp,D,k):
    """Exact upstream: step_fir(short) -> interleave -> split(x2,x1,v) -> step_fir(inner) -> *x2"""
    u=u.astype(np.float64); fs=fs.astype(np.float64); sw=sw.astype(np.float64)
    zp = sw[:,2]*u + fs[:,0]*sw[:,0] + fs[:,1]*sw[:,1] + sb.astype(np.float64)
    nfs = np.stack([fs[:,1], u], -1)
    x2=zp[0::3]; x1=zp[1::3]; v=zp[2::3]          # lane0=x2, lane1=x1, lane2=v
    g = x1*v
    C = istate.shape[1]
    w = h_full.astype(np.float64)                  # (D, k)
    flip = k>=128
    if flip:
        w = w[:, ::-1][:, -(C+1):]                 # flip then take last C+1
    else:
        w = w[:, :C+1]
    h0 = w[:,-1]; h = w[:,:-1]
    y = h0*g + (istate.astype(np.float64)*h).sum(-1)
    y = y + (Dp.astype(np.float64)*g if flip else Dp.astype(np.float64))
    nis = np.concatenate([istate.astype(np.float64)[:,1:], g[:,None]], -1)
    return y*x2, nfs, nis, h0, h

def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

def case(D,k,P=128,seed=0):
    G=D//P; TD=3*D; C=k-1
    r=np.random.default_rng(seed)
    u=r.standard_normal(TD).astype(np.float32)
    fs=r.standard_normal((TD,2)).astype(np.float32)
    sw=r.standard_normal((TD,3)).astype(np.float32)
    sb=np.zeros(TD,dtype=np.float32)
    istate=r.standard_normal((D,C)).astype(np.float32)
    h_full=r.standard_normal((D,k)).astype(np.float32)
    Dp=r.standard_normal(D).astype(np.float32)
    yr,nfr,nir,h0,h = ref(u,fs,sw,sb,istate,h_full,Dp,D,k)
    gated = 1 if k>=128 else 0
    ys,nfs,nis = nki.simulate(hcm_cascade_fused)(
        u.reshape(D,3).reshape(P,G,3), fs.reshape(D,3,2).reshape(P,G,3,2),
        sw.reshape(D,3,3).reshape(P,G,3,3), sb.reshape(D,3).reshape(P,G,3),
        istate.reshape(P,G,C),
        np.ascontiguousarray(h).astype(np.float32).reshape(P,G,C),
        np.ascontiguousarray(h0).astype(np.float32).reshape(P,G,1),
        Dp.reshape(P,G,1), gated)
    ys=np.asarray(ys).reshape(D); nfs=np.asarray(nfs).reshape(D,3,2).reshape(TD,2)
    nis=np.asarray(nis).reshape(D,C)
    cy=cos(ys,yr); cf=cos(nfs,nfr); ci=cos(nis,nir)
    ok=cy>0.999999 and cf>0.999999 and ci>0.999999
    tag="HCM" if k>=128 else "HCS"
    print(f"  {tag} D={D:<5d} k={k:<4d} y cos={cy:.8f} fir cos={cf:.8f} inner cos={ci:.8f} {'PASS' if ok else 'FAIL'}")
    return ok

@pytest.mark.parametrize("D,k", [(4096, 128), (4096, 7), (2048, 128), (1024, 7), (128, 128),
                                 (512, 7), (256, 128)])
def test_hcm_hcs_cascade(D, k):
    """Covers BOTH paths: k>=128 (HCM, flip + gated bias) and k<128 (HCS, plain bias)."""
    assert case(D, k), f"HCM/HCS cascade parity failed at D={D} k={k}"
