"""Route long-FIR (fir_length>=128, i.e. HCM) through F.conv1d instead of fftconv_func.

WHY: engine.parallel_fir dispatches on `fir_length >= 128` -> fftconv_func, which needs
torch.fft.irfft (complex c2r) -- unsupported on Neuron. `prefill_style` does NOT gate this
path (it only gates the IIR path), so config alone cannot avoid it.

HOW: rather than reimplement parallel_fir (whose post-gate/state contract is subtle), we
replace only `engine.fftconv_func` with a mathematically-equivalent direct causal
depthwise conv1d. Everything downstream (post-gate `z = x2*z`, fir_state, padding_mask)
is untouched upstream code.

fftconv_func(u, k, ...) computes a causal depthwise convolution of u with filter k via
FFT: y = irfft(rfft(u)*rfft(k))[..., :L]. The direct equivalent is
F.conv1d(pad(u, (K-1,0)), k_flipped_as_correlation, groups=D).

NOTE: torch's conv1d is a CROSS-CORRELATION, while FFT multiplication is a true
CONVOLUTION. Upstream's own short-FIR branch uses F.conv1d with padding=fir_length-1
and does NOT flip the weight, so upstream treats these filters as correlation kernels.
We match upstream's short-FIR convention so both paths agree; parity vs the CPU
reference is what validates this.
"""
import torch, torch.nn.functional as F

_orig = None

def install():
    global _orig
    from vortex.model import engine as E
    _orig = E.fftconv_func

    def conv_fftconv_func(u, k, D, dropout_mask, gelu=False, k_rev=None,
                          bidirectional=False, print_activations=False,
                          groups=None, layer_idx=None, **kw):
        # u: (B, D, L)   k: (D_filt, L) or (D_filt, 1, L)
        seqlen = u.shape[-1]
        uf = u.to(torch.float32)
        kf = k.to(torch.float32)
        if kf.dim() == 2:
            kf = kf.unsqueeze(1)               # (D_filt, 1, K)
        Dch = uf.shape[1]
        if kf.shape[0] != Dch:                 # filter groups < channels: repeat
            assert Dch % kf.shape[0] == 0, f"channels {Dch} not divisible by filter groups {kf.shape[0]}"
            kf = kf.repeat_interleave(Dch // kf.shape[0], dim=0)
        K = kf.shape[-1]
        y = F.conv1d(F.pad(uf, (K - 1, 0)), kf, bias=None, groups=Dch)[..., :seqlen]
        if D is not None:
            y = y + uf * (D.to(torch.float32).unsqueeze(-1) if D.dim() == 1 else D)
        if gelu:
            y = F.gelu(y)
        if dropout_mask is not None:
            y = y * dropout_mask.unsqueeze(-1)
        return y.to(u.dtype)

    E.fftconv_func = conv_fftconv_func
    return _orig

def uninstall():
    if _orig is not None:
        from vortex.model import engine as E
        E.fftconv_func = _orig
