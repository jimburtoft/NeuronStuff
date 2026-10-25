"""Real-valued modal SSM for StripedHyena 1 (2 complex states per channel), no complex dtypes, no FFT.

The Hyena long filter is modal:  h_t = Re( sum_k r_k p_k^t ),  k = 0..1 (``state_size: 2``).
So the FFT long convolution ``y = h * x`` (``engine.parallel_iir``) is exactly the output of the
recurrence  s_t = p s_{t-1} + x_t,  y_t = Re(sum_k r_k s_{k,t}),  and the decode state after a
prefill of length L (``prefill_via_modal_fft``) is s_{L-1} = sum_t p^(L-1-t) x_t.

``modal_conv`` computes both with real fp32 matmuls only (chunk size C):
  1. intra-chunk:  y_intra[n, t] = sum_{j<=t} h_{t-j} x[n, j]   -> one (C x C) Toeplitz per channel, bmm
  2. chunk ends:   E_n = sum_j p^(C-1-j) x[n, j]                -> bmm, complex as re/im pairs
  3. carry:        S_n = q S_{n-1} + E_n, q = p^C                -> Hillis-Steele scan, log2(N) steps
  4. inter-chunk:  y_inter[n, t] = Re sum_k r_k p^(t+1) S_{n-1}  -> bmm
Input is LEFT-padded with zeros to a multiple of C (zeros before the sequence do not change the
state), so the final state is exactly S_{N-1}. Only multiplications by |p|^m <= 1 occur (no
p^-j factorization), so it is numerically stable for every pole.

Tables are built once per (layer, C) on the host in float64 from the fp32 poles/residues and
stored fp32 (``ModalTables``). Decode uses the contiguous re/im planes (``p_re`` ...).
"""
import math

import torch
import torch.nn.functional as F


class ModalTables:
    def __init__(self, poles, residues, C, max_len, device):
        """poles/residues: (D, K, 1, 2) fp32 parameters (re, im)."""
        p = torch.complex(poles[..., 0, 0].detach().double().cpu(), poles[..., 0, 1].detach().double().cpu())
        r = torch.complex(residues[..., 0, 0].detach().double().cpu(), residues[..., 0, 1].detach().double().cpu())
        D, K = p.shape
        self.C, self.D, self.K = C, D, K
        logp = torch.log(p)
        t = torch.arange(C + 1, dtype=torch.float64)
        pt = torch.exp(logp[..., None] * t)                          # (D, K, C+1): p^0 .. p^C
        h = (r[..., None] * pt[..., :C]).real.sum(1)                 # (D, C)
        # Toeplitz H[d, t, j] = h[t - j] (t >= j)
        idx = torch.arange(C)[:, None] - torch.arange(C)[None, :]
        H = torch.where(idx >= 0, h[:, idx.clamp(min=0)], torch.zeros((), dtype=torch.float64))
        pe = pt[..., :C].flip(-1)                                    # p^(C-1-j)
        A = r[..., None] * pt[..., 1:]                               # r p^(t+1), t = 0..C-1
        q = pt[..., C]                                               # p^C
        n_lvl = max(1, math.ceil(math.log2(max(2, math.ceil(max_len / C)))))
        qp, cur = [], q.clone()
        for _ in range(n_lvl):
            qp.append(cur)
            cur = cur * cur
        qp = torch.stack(qp)                                         # (n_lvl, D, K): q^(2^i)
        f = lambda z: z.to(torch.float32).contiguous().to(device)   # noqa: E731
        self.H = f(H)                                                # (D, C, C)
        self.PE = f(torch.cat([pe.real, pe.imag], 1))                # (D, 2K, C)
        self.AT = f(torch.cat([A.real, -A.imag], 1).transpose(1, 2))  # (D, C, 2K)
        self.q_re = f(qp.real[..., None, None])                      # (n_lvl, D, K, 1, 1)
        self.q_im = f(qp.imag[..., None, None])
        self.n_lvl = n_lvl
        # semiseparable ("tri") form: W = p^-j, G = r p^t, Pc1 = p^(C-1), Utri[j, t] = 1 (j <= t)
        # valid only if p^-(C-1) stays finite in fp32 (trained poles |p| >= 0.94 -> <= 49 at C=64)
        self.tri_ok = float(p.abs().min()) ** (-(C - 1)) < 1e30
        pinv = torch.exp(-logp[..., None] * t[:C])                   # (D, K, C)
        G = r[..., None] * pt[..., :C]
        self.W_re, self.W_im = f(pinv.real), f(pinv.imag)            # (D, K, C)
        self.G_re, self.G_im = f(G.real), f(G.imag)
        self.Pc1_re, self.Pc1_im = f(pt[..., C - 1].real[..., None]), f(pt[..., C - 1].imag[..., None])  # (D,K,1)
        self.Utri = f(torch.triu(torch.ones(C, C, dtype=torch.float64)))   # (C, C): [j, t] = 1 if j <= t


def modal_conv(x, tab: ModalTables):
    """x: (B, D, L) fp32. Returns y (B, D, L) fp32 and final state (s_re, s_im), each (B, D, K) fp32."""
    B, D, L = x.shape
    C, K = tab.C, tab.K
    N = -(-L // C)
    Lp = N * C
    pad = Lp - L
    if pad:
        x = F.pad(x, (pad, 0))
    X = x.reshape(B, D, N, C).permute(1, 3, 0, 2).reshape(D, C, B * N)          # (D, C, BN)
    y = torch.bmm(tab.H, X)                                                       # intra (D, C, BN)
    E = torch.bmm(tab.PE, X).reshape(D, 2, K, B, N)
    S_re, S_im = E[:, 0], E[:, 1]                                                 # (D, K, B, N)
    s = 1
    i = 0
    while s < N:
        qr, qi = tab.q_re[i], tab.q_im[i]
        a_re, a_im = S_re[..., :-s], S_im[..., :-s]
        n_re = S_re[..., s:] + qr * a_re - qi * a_im
        n_im = S_im[..., s:] + qr * a_im + qi * a_re
        S_re = torch.cat([S_re[..., :s], n_re], -1)
        S_im = torch.cat([S_im[..., :s], n_im], -1)
        s *= 2
        i += 1
    if N > 1:
        P = torch.cat([F.pad(S_re[..., :-1], (1, 0)), F.pad(S_im[..., :-1], (1, 0))], 1)   # S_{n-1}
        y = y + torch.bmm(tab.AT, P.reshape(D, 2 * K, B * N))
    y = y.reshape(D, C, B, N).permute(2, 0, 3, 1).reshape(B, D, Lp)
    if pad:
        y = y[..., pad:]
    st_re = S_re[..., N - 1].permute(2, 0, 1)
    st_im = S_im[..., N - 1].permute(2, 0, 1)
    return y, st_re, st_im


def decode_planes(poles, residues, device):
    """Contiguous fp32 (1, D, K) re/im planes for the decode step (measured: a strided pole slice
    caused a host restride round-trip in eager)."""
    g = lambda t, i: t[..., 0, i].detach().to(torch.float32).contiguous()[None].to(device)  # noqa: E731
    return g(poles, 0), g(poles, 1), g(residues, 0), g(residues, 1)


def step_iir_real(x1v, s_re, s_im, p_re, p_im, r_re, r_im):
    """One decode step of the modal IIR in real pairs. x1v: (B, D) any dtype; s_*: (B, D, K) fp32.
    Returns res = Re(sum_k r_k s'_k) (B, D) fp32 and the new state."""
    xf = x1v.to(torch.float32)[..., None]
    n_re = p_re * s_re - p_im * s_im + xf
    n_im = p_re * s_im + p_im * s_re
    res = (r_re * n_re - r_im * n_im).sum(-1)
    return res, n_re, n_im
