"""Neuron Hyena filter for StripedHyena 1: replaces ``ParallelHyenaFilter.parallel_forward`` and
``sequential_forward`` with complex-free, FFT-free equivalents.

Upstream semantics reproduced (``stripedhyena/engine.py`` @ 7e13f61, ``prefill_style: fft``):
- short FIR: depthwise ``conv1d`` k=3, padding 2, bias added AFTER the conv; FIR state = last 2
  inputs (channels-first). Computed as explicit shifted multiply-adds in fp32 (``fir3``);
- column split per head: ``[x2 | x1 | v]`` blocks of 128 channels;
- long conv ``y = h * (x1 v)`` with the modal filter, fp32, cast to the activation dtype, then
  ``(y + x1v * D) * x2``;
- decode: ``step_fir`` + ``step_iir`` with a 2-state complex modal recurrence.

Neuron changes (measured: complex dtypes and torch.fft have no device path):
- long conv + final modal state: ``port/modal.modal_conv`` (chunked real matmuls + scan), which is
  exact for the modal filter (fp32 ~1e-7 vs an exact complex128 recurrence). The modal state after
  prefill is a by-product (no (B, D, 2, 2L) complex buffer, no FFT).
- decode IIR: real-pair recurrence on contiguous fp32 pole/residue planes; state stored as a
  ``(s_re, s_im)`` tuple of (B, D, 2) fp32 tensors in ``state_dict[layer_idx]``.
- FIR step: ``cat`` shift instead of ``roll`` + in-place slot write.
The filter tables are built once per layer from the fp32 poles/residues (float64 on host).
"""
import torch
import torch.nn.functional as F

from . import modal

CHUNK = 64          # modal_conv chunk length (measured tunes it)
MAX_LEN = 32768


def _tables(self, device):
    tab = getattr(self, "_modal_tab", None)
    if tab is None or tab.C != CHUNK:
        tab = modal.ModalTables(self.poles, self.residues, CHUNK, MAX_LEN, device)
        self._modal_tab = tab
    if getattr(self, "_planes", None) is None:
        self._planes = modal.decode_planes(self.poles, self.residues, device)
    return tab


def fir3(z, w, bias):
    """Causal depthwise FIR k=3 over the sequence axis of z (B, L, C), fp32 accumulation, as
    ``conv1d(z^T, w, padding=2)[..., :L] + bias``: y_t = w0 z_{t-2} + w1 z_{t-1} + w2 z_t + b.
    Explicit shifted multiply-adds instead of conv1d: eager conv1d at L=32K fails to compile on
    torch-neuronx 2.13 (NCC_ITIN902, measured)."""
    L = z.shape[1]
    zf = z.float()
    w = w[:, 0].float()                                   # (C, 3)
    zp = F.pad(zf, (0, 0, 2, 0))                          # (B, L+2, C)
    y = zp[:, 2:] * w[:, 2] + zp[:, 1:L + 1] * w[:, 1] + zp[:, :L] * w[:, 0] + bias.float()
    return y.to(z.dtype)


COLPERM = False   # set by port.opts "colperm": projection rows permuted to [x2 | x1 | v] contiguous


def split3(z, H, dh, dim):
    """Column split along ``dim`` (size 3D) into x2, x1, v (each D). Upstream layout: per head h the
    block [x2_h | x1_h | v_h] of 3*dh channels. With COLPERM the rows were permuted at load so the
    three are contiguous [x2 | x1 | v] -- plain slices."""
    D = H * dh
    if COLPERM:
        return z.narrow(dim, 0, D), z.narrow(dim, D, D), z.narrow(dim, 2 * D, D)
    zs = list(z.shape)
    zr = z.reshape(zs[:dim] + [H, 3 * dh] + zs[dim + 1:])
    out = [zr.narrow(dim + 1, i * dh, dh).reshape(zs[:dim] + [D] + zs[dim + 1:]) for i in range(3)]
    return out[0], out[1], out[2]


def colperm_index(H, dh):
    """new row r -> old row: new = [x2 (h, j) | x1 (h, j) | v (h, j)], old = h*3dh + part*dh + j."""
    import torch as _t
    h = _t.arange(H)[:, None] * 3 * dh
    j = _t.arange(dh)[None, :]
    return _t.cat([(h + p * dh + j).reshape(-1) for p in range(3)])


def _column_split_cf(z, H, dh):
    """z: (B, 3D, L) channels-first -> x2, x1, v each (B, D, L)."""
    if COLPERM:
        return split3(z, H, dh, 1)
    B, _, L = z.shape
    z = z.reshape(B, H, 3 * dh, L)
    return (z[:, :, :dh].reshape(B, H * dh, L), z[:, :, dh:2 * dh].reshape(B, H * dh, L),
            z[:, :, 2 * dh:].reshape(B, H * dh, L))


def parallel_forward(self, u, inference_params=None, padding_mask=None):
    assert padding_mask is None
    B, L, _ = u.shape
    uc = u.permute(0, 2, 1)                                                     # (B, 3D, L)
    k = self.short_filter_length
    assert k == 3
    z = fir3(u, self.short_filter_weight, self.short_filter_bias).permute(0, 2, 1)
    x2, x1, v = _column_split_cf(z, self.num_attention_heads, self.hidden_size_per_attention_head)
    x1v = x1 * v
    tab = _tables(self, u.device)
    y, s_re, s_im = modal.modal_conv(x1v.to(torch.float32), tab)
    y = y.to(x1v.dtype)
    y = (y + x1v * self.D.unsqueeze(-1)) * x2
    if inference_params is not None:
        st = uc[..., -(k - 1):]
        if L < k - 1:
            st = F.pad(st, (k - 1 - L, 0))
        inference_params.fir_state_dict[self.layer_idx] = st.contiguous()
        inference_params.state_dict[self.layer_idx] = (s_re.contiguous(), s_im.contiguous())
    return y.permute(0, 2, 1), inference_params


def sequential_forward(self, u, inference_params):
    if self.data_dtype is None:
        self.data_dtype = u.dtype
    if u.dim() > 2:
        u = u[:, -1]
    fir_state = inference_params.fir_state_dict[self.layer_idx]                 # (B, 3D, 2)
    s_re, s_im = inference_params.state_dict[self.layer_idx]
    w = self.short_filter_weight                                                # (3D, 1, 3)
    z = w[:, 0, -1][None] * u + torch.sum(fir_state * w[:, 0, :-1][None], dim=-1) + self.short_filter_bias
    fir_state = torch.cat([fir_state[..., 1:], u[..., None]], -1)
    H, dh = self.num_attention_heads, self.hidden_size_per_attention_head
    x2, x1, v = split3(z, H, dh, 1)
    x1v = x1 * v
    _tables(self, u.device)
    p_re, p_im, r_re, r_im = self._planes
    res, s_re, s_im = modal.step_iir_real(x1v, s_re, s_im, p_re, p_im, r_re, r_im)
    y = x2 * (res + self.D * x1v)
    inference_params.fir_state_dict[self.layer_idx] = fir_state
    inference_params.state_dict[self.layer_idx] = (s_re, s_im)
    return y.to(dtype=self.data_dtype)[:, None], inference_params


def install():
    from stripedhyena import model as M
    M.ParallelHyenaFilter.parallel_forward = parallel_forward
    M.ParallelHyenaFilter.sequential_forward = sequential_forward
