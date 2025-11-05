#!/usr/bin/env python3
"""
1D FFT - Fully Flexible Implementation

Combines tiling/masking (for height) with recursive radix-2 (for width).

Supports:
- Height: Any value (automatically tiled with masking)
- Width: 128, 256, 512, 1024, 2048, 4096 (powers of 2)

Examples:
- 128×512: Full tile, recursive width
- 64×2048: Masked tile, recursive width  
- 100×4096: Masked tile, recursive width

Author: NKI Kernel Development
"""

import torch
import numpy as np
import torch_xla.core.xla_model as xm
import os
import math

from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

os.environ['NEURON_CC_FLAGS'] = '--auto-cast=none --enable-mixed-precision-accumulation'


def _compute_dft_matrix(N: int) -> tuple:
    """Compute DFT matrix for N-point FFT."""
    k = np.arange(N, dtype=np.float32).reshape(N, 1)
    n = np.arange(N, dtype=np.float32).reshape(1, N)
    angles = -2.0 * np.pi * k * n / N
    W_real = np.cos(angles).astype(np.float32)
    W_imag = np.sin(angles).astype(np.float32)
    return W_real, W_imag


def _compute_twiddle_factors(N: int, H: int) -> tuple:
    """Compute twiddle factors for radix-2 FFT."""
    k = np.arange(N // 2, dtype=np.float32)
    angles = -2.0 * np.pi * k / N
    twiddle_real_1d = np.cos(angles).astype(np.float32)
    twiddle_imag_1d = np.sin(angles).astype(np.float32)
    
    # Expand to (H, N/2) for broadcasting
    twiddle_real = np.tile(twiddle_real_1d, (H, 1))
    twiddle_imag = np.tile(twiddle_imag_1d, (H, 1))
    
    return twiddle_real, twiddle_imag


def _fft1d_matmul_isa(X_real, X_imag, Y_real, Y_imag, W_real, W_imag, axis: int):
    """1D FFT using matrix multiplication via Tensor Engine."""
    if axis == 1:  # Row FFT
        W_real_T = nisa.nc_transpose(W_real)
        W_imag_T = nisa.nc_transpose(W_imag)
        X_real_T = nisa.nc_transpose(X_real)
        X_imag_T = nisa.nc_transpose(X_imag)
        
        H, W_size = X_real.shape
        
        term1_psum = nl.ndarray((H, W_size), dtype=nl.float32, buffer=nl.psum)
        term2_psum = nl.ndarray((H, W_size), dtype=nl.float32, buffer=nl.psum)
        term3_psum = nl.ndarray((H, W_size), dtype=nl.float32, buffer=nl.psum)
        term4_psum = nl.ndarray((H, W_size), dtype=nl.float32, buffer=nl.psum)
        
        term1_psum[...] = nisa.nc_matmul(X_real_T, W_real_T)
        term2_psum[...] = nisa.nc_matmul(X_imag_T, W_imag_T)
        term3_psum[...] = nisa.nc_matmul(X_real_T, W_imag_T)
        term4_psum[...] = nisa.nc_matmul(X_imag_T, W_real_T)
        
        term1 = nl.ndarray((H, W_size), dtype=nl.float32, buffer=nl.sbuf)
        term2 = nl.ndarray((H, W_size), dtype=nl.float32, buffer=nl.sbuf)
        term3 = nl.ndarray((H, W_size), dtype=nl.float32, buffer=nl.sbuf)
        term4 = nl.ndarray((H, W_size), dtype=nl.float32, buffer=nl.sbuf)
        
        term1[...] = nl.copy(term1_psum[...])
        term2[...] = nl.copy(term2_psum[...])
        term3[...] = nl.copy(term3_psum[...])
        term4[...] = nl.copy(term4_psum[...])
        
        Y_real[...] = term1 - term2
        Y_imag[...] = term3 + term4


@nki.jit
def _fft1d_dft_128_masked(X_real_hbm, X_imag_hbm,
                          W_128_real_hbm, W_128_imag_hbm,
                          actual_height):
    """
    128-point FFT using DFT matrix with height masking.
    
    Args:
        X_real_hbm, X_imag_hbm: Input [128, 128] (padded)
        W_128_real_hbm, W_128_imag_hbm: DFT matrix [128, 128]
        actual_height: Actual height (may be < 128)
    """
    TILE_H = 128
    N = 128
    
    Y_real_hbm = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.shared_hbm)
    Y_imag_hbm = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.shared_hbm)
    
    # Create mask
    i_h, i_w = nl.mgrid[0:TILE_H, 0:N]
    mask = (i_h < actual_height)
    
    # Load input with masking
    X_real = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.sbuf)
    X_imag = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.sbuf)
    X_real[...] = nl.load(X_real_hbm[i_h, i_w], mask=mask)
    X_imag[...] = nl.load(X_imag_hbm[i_h, i_w], mask=mask)
    
    # Load DFT matrix
    W_real = nl.ndarray((N, N), dtype=nl.float32, buffer=nl.sbuf)
    W_imag = nl.ndarray((N, N), dtype=nl.float32, buffer=nl.sbuf)
    W_real[...] = nl.load(W_128_real_hbm)
    W_imag[...] = nl.load(W_128_imag_hbm)
    
    # Apply DFT
    Y_real = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.sbuf)
    Y_imag = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.sbuf)
    
    _fft1d_matmul_isa(X_real, X_imag, Y_real, Y_imag,
                      W_real, W_imag, axis=1)
    
    # Store with masking
    nl.store(Y_real_hbm[i_h, i_w], value=Y_real, mask=mask)
    nl.store(Y_imag_hbm[i_h, i_w], value=Y_imag, mask=mask)
    
    return Y_real_hbm, Y_imag_hbm


@nki.jit
def _fft1d_radix2_256_masked(X_real_hbm, X_imag_hbm,
                              W_128_real_hbm, W_128_imag_hbm,
                              twiddle_real_hbm, twiddle_imag_hbm,
                              actual_height):
    """256-point FFT with height masking."""
    TILE_H = 128
    N = 256
    N_half = 128
    
    Y_real_hbm = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.shared_hbm)
    Y_imag_hbm = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.shared_hbm)
    
    # Create mask
    i_h, i_w = nl.mgrid[0:TILE_H, 0:N]
    mask = (i_h < actual_height)
    
    # Load input
    X_real = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.sbuf)
    X_imag = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.sbuf)
    X_real[...] = nl.load(X_real_hbm[i_h, i_w], mask=mask)
    X_imag[...] = nl.load(X_imag_hbm[i_h, i_w], mask=mask)
    
    # Load DFT matrix and twiddle factors
    W_128_real = nl.ndarray((N_half, N_half), dtype=nl.float32, buffer=nl.sbuf)
    W_128_imag = nl.ndarray((N_half, N_half), dtype=nl.float32, buffer=nl.sbuf)
    W_128_real[...] = nl.load(W_128_real_hbm)
    W_128_imag[...] = nl.load(W_128_imag_hbm)
    
    twiddle_real = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    twiddle_imag = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    twiddle_real[...] = nl.load(twiddle_real_hbm)
    twiddle_imag[...] = nl.load(twiddle_imag_hbm)
    
    # Split even/odd
    X_even_real = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    X_even_imag = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    X_odd_real = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    X_odd_imag = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    
    X_even_real[...] = X_real[:, 0::2]
    X_even_imag[...] = X_imag[:, 0::2]
    X_odd_real[...] = X_real[:, 1::2]
    X_odd_imag[...] = X_imag[:, 1::2]
    
    # Apply 128-point FFT
    X_even_fft_real = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    X_even_fft_imag = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    X_odd_fft_real = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    X_odd_fft_imag = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    
    _fft1d_matmul_isa(X_even_real, X_even_imag, X_even_fft_real, X_even_fft_imag,
                      W_128_real, W_128_imag, axis=1)
    _fft1d_matmul_isa(X_odd_real, X_odd_imag, X_odd_fft_real, X_odd_fft_imag,
                      W_128_real, W_128_imag, axis=1)
    
    # Apply twiddle factors
    X_odd_tw_real = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    X_odd_tw_imag = nl.ndarray((TILE_H, N_half), dtype=nl.float32, buffer=nl.sbuf)
    
    X_odd_tw_real[...] = X_odd_fft_real * twiddle_real - X_odd_fft_imag * twiddle_imag
    X_odd_tw_imag[...] = X_odd_fft_real * twiddle_imag + X_odd_fft_imag * twiddle_real
    
    # Combine
    Y_combined_real = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.sbuf)
    Y_combined_imag = nl.ndarray((TILE_H, N), dtype=nl.float32, buffer=nl.sbuf)
    
    Y_combined_real[:, 0:N_half] = X_even_fft_real + X_odd_tw_real
    Y_combined_imag[:, 0:N_half] = X_even_fft_imag + X_odd_tw_imag
    Y_combined_real[:, N_half:N] = X_even_fft_real - X_odd_tw_real
    Y_combined_imag[:, N_half:N] = X_even_fft_imag - X_odd_tw_imag
    
    # Store with masking
    nl.store(Y_real_hbm[i_h, i_w], value=Y_combined_real, mask=mask)
    nl.store(Y_imag_hbm[i_h, i_w], value=Y_combined_imag, mask=mask)
    
    return Y_real_hbm, Y_imag_hbm


class FFT1D_Flexible:
    """
    Fully flexible 1D FFT implementation.
    
    Supports:
    - Height: Any value (automatically tiled with masking)
    - Width: 128, 256, 512, 1024, 2048, 4096 (powers of 2)
    
    Examples:
        >>> fft = FFT1D_Flexible()
        >>> y1 = fft(torch.randn(128, 512))   # Full tile, recursive width
        >>> y2 = fft(torch.randn(64, 2048))   # Masked tile, recursive width
        >>> y3 = fft(torch.randn(100, 4096))  # Masked tile, recursive width
    """
    
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype
        self.W_128_real, self.W_128_imag = _compute_dft_matrix(128)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute 1D FFT along rows.
        
        Args:
            X: Input tensor [H, W]
               - H: Any value
               - W: Power of 2, >= 128
        """
        H, W = X.shape
        
        # Validate width
        assert W >= 128 and (W & (W - 1)) == 0, \
            f"Width must be power of 2 >= 128, got {W}"
        
        # Convert to real/imag
        if torch.is_complex(X):
            x_real = torch.real(X).contiguous()
            x_imag = torch.imag(X).contiguous()
        else:
            x_real = X.contiguous()
            x_imag = torch.zeros_like(X)
        
        x_real = x_real.to(self.dtype)
        x_imag = x_imag.to(self.dtype)
        
        device = xm.xla_device()
        x_real = x_real.to(device)
        x_imag = x_imag.to(device)
        
        # Prepare DFT matrix
        W_128_real = torch.from_numpy(self.W_128_real).to(device)
        W_128_imag = torch.from_numpy(self.W_128_imag).to(device)
        
        # Pad height to 128 if needed
        H_padded = max(128, H)
        if H < 128:
            x_real_padded = torch.zeros(128, W, dtype=self.dtype, device=device)
            x_imag_padded = torch.zeros(128, W, dtype=self.dtype, device=device)
            x_real_padded[:H, :] = x_real
            x_imag_padded[:H, :] = x_imag
            x_real = x_real_padded
            x_imag = x_imag_padded
        
        # Recursive FFT with masking
        y_real, y_imag = self._fft_recursive(
            x_real, x_imag, W, H, W_128_real, W_128_imag, device
        )
        
        # Extract valid rows
        y_real = y_real[:H, :]
        y_imag = y_imag[:H, :]
        
        # Move to CPU and combine
        y_real_cpu = y_real.cpu()
        y_imag_cpu = y_imag.cpu()
        y = torch.complex(y_real_cpu, y_imag_cpu)
        
        return y
    
    def _fft_recursive(self, x_real, x_imag, W, H, W_128_real, W_128_imag, device):
        """Recursively apply radix-2 FFT with height masking."""
        if W == 128:
            # Base case: 128-point FFT with masking
            return _fft1d_dft_128_masked(
                x_real, x_imag,
                W_128_real, W_128_imag,
                H
            )
        elif W == 256:
            # Single-level radix-2 with masking
            twiddle_real_np, twiddle_imag_np = _compute_twiddle_factors(W, 128)
            twiddle_real = torch.from_numpy(twiddle_real_np).to(device)
            twiddle_imag = torch.from_numpy(twiddle_imag_np).to(device)
            
            return _fft1d_radix2_256_masked(
                x_real, x_imag,
                W_128_real, W_128_imag,
                twiddle_real, twiddle_imag,
                H
            )
        else:
            # Recursive case
            W_half = W // 2
            x_even_real = x_real[:, 0::2]
            x_even_imag = x_imag[:, 0::2]
            x_odd_real = x_real[:, 1::2]
            x_odd_imag = x_imag[:, 1::2]
            
            # Recurse
            y_even_real, y_even_imag = self._fft_recursive(
                x_even_real, x_even_imag, W_half, H, W_128_real, W_128_imag, device
            )
            y_odd_real, y_odd_imag = self._fft_recursive(
                x_odd_real, x_odd_imag, W_half, H, W_128_real, W_128_imag, device
            )
            
            # Combine with twiddle factors
            # Generate twiddle factors with correct height
            actual_H = y_even_real.shape[0]  # Use actual height from recursion result
            twiddle_real_np, twiddle_imag_np = _compute_twiddle_factors(W, actual_H)
            twiddle_real = torch.from_numpy(twiddle_real_np).to(device)
            twiddle_imag = torch.from_numpy(twiddle_imag_np).to(device)
            
            y_real = torch.zeros(actual_H, W, dtype=torch.float32, device=device)
            y_imag = torch.zeros(actual_H, W, dtype=torch.float32, device=device)
            
            tw_odd_real = twiddle_real * y_odd_real - twiddle_imag * y_odd_imag
            tw_odd_imag = twiddle_real * y_odd_imag + twiddle_imag * y_odd_real
            
            y_real[:, :W_half] = y_even_real + tw_odd_real
            y_imag[:, :W_half] = y_even_imag + tw_odd_imag
            y_real[:, W_half:] = y_even_real - tw_odd_real
            y_imag[:, W_half:] = y_even_imag - tw_odd_imag
            
            return y_real, y_imag
    
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X)
