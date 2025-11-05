"""
NKI 1D FFT - Hardware-accelerated 1D FFT for AWS Neuron

High-performance 1D Fast Fourier Transform implementation optimized for
AWS Trainium and Inferentia accelerators using the Neuron Kernel Interface (NKI).

Example:
    >>> from nki_fft1d import FFT1D_Flexible
    >>> import torch
    >>> 
    >>> fft = FFT1D_Flexible()
    >>> x = torch.randn(128, 512)
    >>> y = fft(x)  # Compute 1D FFT along rows
"""

from .fft1d_nki import FFT1D_Flexible

__version__ = "1.0.0"
__all__ = ["FFT1D_Flexible"]
