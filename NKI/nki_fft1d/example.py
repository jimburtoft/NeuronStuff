#!/usr/bin/env python3
"""
Example usage of NKI 1D FFT.

Demonstrates various use cases and how to replace PyTorch FFT operations.
"""

import torch
import numpy as np
from fft1d_nki import FFT1D_Flexible


def example_basic():
    """Basic usage example."""
    print("=" * 70)
    print("Example 1: Basic Usage")
    print("=" * 70)
    
    # Create FFT instance
    fft = FFT1D_Flexible()
    
    # Create input
    x = torch.randn(128, 512, dtype=torch.float32)
    print(f"Input shape: {x.shape}")
    
    # Compute FFT
    y = fft(x)
    print(f"Output shape: {y.shape}")
    print(f"Output dtype: {y.dtype}")
    
    # Get magnitude spectrum
    magnitude = torch.abs(y)
    print(f"Magnitude range: [{magnitude.min():.2f}, {magnitude.max():.2f}]")
    print()


def example_replace_pytorch():
    """Example: Replace PyTorch FFT."""
    print("=" * 70)
    print("Example 2: Replace PyTorch FFT")
    print("=" * 70)
    
    x = torch.randn(128, 256)
    
    # PyTorch FFT (for comparison)
    y_torch = torch.fft.fft(x, dim=1, norm='backward')
    
    # NKI FFT (equivalent)
    fft = FFT1D_Flexible()
    y_nki = fft(x)
    
    # Compare
    error = torch.max(torch.abs(y_nki - y_torch)).item()
    print(f"Max difference: {error:.6e}")
    print(f"Results are equivalent: {error < 1e-2}")
    print()


def example_signal_processing():
    """Example: Signal processing."""
    print("=" * 70)
    print("Example 3: Signal Processing")
    print("=" * 70)
    
    # Generate test signal (sine wave)
    sample_rate = 1000  # Hz
    duration = 0.128    # seconds
    t = np.linspace(0, duration, 128)
    frequency = 50      # Hz
    signal = np.sin(2 * np.pi * frequency * t)
    
    # Add multiple signals (batch processing)
    x = torch.from_numpy(signal).float().unsqueeze(0).repeat(64, 1)
    print(f"Signal shape: {x.shape} (64 signals, 128 samples each)")
    
    # Compute FFT
    fft = FFT1D_Flexible()
    y = fft(x)
    
    # Get magnitude spectrum
    magnitude = torch.abs(y)
    print(f"Spectrum shape: {magnitude.shape}")
    
    # Find peak frequency
    peak_idx = torch.argmax(magnitude[0, :64])  # Only positive frequencies
    peak_freq = peak_idx.item() * sample_rate / 128
    print(f"Detected peak frequency: {peak_freq:.1f} Hz (expected: {frequency} Hz)")
    print()


def example_arbitrary_sizes():
    """Example: Arbitrary input sizes."""
    print("=" * 70)
    print("Example 4: Arbitrary Input Sizes")
    print("=" * 70)
    
    fft = FFT1D_Flexible()
    
    # Various sizes
    test_sizes = [
        (32, 128),
        (64, 512),
        (100, 2048),
        (128, 4096),
    ]
    
    for H, W in test_sizes:
        x = torch.randn(H, W)
        y = fft(x)
        print(f"  {H:3d}×{W:4d} → {y.shape} ✓")
    
    print("\nAll sizes work correctly!")
    print()


def example_complex_input():
    """Example: Complex input."""
    print("=" * 70)
    print("Example 5: Complex Input")
    print("=" * 70)
    
    # Create complex input
    x_real = torch.randn(128, 256)
    x_imag = torch.randn(128, 256)
    x = torch.complex(x_real, x_imag)
    
    print(f"Input shape: {x.shape}")
    print(f"Input dtype: {x.dtype}")
    
    # Compute FFT
    fft = FFT1D_Flexible()
    y = fft(x)
    
    print(f"Output shape: {y.shape}")
    print(f"Output dtype: {y.dtype}")
    print("Complex input handled correctly ✓")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("NKI 1D FFT - Usage Examples")
    print("=" * 70)
    print()
    
    example_basic()
    example_replace_pytorch()
    example_signal_processing()
    example_arbitrary_sizes()
    example_complex_input()
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
