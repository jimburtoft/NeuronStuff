# NKI 1D FFT

High-performance 1D Fast Fourier Transform (FFT) implementation for AWS Neuron using the Neuron Kernel Interface (NKI).

## Overview

This package provides a flexible, hardware-accelerated 1D FFT implementation optimized for AWS Trainium and Inferentia accelerators. It combines tiling with masking (for arbitrary heights) and recursive radix-2 Cooley-Tukey algorithm (for large widths) to handle a wide range of input sizes efficiently.

## Features

- ✅ **Flexible Input Sizes**: Supports arbitrary heights and widths up to 4096
- ✅ **High Accuracy**: < 0.003% relative error
- ✅ **Hardware Accelerated**: Runs on AWS Neuron NeuronCores
- ✅ **Production Ready**: Validated on Trainium hardware
- ✅ **Easy to Use**: Drop-in replacement for PyTorch FFT operations

## Supported Input Sizes

### Height (Partition Dimension)
- **Any value**: 1, 32, 64, 100, 128, 256, 512, ...
- Automatically handled with tiling and masking

### Width (Free Dimension)
- **Powers of 2**: 128, 256, 512, 1024, 2048, 4096
- Uses recursive radix-2 Cooley-Tukey algorithm

### Validated Combinations
- ✅ 128×128 (base case)
- ✅ 128×512 (full tile, recursive width)
- ✅ 64×2048 (masked tile, large width)
- ✅ 100×4096 (arbitrary height, maximum width)

## Installation

### Prerequisites

- AWS Trainium or Inferentia instance
- Neuron SDK installed
- PyTorch with XLA support

### Setup

```bash
# Ensure Neuron SDK is installed
# See: https://awsdocs-neuron.readthedocs-hosted.com/

# Copy the package to your project
cp -r nki_fft1d /path/to/your/project/
```

## Quick Start

```python
import torch
from nki_fft1d.fft1d_nki import FFT1D_Flexible

# Create FFT instance
fft = FFT1D_Flexible()

# Create input tensor
x = torch.randn(128, 512, dtype=torch.float32)

# Compute 1D FFT along rows (axis=1)
y = fft(x)

# Result is a complex tensor of shape (128, 512)
print(y.shape)  # torch.Size([128, 512])
print(y.dtype)  # torch.complex64
```

## Usage Examples

### Basic Usage

```python
from nki_fft1d.fft1d_nki import FFT1D_Flexible

# Initialize
fft = FFT1D_Flexible()

# Compute FFT
x = torch.randn(128, 256)
y = fft(x)
```

### Replace PyTorch FFT

**Before (PyTorch CPU/CUDA):**
```python
import torch

x = torch.randn(128, 512)
y = torch.fft.fft(x, dim=1, norm='backward')
```

**After (NKI on Neuron):**
```python
import torch
from nki_fft1d.fft1d_nki import FFT1D_Flexible

fft = FFT1D_Flexible()
x = torch.randn(128, 512)
y = fft(x)  # Equivalent to torch.fft.fft(x, dim=1, norm='backward')
```

### Batch Processing with Varying Sizes

```python
from nki_fft1d.fft1d_nki import FFT1D_Flexible

fft = FFT1D_Flexible()

# Process different sizes with same instance
inputs = [
    torch.randn(64, 128),
    torch.randn(100, 512),
    torch.randn(128, 2048),
]

outputs = [fft(x) for x in inputs]
```

### Signal Processing Example

```python
import torch
import numpy as np
from nki_fft1d.fft1d_nki import FFT1D_Flexible

# Initialize FFT
fft = FFT1D_Flexible()

# Generate test signal (sine wave)
sample_rate = 1000  # Hz
duration = 0.128    # seconds
t = np.linspace(0, duration, 128)
frequency = 50      # Hz
signal = np.sin(2 * np.pi * frequency * t)

# Prepare for FFT (add batch dimension for multiple signals)
x = torch.from_numpy(signal).float().unsqueeze(0).repeat(64, 1)

# Compute FFT
y = fft(x)

# Get magnitude spectrum
magnitude = torch.abs(y)
print(f"Spectrum shape: {magnitude.shape}")  # (64, 128)
```

### Audio Processing Example

```python
from nki_fft1d.fft1d_nki import FFT1D_Flexible
import torch

# Initialize FFT
fft = FFT1D_Flexible()

# Simulate audio frames (e.g., 100 frames of 512 samples each)
audio_frames = torch.randn(100, 512)

# Compute FFT for all frames
spectrogram = fft(audio_frames)

# Get power spectrum
power_spectrum = torch.abs(spectrogram) ** 2
print(f"Power spectrum shape: {power_spectrum.shape}")  # (100, 512)
```

## API Reference

### FFT1D_Flexible

Main class for computing 1D FFT.

#### Constructor

```python
FFT1D_Flexible(dtype=torch.float32)
```

**Parameters:**
- `dtype` (torch.dtype, optional): Data type for computation. Default: `torch.float32`

#### forward() / __call__()

```python
def forward(X: torch.Tensor) -> torch.Tensor
```

Compute 1D FFT along rows (axis=1).

**Parameters:**
- `X` (torch.Tensor): Input tensor of shape `[H, W]`
  - `H`: Height (any value)
  - `W`: Width (must be power of 2, >= 128, <= 4096)
  - Can be real or complex

**Returns:**
- `torch.Tensor`: Complex output tensor of shape `[H, W]`

**Example:**
```python
fft = FFT1D_Flexible()
x = torch.randn(128, 512)
y = fft(x)  # or fft.forward(x)
```

## Comparison with PyTorch FFT

### Equivalence

```python
# NKI FFT
from nki_fft1d.fft1d_nki import FFT1D_Flexible
fft = FFT1D_Flexible()
y_nki = fft(x)

# PyTorch FFT (equivalent)
y_torch = torch.fft.fft(x, dim=1, norm='backward')

# Results are nearly identical (< 0.003% error)
```

### When to Use NKI FFT

**Use NKI FFT when:**
- ✅ Running on AWS Trainium/Inferentia
- ✅ Need hardware acceleration on Neuron
- ✅ Processing multiple FFTs in parallel
- ✅ Want to leverage NeuronCore compute engines

**Use PyTorch FFT when:**
- ✅ Running on CPU/CUDA
- ✅ Need arbitrary input sizes (non-power-of-2)
- ✅ Need different normalization modes
- ✅ Need multi-dimensional FFT (2D, 3D)

## Performance

### Accuracy
- **Relative Error**: < 0.003% compared to NumPy FFT
- **Validated**: All test cases pass on Trainium hardware

### Throughput
- **Parallel Processing**: Processes multiple rows simultaneously
- **Hardware Accelerated**: Uses NeuronCore Tensor Engine
- **Efficient Tiling**: Minimizes memory transfers

## Testing

Run the test suite to validate the implementation:

```bash
python test_fft1d.py
```

Expected output:
```
======================================================================
NKI 1D FFT Test Suite
======================================================================

Testing Base case - 128×128 (128×128):
  Max absolute error: 9.821401e-04
  Max relative error: 0.002674%
  ✅ PASS

Testing Full tile - 128×512 (128×512):
  Max absolute error: 3.894919e-03
  Max relative error: 0.002353%
  ✅ PASS

...

🎉 All tests passed!
```

## Limitations

### Current Limitations
1. **Width must be power of 2**: 128, 256, 512, 1024, 2048, 4096
2. **Maximum width**: 4096 (hardware constraint)
3. **1D only**: Computes FFT along rows (axis=1)
4. **Normalization**: Only 'backward' mode (no normalization on forward FFT)

### Workarounds

**Non-power-of-2 widths:**
```python
# Pad to next power of 2
import math
W = 1000  # Original width
W_padded = 2 ** math.ceil(math.log2(W))
x_padded = torch.nn.functional.pad(x, (0, W_padded - W))
y_padded = fft(x_padded)
y = y_padded[:, :W]  # Trim to original size
```

**Different normalization:**
```python
# For 'ortho' normalization
y = fft(x) / math.sqrt(W)

# For 'forward' normalization
y = fft(x) / W
```

## Technical Details

### Algorithm
- **Base case (W=128)**: Direct DFT matrix multiplication
- **Recursive case (W>128)**: Radix-2 Cooley-Tukey algorithm
- **Height handling**: Tiling with masking for arbitrary sizes

### Hardware Utilization
- **Tensor Engine**: Matrix multiplications (DFT matrix)
- **Scalar Engine**: Twiddle factor multiplication
- **Vector Engine**: Element-wise operations
- **SBUF**: On-chip memory for tiles (~24MB)

### Memory Layout
- **Partition Dimension (P)**: Height (up to 128 per tile)
- **Free Dimension (F)**: Width (128-4096)
- **Tiling**: Automatic for heights > 128

## Troubleshooting

### Common Issues

**1. Width not power of 2**
```
AssertionError: Width must be power of 2 >= 128, got 1000
```
**Solution**: Pad input to next power of 2 (see Workarounds above)

**2. Width too large**
```
AssertionError: Width must be power of 2 >= 128, got 8192
```
**Solution**: Maximum supported width is 4096. Split into smaller chunks if needed.

**3. XLA device error**
```
RuntimeError: XLA device not found
```
**Solution**: Ensure running on Neuron instance with XLA support

## License

This implementation is provided as-is for use with AWS Neuron SDK.

## Support

For issues related to:
- **NKI/Neuron SDK**: See [AWS Neuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)
- **This implementation**: Check test suite and examples above

## Acknowledgments

Built using AWS Neuron Kernel Interface (NKI) and optimized for AWS Trainium/Inferentia accelerators.
