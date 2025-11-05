# NKI 1D FFT Package

## Package Contents

```
nki_fft1d/
├── __init__.py          # Package initialization
├── fft1d_nki.py         # Main FFT implementation
├── test_fft1d.py        # Test suite
├── README.md            # Complete documentation
└── PACKAGE_INFO.md      # This file
```

## Quick Start

```python
from nki_fft1d import FFT1D_Flexible
import torch

# Create FFT instance
fft = FFT1D_Flexible()

# Compute FFT
x = torch.randn(128, 512)
y = fft(x)
```

## Key Features

- ✅ Flexible input sizes (any height, widths up to 4096)
- ✅ High accuracy (< 0.003% error)
- ✅ Hardware accelerated on AWS Neuron
- ✅ Drop-in replacement for `torch.fft.fft(x, dim=1)`

## Installation

Simply copy the `nki_fft1d` directory to your project:

```bash
cp -r nki_fft1d /path/to/your/project/
```

Then import and use:

```python
from nki_fft1d import FFT1D_Flexible
```

## Testing

```bash
cd nki_fft1d
python test_fft1d.py
```

## Documentation

See `README.md` for complete documentation including:
- Detailed usage examples
- API reference
- PyTorch FFT comparison
- Performance characteristics
- Troubleshooting guide

## Requirements

- AWS Trainium or Inferentia instance
- Neuron SDK
- PyTorch with XLA support

## Version

1.0.0 - Initial release
