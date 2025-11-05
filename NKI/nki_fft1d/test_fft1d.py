#!/usr/bin/env python3
"""
Test suite for NKI 1D FFT implementation.

Validates correctness against NumPy FFT across various input sizes.
"""

import torch
import numpy as np
from fft1d_nki import FFT1D_Flexible


def test_fft1d():
    """Test 1D FFT with various input sizes."""
    print("=" * 70)
    print("NKI 1D FFT Test Suite")
    print("=" * 70)
    
    fft = FFT1D_Flexible()
    
    test_cases = [
        # (height, width, description)
        (128, 128, "Base case - 128×128"),
        (128, 256, "Full tile - 128×256"),
        (128, 512, "Full tile - 128×512"),
        (64, 128, "Masked tile - 64×128"),
        (64, 2048, "Masked tile - 64×2048"),
        (100, 4096, "Arbitrary - 100×4096"),
    ]
    
    results = []
    
    for H, W, desc in test_cases:
        print(f"\nTesting {desc} ({H}×{W}):")
        
        # Create random input
        torch.manual_seed(42)
        x = torch.randn(H, W, dtype=torch.float32)
        
        # Compute reference using NumPy
        x_np = x.numpy()
        y_ref_np = np.fft.fft(x_np, axis=1)
        y_ref = torch.from_numpy(y_ref_np)
        
        try:
            # Compute using NKI
            y_nki = fft(x)
            
            # Calculate error
            error = torch.max(torch.abs(y_nki - y_ref)).item()
            rel_error = error / (torch.max(torch.abs(y_ref)).item() + 1e-10)
            
            # Check if passed
            passed = rel_error < 0.01  # 1% tolerance
            status = "✅ PASS" if passed else "❌ FAIL"
            
            print(f"  Max absolute error: {error:.6e}")
            print(f"  Max relative error: {rel_error:.6%}")
            print(f"  {status}")
            
            results.append((H, W, desc, passed, rel_error))
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append((H, W, desc, False, float('inf')))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed_count = sum(1 for _, _, _, passed, _ in results if passed)
    total_count = len(results)
    
    for H, W, desc, passed, rel_error in results:
        status = "✅" if passed else "❌"
        if rel_error != float('inf'):
            print(f"{status} {desc}: {rel_error:.4%} error")
        else:
            print(f"{status} {desc}: ERROR")
    
    print(f"\nPassed: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed")
        return False


if __name__ == "__main__":
    success = test_fft1d()
    exit(0 if success else 1)
