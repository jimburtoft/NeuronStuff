"""
NKI template file

"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor
import math

@nki.jit
def nki_kernel(in_tensor, input1, default_input=0, unlikely_input=None):
    """NKI Kernel layout
    
    Args:
        

    """
    
    # Get input/output dimensions
    sz_cin, sz_hin, sz_win = in_tensor.shape
    sz_hout = sz_hin
    sz_wout = sz_win #maybe you need to calculate these?

    # Assert any of your assumptions about size
    # to make sure our code runs without any manipulation, we are asserting that input sizes=output sizes
    assert sz_hout == sz_hin
    assert sz_wout == sz_win
    
        
    # Create output tensor the size of our expected output in hbm
    out_tensor = nl.ndarray((sz_cin, sz_hout, sz_wout), dtype=in_tensor.dtype,
                           buffer=nl.shared_hbm)

    # for more complicated kernels, you would generate tiles with an affine_range.  

    in_tile = nl.load(in_tensor)

    #Do your maniplation here
    in_tile = in_tile * input1

    nl.store(out_tensor, value=in_tile)
   
    return out_tensor

def benchmark_nki_kernel():
    # Set parameters here so we can reuse them in the kernel and cpu functions
    channels = 1
    height = 224
    width = 224
    input1 = 0
    # default_input = 0 # Not set because the function assums them
    # unlikely_input = None
    
    # Create example input
    x_np = np.random.randn(channels, height, width).astype(np.int8)
    x_torch = torch.from_numpy(x_np).unsqueeze(0)  # For comparison with python
    
    print("Running inference...")
    
    """
        For info on nki.bencmark or to install see:  
        https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.benchmark.html 
        
        
        View the nttf file in Neuron Profiler
        ###Add link here
    """
    benchmark_func = nki.benchmark(
        nki_kernel,
        warmup=10,
        iters=100,
        save_neff_name='file.neff',
        save_trace_name='nki_kernel.ntff'
    )
    output_nki = benchmark_func(x_np, input1)
    #reassigning because the benchmark output doesn't deliver the kernel output.  It includes additional metrics.
    output_nki = nki_kernel(x_np, input1)


    # Print benchmark results
    # Thank you Amazon Q!
    metrics = benchmark_func.benchmark_result.nc_latency
    print("\nBenchmark Results:")
    print(f"P50 Latency: {metrics.get_latency_percentile(50):>8.2f} us")
    print(f"P90 Latency: {metrics.get_latency_percentile(90):>8.2f} us")
    print(f"P99 Latency: {metrics.get_latency_percentile(99):>8.2f} us")

    # Verify shapes
    print(f"\nShape verification:")
    print(f"NKI Input shape: {x_np.shape}")
    print(f"NKI Output shape: {output_nki.shape}")
    
    # Any other verifications you want to display?  Calculations of output shapes or values?
    
    
    # Verify against CPU reference
    print("\nVerifying against CPU reference...")
    with torch.no_grad():
        ref_output = x_torch * input1
    
    ref_output = ref_output.squeeze(0).numpy()
    print(f"Torch Input shape: {x_torch.shape}")
    print(f"Torch Output shape: {ref_output.shape}")
    assert output_nki.shape == ref_output.shape

    max_diff = np.max(np.abs(output_nki - ref_output))
    print(f"Maximum difference from reference: {max_diff}")

if __name__ == "__main__":
    benchmark_nki_kernel()
