---
inclusion: always
---

# AWS Neuron SDK Conversion Guidelines

## Virtual Environment

**CRITICAL**: All Neuron SDK commands must be run from the project-specific virtual environment:

```bash
source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate
```

Always activate this venv before:
- Running any Python scripts
- Installing packages
- Executing model compilation or inference
- Testing Neuron-optimized code

## Documentation Access

AWS Neuron SDK documentation is available via MCP server tools:

### Search Documentation
Use `mcp_awslabsaws_documentation_mcp_server_search_documentation` to find relevant pages:
```
search_phrase: "neuron sdk pytorch inference"
```

### Read Documentation
Use `mcp_awslabsaws_documentation_mcp_server_read_documentation` to fetch full content:
```
url: "https://docs.aws.amazon.com/..."
```

### Get Recommendations
Use `mcp_awslabsaws_documentation_mcp_server_recommend` to discover related pages:
```
url: "https://docs.aws.amazon.com/..."
```

**Important**: You must use the fetch/read tools to actually retrieve documentation content. The MCP server provides access but does not automatically include the docs.

## Neuron SDK Conversion Workflow

### 1. Assess Compatibility
- Check model architecture for Neuron support
- Identify unsupported operations
- Review input/output shapes and data types
- Verify PyTorch version compatibility (2.8+ for this venv)

### 2. Model Compilation
- Use `torch_neuronx.trace()` for inference models
- Specify input shapes explicitly
- Set compiler flags for optimization
- Save compiled model artifacts

### 3. Inference Optimization
- Load compiled models with `torch.jit.load()`
- Use Neuron-optimized data loaders
- Batch inputs when possible
- Monitor Neuron device utilization

### 4. Testing & Validation
- Compare outputs with CPU/GPU baseline
- Measure latency and throughput
- Check memory usage on Neuron devices
- Validate numerical accuracy

## Common Neuron SDK Patterns

### Model Tracing
```python
import torch
import torch_neuronx

# Trace model for Neuron
model_neuron = torch_neuronx.trace(
    model,
    example_inputs,
    compiler_workdir='./neuron_compile',
    compiler_args=['--verbose', 'info']
)

# Save compiled model
torch.jit.save(model_neuron, 'model_neuron.pt')
```

### Loading Compiled Models
```python
import torch

# Load on Neuron device
model_neuron = torch.jit.load('model_neuron.pt')
model_neuron.eval()

# Run inference
with torch.no_grad():
    output = model_neuron(input_tensor)
```

### Device Management
```python
import torch

# Check Neuron device availability
if torch.neuron.is_available():
    device = torch.device('neuron')
else:
    device = torch.device('cpu')
```

## Key Considerations

### Supported Operations
- Not all PyTorch operations are supported on Neuron
- Dynamic shapes may require special handling
- Some operations fall back to CPU automatically

### Performance Optimization
- Compile with appropriate batch sizes
- Use static shapes when possible
- Enable operator fusion via compiler flags
- Profile with Neuron tools (`neuron-top`, `neuron-monitor`)

### Debugging
- Use `NEURON_RT_LOG_LEVEL=INFO` for detailed logs
- Check compilation warnings carefully
- Validate model outputs against baseline
- Monitor device memory with `neuron-ls`

## Troubleshooting

### Compilation Errors
1. Check PyTorch version compatibility
2. Verify input tensor shapes and dtypes
3. Review unsupported operations in model
4. Consult Neuron SDK documentation for workarounds

### Runtime Issues
1. Ensure venv is activated
2. Check Neuron device availability
3. Verify compiled model matches input shapes
4. Monitor system resources

### Performance Issues
1. Profile with Neuron tools
2. Adjust batch sizes
3. Review compiler optimization flags
4. Check for CPU fallback operations

## Resources

- Always search Neuron SDK docs via MCP tools first
- Check for model-specific migration guides
- Review Neuron SDK release notes for version-specific changes
- Use Neuron GitHub samples for reference implementations
