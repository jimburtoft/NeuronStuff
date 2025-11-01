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


## Resources

- Always search Neuron SDK docs via MCP tools first
- Check for model-specific migration guides
- Review Neuron SDK release notes for version-specific changes
- Use Neuron GitHub samples for reference implementations
