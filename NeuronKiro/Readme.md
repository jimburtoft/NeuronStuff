# Neuron SDK documentation
The Neuron SDK documentation is available on the aws-documentation mcp server.  However, it is only available for searching.  If you try to retrieve it you get a 403 error.  However, you can retrieve it with the fetch server.  

To use the MCP servers, you need to have uv installed on the system and accessible by Kiro.  I do this by using 

```
sudo pip3 install uv
```

on Ubuntu or

```
brew install uv
``` 

on the mac.  Make sure you use the config with the right path (and change your user directory for the cache!)

You should also add in the neuron-sdk.md to your steering docs.

# NKI steering docs

IF you are coding NKI kernels, make sure that you set up the MCP server and then add the NKI steering docs.  

# Pytorch steering docs

If you are tracing a model in pytorch, consider the pytorch steering doc.