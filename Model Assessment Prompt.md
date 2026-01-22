For a quick, high level overview of what effort is required, try this prompt in your favorite LLM:

```
I need you to assess the effort and tasks required to run this model on Trainium:
<link to github or hugging face>

Find similar model definitions here to see how they were implemented on Trainium and decide what you could reuse:
https://github.com/aws-neuron/neuronx-distributed-inference/tree/main/src/neuronx_distributed_inference

For specific kernels, see what is available in the NKI library for Trainium and what could be reused and/or adapted:  https://github.com/aws-neuron/nki-library/tree/main/src/nkilib_standalone/nkilib/core
```
