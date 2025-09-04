# Using vllm from Optimum Neuron for Qwen.  
Based on the docs here:  https://huggingface.co/docs/optimum-neuron/main/en/guides/vllm_plugin#optimum-neuron-plugin-for-vllm

Use Qwen/Qwen2.5-0.5B to work out of the box (because it is compiled and cached).

If you want to compile first:
```
optimum-cli export neuron -m Qwen/Qwen2.5-0.5B-Instruct --batch_size 1 --num_cores 2 --sequence_length 4096 QwenCompiled
```
Make sure your compilation matches your vllm parameters.
max_num_seqs = batch_size
max_model_len = sequence_length
tensor_parallel_size = num_cores


Deploy using the Hugging Face DLAMI using the instructions here:  https://repost.aws/articles/ARTxLi0wndTwquyl7frQYuKg/easy-deploy-of-an-inferentia-or-trainium-instance-on-ec2
```
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="Qwen/Qwen2.5-0.5B", # or use model="QwenCompiled" if you compiled locally.
          max_num_seqs=1,
          max_model_len=4096,
          tensor_parallel_size=2,
          device="neuron")

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```


# Example using Phi and vllm server

If you want to run the optimum-neuron vllm implementation, you can start on any machine with the Neuron SDK installed (2.24 or 2.25 for Optimum Neuron 0.3.0 -- Current version as of 9/3/25).

(You can confirm your SDK version with this script:  ```wget https://raw.githubusercontent.com/jimburtoft/Neuron-SDK-Detector/main/neuron_detector.py``` )

(Instructions on how to deploy an EC2 instance with the Hugging Face DLAMI here: https://repost.aws/articles/ARTxLi0wndTwquyl7frQYuKg/easy-deploy-of-an-inferentia-or-trainium-instance-on-ec2)

### Install optimum-neuron
```pip install optimum-neuron[neuronx,vllm]```

### Compile
Compile the model with your parameters:

```
optimum-cli export neuron --model microsoft/phi-4 --task text-generation \
--sequence_length 1000 --batch_size 10 --num_cores 2 ~/phi4Compiled
```
### Serve
Then start a server (make sure your max-num-seqs matches your batch_size from compilation, as well as your max-model-len matches sequence_length and tensor-parallel-size matches num_cores, and --model matches the output directory at the end of the optimum-cli command):

```
python3 -m vllm.entrypoints.openai.api_server --model ~/phi4Compiled \
 --max-model-len 1000 --tensor-parallel-size 2 --max-num-seqs=10
```


