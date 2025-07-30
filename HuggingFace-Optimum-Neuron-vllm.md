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
