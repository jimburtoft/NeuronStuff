# Qwen3-ASR + DFlash speculative decoding on Trainium2 with vllm-neuron (Neuron SDK 2.33)

Serve [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) (and `Qwen/Qwen3-ASR-0.6B`)
with vLLM on a trn2.3xlarge, with or without a [DFlash](https://arxiv.org/abs/2602.06036)
block-diffusion draft model. The environment is the public Neuron SDK 2.33 DLAMI plus a public
vllm-neuron fork branch that adds the Qwen3-ASR performance fixes and the DFlash target hooks on top
of the `0.29.0.1.2.0` release (vLLM 0.29.0).

- **Target:** `Qwen/Qwen3-ASR-1.7B`, BF16.
- **Draft (optional):** the public [`jburtoft/Qwen3-ASR-1.7B-DFlash`](https://huggingface.co/jburtoft/Qwen3-ASR-1.7B-DFlash)
  (Apache-2.0, block size 16). Its training code is at
  [jimburtoft/qwen3-asr-dflash](https://github.com/jimburtoft/qwen3-asr-dflash).
- **Endpoint:** the standard OpenAI-compatible `/v1/audio/transcriptions`.

## Install and run

The full walkthrough — cloning the fork on top of the public SDK 2.33 / vLLM 0.29 release, the
`setup_champion.sh` step, the throughput and single-stream `vllm serve` commands, FP8 KV, the
async reorder / block-KV-write options, long audio, other languages, `stream=true`, Qwen3-ASR-0.6B,
and the DFlash speculative-decoding section — is in the fork's own README:

**https://github.com/jimburtoft/vllm-neuron/blob/qwen3-asr/0.29-champion-spec/QWEN3_ASR_FIXES.md**

Clone the two public forks and set up the stack:

```bash
git clone -b qwen3-asr/0.29-champion-spec https://github.com/jimburtoft/vllm-neuron.git
git clone -b qwen3-asr/2.33-champion https://github.com/jimburtoft/nki-library.git
source vllm-neuron/qwen3_asr/setup_champion.sh $PWD/nki-library
```

Run inside the public SDK 2.33 DLAMI vLLM venv
`/opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0` (or the 2.33 vLLM container).

To enable DFlash speculative decoding, add one flag to `vllm serve`:

```bash
  --speculative-config '{"method":"dflash","model":"jburtoft/Qwen3-ASR-1.7B-DFlash","num_speculative_tokens":15}'
```

Set `num_speculative_tokens` to the draft's `block_size - 1` (15 for the block-16 public draft).
Speculation is off unless `--speculative-config` is passed, and it lowers single-stream / low-concurrency
latency rather than peak throughput (the 16-token verify step costs about 2x a decode step).
See the fork README for the exact commands and measured acceptance / latency.
