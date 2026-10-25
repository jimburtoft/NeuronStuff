# Voxtral + DFlash speculative decoding on Trainium2 with vllm-neuron (Neuron SDK 2.33)

Serve [`mistralai/Voxtral-Mini-3B-2507`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) with
vLLM on a trn2.3xlarge, with or without a [DFlash](https://arxiv.org/abs/2602.06036)
block-diffusion draft model. The environment is the public Neuron SDK 2.33 DLAMI plus the Voxtral
support from the open vllm-neuron pull request
[vllm-project/vllm-neuron#69](https://github.com/vllm-project/vllm-neuron/pull/69).

- **Target:** `mistralai/Voxtral-Mini-3B-2507`, BF16, tensor parallel 4.
- **Draft (optional):** any DFlash draft in the upstream vLLM DFlash checkpoint format. The examples
  use the public [`jburtoft/Voxtral-Mini-3B-2507-DFlash`](https://huggingface.co/jburtoft/Voxtral-Mini-3B-2507-DFlash)
  (Apache-2.0, block size 16). Its training code is at
  [jimburtoft/voxtral-dflash](https://github.com/jimburtoft/voxtral-dflash).
- **Endpoint:** the standard OpenAI-compatible `/v1/audio/transcriptions`.

Why the extra steps: vllm-neuron 0.29.0.1.2.0, the version in the SDK 2.33 DLAMI, supports DFlash
but not Voxtral yet. Started on stock 0.29.0.1.2.0, `vllm serve` fails with
`AttributeError: type object 'VoxtralForConditionalGeneration' has no attribute 'from_configs'`.
PR #69 is the 0.29.0.1.2.0 release plus the Voxtral model and the DFlash fixes it needs. You install
it next to the DLAMI's vLLM environment, which stays unchanged. Once the PR is merged and released,
steps 3 and 4 go away.

## 1. Launch an instance

- **Instance:** `trn2.3xlarge` (1 Trainium2 device, 4 logical NeuronCores at the default LNC=2).
- **AMI:** *Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0* (multi-framework). Latest ID:
  ```bash
  aws ssm get-parameter --region <region> \
    --name /aws/service/neuron/dlami/multi-framework/ubuntu-24.04/latest/image_id \
    --query Parameter.Value --output text
  ```
  (us-east-1: `ami-091af6c736073d840`, `20261020`.)
- **Disk:** 300 GB or more.

## 2. Add swap (once per instance)

The DLAMI ships without swap, and the compiler can use a lot of host memory while it builds the
TP=4 graphs.
```bash
sudo fallocate -l 64G /swapfile && sudo chmod 600 /swapfile
sudo mkswap --force /swapfile && sudo swapon /swapfile
```

## 3. Install the Voxtral support next to the DLAMI vLLM environment

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate   # vllm 0.29.0, vllm-neuron 0.29.0.1.2.0

# vllm-neuron with Voxtral (PR #69, pinned commit, includes the batched DFlash mask fix), in a side directory. --no-deps leaves the venv as shipped.
git clone https://github.com/jimburtoft/vllm-neuron.git -b voxtral-dflash ~/vllm-neuron-voxtral
git -C ~/vllm-neuron-voxtral checkout 66ed48f
pip install --target ~/vn_voxtral --no-deps ~/vllm-neuron-voxtral

# vLLM's "audio" extra is not in the DLAMI venv. Without it every transcription request returns HTTP 400
# ("Please install vllm[audio] for audio support").
pip install --target ~/vn_audio --no-deps soundfile soxr av

export PYTHONPATH=~/vn_voxtral:~/vn_audio
python -c "import vllm_neuron; print(vllm_neuron.__file__)"   # must print ~/vn_voxtral/vllm_neuron/__init__.py
```

Set `PYTHONPATH` the same way in every shell that runs `vllm serve`.

## 4. Serve

Without a draft (baseline):
```bash
vllm serve mistralai/Voxtral-Mini-3B-2507 \
  --tokenizer-mode mistral --tensor-parallel-size 4 --dtype bfloat16 \
  --max-model-len 2048 --max-num-seqs 1 --max-num-batched-tokens 2048 \
  --limit-mm-per-prompt '{"audio": 1}' --no-enable-prefix-caching --mm-processor-cache-gb 0 \
  --additional-config '{
    "neuron_config": {"num_batched_tokens_buckets": [512, 2048], "num_seqs_buckets": [1],
                      "on_device_sampling_config": {"all_greedy": true}},
    "encoder_neuron_config": {"num_encoder_input_buckets": [1, 4], "encoder_attention_block_size": 375, "tp_size": 4}}' \
  --port 8000
```

With a DFlash draft, add one flag:
```bash
  --speculative-config '{"method": "dflash", "model": "jburtoft/Voxtral-Mini-3B-2507-DFlash", "num_speculative_tokens": 15}'
```
`model` can be a Hugging Face repo id or a local directory. Set `num_speculative_tokens` to the
draft's `block_size - 1`.

The first start compiles the encoder, the decoder and (with a draft) the draft graphs. In our runs,
startup to ready took 4-8 minutes, depending on the draft and the long-audio buckets. The server
is ready when `curl -s localhost:8000/health` returns 200.

Settings:
- `encoder_attention_block_size: 375` is fixed: one 30 s audio window is 375 audio tokens.
- `num_encoder_input_buckets`: the most 30 s windows one request can have. `[1, 4]` covers clips up
  to 2 minutes.
- `encoder_neuron_config.tp_size` must divide 20 (the encoder's head count). Use 4 on trn2.3xlarge.
- For long audio (up to 5 minutes): `--max-model-len 8192 --max-num-batched-tokens 8192`,
  `"num_batched_tokens_buckets": [512, 8192]`, `"num_encoder_input_buckets": [1, 4, 10]`, and
  `export VLLM_NEURON_RESERVED_HEADROOM_GIB=8` before `vllm serve`.
- Batched serving (several requests in flight): use `--max-num-seqs 16` and add
  `"decode_context_length_buckets": [512]` to `neuron_config` and `"encoder_cache_num_blocks": 34`
  (2 x max-num-seqs + 2) to `encoder_neuron_config`. Use commit `66ed48f` or later: with earlier
  commits the draft's acceptance drops by about a third when 2-8 requests are in flight. See the
  Voxtral recipe in the PR (`docs/model-recipes/voxtral.md`) for the measured throughput.

## 5. Transcribe

A WAV or FLAC file (tested with 16 kHz mono 16-bit WAV):
```bash
curl -s localhost:8000/v1/audio/transcriptions \
  -F file=@clip.wav -F model=mistralai/Voxtral-Mini-3B-2507 -F language=en -F temperature=0.0
```

Check that the draft is actually being used. A draft that loads but is not used still gives correct
transcripts, so look at the acceptance counters, not only the text:
```bash
curl -s localhost:8000/metrics | grep -E '^vllm:spec_decode_num_(drafts|accepted_tokens)'
```
Mean acceptance length (tau) = 1 + `accepted_tokens / drafts`. With the public draft on English
read speech, expect about 6-8. A value near 1 means the draft path is broken: wrong
`target_layer_ids`, wrong `mask_token_id`, or a `num_speculative_tokens` that does not match the
block size.

## Using your own DFlash checkpoint

The draft directory must follow the upstream vLLM DFlash format:
- `config.json` with `"architectures": ["DFlashDraftModel"]`, `"model_type": "qwen3"`, `block_size`,
  and `"dflash_config": {"target_layer_ids": [...], "mask_token_id": N}`. `target_layer_ids` are
  Voxtral text-decoder layer indices; the server captures their outputs as the draft's context.
- `model.safetensors` with the draft weights. The token embedding and LM head can be left out:
  the draft then uses Voxtral's own.
- Optional: a reduced draft vocabulary (`draft_vocab_size` + a `d2t` tensor) and/or
  `mask_embedding.pt`. PR #69 reads both.
- Set `rope_theta` both at the top level and in `rope_parameters`. vLLM falls back to 1e6 when it
  cannot find it, which silently breaks a draft trained with a different value.

Point `--speculative-config` `model` at that directory and set `num_speculative_tokens` to
`block_size - 1`.

## Measured results

The settings above with the public draft, measured on trn2.3xlarge with SDK 2.33 on 2026-10-22.
One request at a time, greedy, real audio through `/v1/audio/transcriptions`. Short clips: 50
random held-out clips per set (<= 30 s, 300 in total). Long-form: 60 AMI test windows (30 s - 5 min)
with the long-audio settings.

| set | tau | mean latency without / with draft | speedup | WER without / with draft |
|---|---|---|---|---|
| LibriSpeech test-clean | 8.47 | 249 / 93 ms | 2.67x | 2.29 / 2.29 % |
| LibriSpeech test-other | 7.08 | 222 / 92 ms | 2.40x | 4.45 / 4.45 % |
| VoxPopuli test | 8.50 | 247 / 92 ms | 2.68x | 8.22 / 8.22 % |
| FLEURS en_us test | 6.42 | 265 / 108 ms | 2.46x | 4.10 / 4.18 % |
| AMI ihm test | 6.02 | 140 / 76 ms | 1.84x | 16.84 / 17.91 % |
| Common Voice 17 test | 4.17 | 136 / 87 ms | 1.56x | 11.53 / 11.95 % |
| **all short clips** | **6.84** | **210 / 91 ms** | **2.30x** | **6.55 / 6.71 %** |
| AMI long-form | 3.78 | 4,992 / 2,487 ms | 2.01x | 34.59 / 34.55 % |

290 of the 300 short transcripts are identical with and without the draft. The others differ
through bf16 rounding in the multi-token verify graph, not through the draft's proposals. The
benchmark script is
[`eval/vllm_neuron_serve_bench.py`](https://github.com/jimburtoft/voxtral-dflash/blob/main/eval/vllm_neuron_serve_bench.py),
and the full method is in that repo's README, "Serving benchmark". Your speedup depends on your
audio and on how well your draft matches it.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `VoxtralForConditionalGeneration has no attribute 'from_configs'` | Stock vllm-neuron is being used. Check `PYTHONPATH` (step 3). |
| HTTP 400, server log says `Please install vllm[audio]` | The audio packages are missing from `PYTHONPATH` (step 3). |
| Encoder head-count error | `encoder_neuron_config.tp_size` must divide 20. |
| Encoder bucket error on a long clip | Add a larger `num_encoder_input_buckets` value (one per 30 s window). |
| `nrt_tensor_allocate status=4` during warmup with long audio | `export VLLM_NEURON_RESERVED_HEADROOM_GIB=8`. |
| `neuronx-cc was forcibly killed` during the first start | Host out of memory. Add swap (step 2). |
| tau near 1 with a draft | Draft config mismatch. See "Using your own DFlash checkpoint". |
