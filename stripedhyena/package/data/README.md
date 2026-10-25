# data/

`parity_reference.json` (62 KB): the parity reference used by `tests/test_hardware.py` when the full reference
logits are not available (`SHN_REFDATA` unset). Produced from the CUDA reference run (upstream
`togethercomputer/stripedhyena` @ `7e13f61` with flash-attn 2.8.3 + flash RMSNorm, StripedHyena-Hessian-7B rev
`59e474c`, bf16 weights with fp32 poles/residues) for two prompts, `short` (8 tokens) and `medium_512` (512 tokens):

- `prompt`: prompt token ids (Llama tokenizer; `short` = 8 tokens, `medium_512` = 512 tokens of WikiText-2)
- `gen`: the reference greedy continuation (128 tokens)
- `top_ids` / `top_logp`: for each of the 128 teacher-forced positions, the reference top-16 token ids and their
  fp32 log-probabilities (exact ties may be ordered differently from argmax; top-1 is checked against `gen`, which equals
  the reference argmax at every position)

Gate: top-1 agreement >= 0.98 and mean KL over the reference top-16 <= 0.005. The full-logit gate (rel L2 <= 0.03,
top-1 >= 0.98, full KL <= 0.005) is used when `SHN_REFDATA/ref/forced_cached_<prompt>.pt` exist.
