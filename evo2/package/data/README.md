# data/

`cpu_fp32_reference.json` (84 KB): the reference used by `tests/test_logit_parity.py`. Produced by
`../harness/cpu_fp32_reference.py` from the same `evo2_7b_base.pt` checkpoint and vortex commit, on CPU in fp32
(no Neuron, no bf16, no NKI kernels): prompt `ACGTACGTGGCCATTAGCTAGCTAGGCATCGA`, `max_seqlen` 256, the prefill logits
(512-entry vocabulary; 4 of the entries are DNA bases) and 6 greedy decode steps with their full logit vectors and top-1/top-2 margins.
