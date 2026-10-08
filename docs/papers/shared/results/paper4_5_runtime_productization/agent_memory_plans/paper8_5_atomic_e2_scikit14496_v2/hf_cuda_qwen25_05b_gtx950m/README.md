# HF/CUDA Atomic-E2 frozen-ledger qualification

This bundle consumes the two hash-bound Paper 8.5 Atomic-E2 ledgers without
rerunning or retuning the selector.  It uses Transformers/PyTorch on a local
4-GB NVIDIA GTX 950M and `Qwen/Qwen2.5-0.5B-Instruct` revision
`7ae557604adf67be50417f59c2c2f167def9a775`.  The smaller model is a bounded
engine-mechanism probe; this is not autonomous SWE-bench quality evidence.

| Repeat | Source K/V | Selected K/V | Wire suffix | Visible saving | Resident omission | Re-encoded | Selected K/V copied | Attention temporary | Same-subset / restore |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| r01 / request 8 | 28,503 | 12,641 | 306 | 55.06% | 55.65% | 0 | 0 B | 316,882,944 B | logit exact / logit exact |
| r02 / request 9 | 28,321 | 12,459 | 305 | 55.41% | 56.01% | 0 | 0 B | 315,850,752 B | logit exact / logit exact |

Both runs preserve the exact record intervals and original source positions.
They use the `streaming_segmented` CUDA attention path, create no interval
packing or transient K/V-copy buffer, and pass concurrent borrowing,
cancellation, stale-generation rejection, borrower-safe offload, exact
restoration, termination and tombstone checks.  Offload serializes
350,259,283 and 348,022,867 bytes respectively; those lifecycle transfers are
reported separately from zero-copy request attachment.

The source fixtures are the same `r01_frozen_plan.jsonl` and
`r02_frozen_plan.jsonl` used by llama.cpp, MLX, SGLang-MLX and vLLM-Metal.
Tokenizer geometry is unchanged for this fixture: HF realizes the same
28,503/12,641 and 28,321/12,459 source/selected counts.  Bounded-prefill step
sizes differ (128 for r01, 256 for r02) only to fit and accelerate the old GPU;
they do not alter model-visible tokens or selected K/V.

Raw artifact SHA-256 values:

- `r01.json`: `686ccb54906aa9bbc720cd412246edf0087d4b416dcc2122e95e97feb2f4b891`
- `r02.json`: `efde9c1ab17880d7b6c5bd097ed6bc227c2da8dec14f3ea6abc13140e15db6d4`

`provenance.json` binds those artifacts to the execution revision, CUDA
environment, Hugging Face snapshot revision, and hashes of every model and
tokenizer file used by the offline run.

Claim boundary: repeat-qualified request-level same-subset correctness,
zero-copy attachment and lifecycle only.  Latency is quarantined because the
GTX 950M is not a representative deployment device, and no autonomous HF
agent run is included.
