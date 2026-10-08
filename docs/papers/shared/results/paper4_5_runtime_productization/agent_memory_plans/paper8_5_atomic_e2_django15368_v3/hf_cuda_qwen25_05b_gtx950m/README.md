# HF/CUDA held-out Atomic-E2 qualification

This bundle consumes the two hash-bound held-out Paper 8.5 Atomic-E2 ledgers
without rerunning or retuning the selector. It uses Transformers/PyTorch on a
local 4-GB NVIDIA GTX 950M and `Qwen/Qwen2.5-0.5B-Instruct` revision
`7ae557604adf67be50417f59c2c2f167def9a775`. The smaller model is a bounded
engine-mechanism probe; this is not autonomous SWE-bench quality evidence.

| Repeat | Source K/V | Selected K/V | Wire suffix | Visible saving | Resident omission | Re-encoded | Selected K/V copied | Attention temporary | Same-subset / restore |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| r01 / request 9 | 27,796 | 11,934 | 139 | 56.78% | 57.07% | 0 | 0 B | 144,506,880 B | logit exact / logit exact |
| r02 / request 7 | 27,393 | 11,531 | 138 | 57.62% | 57.91% | 0 | 0 B | 143,474,688 B | logit exact / logit exact |

Both runs preserve the exact record intervals and original source positions.
They use the `streaming_segmented` CUDA attention path, create no interval
packing or transient K/V-copy buffer, and pass concurrent borrowing,
cancellation, stale-generation rejection, borrower-safe offload, exact
restoration, termination and tombstone checks. Offload serializes 341,571,667
and 336,619,603 bytes respectively; those lifecycle transfers are reported
separately from zero-copy request attachment.

The source fixtures are the same `r01_frozen_plan.jsonl` and
`r02_frozen_plan.jsonl` used by llama.cpp, MLX, SGLang-MLX and vLLM-Metal.
Tokenizer geometry is unchanged for this fixture: HF realizes the same
27,796/11,934 and 27,393/11,531 source/selected counts. The bounded-prefill
step size is 256 in both runs and does not alter model-visible tokens or
selected K/V.

Raw artifact SHA-256 values:

- `r01.json`: `5b6a6ce5b3123d52f38659074487db586fa0e97dbf77598a58c2b839d4e9b7a2`
- `r02.json`: `c71bc647fd4a20fa6904bb5dd52645eb3a751f3fd09244e3b2ca76425d318b2b`

`provenance.json` binds those artifacts to the execution revision, CUDA
environment, Hugging Face snapshot revision, and hashes of every model and
tokenizer file used by the offline run.

Claim boundary: repeat-qualified request-level same-subset correctness,
zero-copy attachment and lifecycle only. Latency is quarantined because the
GTX 950M is not a representative deployment device, and no autonomous HF
agent run is included.
