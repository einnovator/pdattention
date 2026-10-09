# HF/CUDA Atomic-E2 full request sequence

This directory realizes all eight frozen requests from Paper 8.5 repeat 1 on
the native HF/CUDA live-K/V path.  The selector was not rerun.  Every receipt
is bound to `r01_frozen_plan.jsonl` and `r01_request_replay.jsonl` in the
parent fixture directory.

## Result

| Coordinate | Result |
| --- | ---: |
| Qualified requests | 8/8 |
| Historical source K/V entries | 217,819 |
| Selected historical K/V entries | 90,923 |
| Newly encoded wire-suffix tokens | 4,288 |
| Historical-K/V omission | 58.2575% |
| Total visible-context omission | 57.1328% |
| Paper 8.5 own logical omission | 57.0309% |
| Visible-minus-logical difference | +0.1019 percentage point |
| Selected-history re-encoding | 0 tokens |
| Selection packing | 0 bytes |
| Requests with selected-K/V attachment copy | 0 |
| Maximum consumer temporary | 1,887,879,168 bytes |
| Cumulative explicit offload payload | 2,676,675,224 bytes |

Every request passes same-subset token and logit equality, the ordinary
full-engine token oracle, original-position execution, concurrent-borrower
isolation, cancellation, stale-generation rejection, offload, exact restore,
and termination.  `summary.json` is the authoritative reducer output and
contains the artifact hash for every request receipt.

The 0.1019-point agreement is measured rather than assumed.  Paper 8.5 uses
the campaign tokenizer and logical materialization denominator; this engine
run uses the pinned HF tokenizer and includes the current wire suffix in its
visible denominator.  The historical-only engine coordinate is therefore
58.2575%, not 57.0309%.

## Claim boundary

This is a frozen request-sequence mechanism result, not a fresh autonomous
agent execution.  It establishes that the Paper 8.5 record policy maps to an
almost identical cumulative visible-context omission on HF while consuming
resident original-position K/V without re-encoding or copying selected
history.  It does not establish lower total HBM residence or a speedup: the
canonical source K/V remains separately resident or offloaded, consumer
temporary memory peaks at 1.89 GB, and lifecycle offload traffic is reported
separately.  No elapsed-time value from this run is admitted as runtime
economics.

## Environment and execution

- Paper 4.5 revision: `82bafb9a4007b3ba311966b051b3c432369d92d9`.
- Model: `Qwen/Qwen2.5-0.5B-Instruct` at revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Runtime: PyTorch `2.12.1+cu126`, Transformers `4.55.4`, Python `3.10.11`.
- Consumer: FP16 streaming segmented attention on the local 4-GB GTX 950M.
- Requests 1--3 were checkpointed by isolated model processes.  Requests
  4--8 reused one loaded model but created a fresh PRA runtime and lifecycle
  state for each request.  This changes setup cost only; timing is excluded.
- Continuation length: two tokens; bounded prefill step: 128 tokens.

