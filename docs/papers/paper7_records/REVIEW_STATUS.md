# Paper 7 Review Status

**EXPERIMENTALLY FROZEN / READY FOR EXTERNAL REVIEW**

- Branch: `research/paper7-typed-adaptive-context`
- Frozen manuscript commit: `23dbc397c875ff6ec4af218d2e820bdf15a40edf`
- Paper: `paper7_typed_adaptive_context_inception.pdf` (17 pages)
- Focused verification: 25 tests passed
- Full verification: 775 tests passed, 1 upstream Transformers deprecation warning
- Visual verification: all 17 rendered pages inspected

## Frozen Results

- `PRA_NATIVE` matches `FULL_BACKING` at 83.3% held-out task success while
  reducing active K/V from 304.6 to 182.9 tokens (39.9%).
- At the 256K-token payload, compact-first gating reduces instrumented Torch
  time to usable context from 2223.4 ms to 961.5 ms (2.3x).
- The controlled 32-token selected-region test reaches 1.000 marker recall.
- The frozen controller matches `PRA_NATIVE`; oracle control reaches 100.0% by
  selecting `CALL_TOOL` when required information is absent from backing state.

## Claim Boundaries

- The TTUC profile measures the instrumented Torch mechanism, not Qwen
  end-to-end latency or a universal speedup.
- Perfect controlled marker recall validates lifecycle and bounded recovery,
  not semantic equivalence to full-context generation.
- Thresholds are configurable policy values rather than learned or optimal
  constants.
- Distributed storage, multi-GB streams, concurrent mutation, encrypted-store
  overhead, remote authorization propagation, and durable audit delivery remain
  deployment-specific work.
