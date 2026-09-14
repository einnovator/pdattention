# Paper 7 Review Status

**EXPERIMENTALLY FROZEN / READY FOR EXTERNAL REVIEW**

- Branch: `research/paper7-typed-adaptive-context`
- Previous frozen manuscript commit: `23dbc397c875ff6ec4af218d2e820bdf15a40edf`
- Cross-evaluation add-on: refrozen on this branch; commit recorded in Git history
- Paper: `paper7_typed_adaptive_context_inception.pdf` (21 pages)
- Focused verification: 22 selected tests passed
- Full verification: 779 tests passed, 1 upstream Transformers deprecation warning
- Visual verification: all 21 rendered pages inspected, with result pages 10--12 checked at full resolution

## Frozen Results

- `PRA_NATIVE` matches `FULL_BACKING` at 83.3% held-out task success while
  reducing active K/V from 304.6 to 182.9 tokens (39.9%).
- At the 256K-token payload, compact-first gating reduces instrumented Torch
  time to usable context from 2223.4 ms to 961.5 ms (2.3x).
- The controlled 32-token selected-region test reaches 1.000 marker recall.
- The frozen controller matches `PRA_NATIVE`; oracle control reaches 100.0% by
  selecting `CALL_TOOL` when required information is absent from backing state.
- Released Headroom 0.37.0 default and validation-selected profiles match the
  83.3% controlled endpoint; tuned visible context is 123.8 tokens. The matched
  controller fails every C5 external-acquisition action.
- Frozen PRA reaches 100% CCR-needle Recall@4 and 100% tool-output Recall@8 in
  the focused reverse evaluation, but 0% on three extractive HotpotQA cases and
  20% MS MARCO Recall@8 over five selected-passage cases.

## Claim Boundaries

- The TTUC profile measures the instrumented Torch mechanism, not Qwen
  end-to-end latency or a universal speedup.
- Perfect controlled marker recall validates lifecycle and bounded recovery,
  not semantic equivalence to full-context generation.
- Thresholds are configurable policy values rather than learned or optimal
  constants.
- `CCR_STYLE` is an in-house reproduction, not released Headroom. Official
  Kompress ML, external-provider execution, and TOIN cold/warm behavior remain
  unmeasured; plain-text rows are evidence-preservation controls.
- Distributed storage, multi-GB streams, concurrent mutation, encrypted-store
  overhead, remote authorization propagation, and durable audit delivery remain
  deployment-specific work.
