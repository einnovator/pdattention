# Paper 2 Oracle-Gap Audit

This directory records the controlled diagnostic requested after the canonical
Qwen3-0.6B oracle-depth experiment. It does not change routing, materialization,
adapters, or SDK defaults.

## Protocol receipt

- Model: `Qwen/Qwen3-0.6B` at revision
  `c1899de289a04d12100db370d81485cdf75e47ca`.
- Split/cohort: validation, seed `20260811`, offset 8, four HotpotQA and four
  QASPER examples.
- Reference publication: 128-token independent blocks, 32-token nonoverlapping
  routing parents, source-position post-RoPE K and native V, CPU residency.
- Consumption: canonical last 14 layers and all 28 layers; 128 direct prompt
  tokens and at most 512 memory tokens under the 640-token native bound.
- Oracle input: evidence annotations only. Gold answer strings and target
  likelihood never enter span extraction or parent selection.

The canonical direct, last-14, and all-layer deltas reproduce the prior artifact
with maximum absolute difference `0.0`. Every representable annotation is
covered at every requested consumer layer, every requested identity survives
materialization, all 28 Qwen layer IDs are present, native-limit violations are
zero, and fresh-versus-cached post-RoPE K/V maximum absolute error is `0.0`.
The audit therefore found no hidden oracle-path bug or protocol mismatch.

## Main findings

| Diagnostic | HotpotQA | QASPER | Interpretation |
|---|---:|---:|---|
| Exact evidence delta | +4.635 | -1.791 | Hotpot annotations are sufficient; this QASPER slice is not a positive native-text control |
| Parent context delta | +5.574 | -1.791 | Bridge/parent context helps Hotpot; QASPER parents collapse to the exact annotation in this cohort |
| Feasible full-context delta | +3.771 | unavailable | Hotpot exact evidence is not weaker than full context under the bounded prompt |
| Late-14 pre-RoPE K cosine | .940 | .963 | Source-only bounded encoding differs from question-conditioned direct encoding |
| Late-14 V cosine | .892 | .926 | Contextual representation mismatch persists after removing RoPE position effects |
| Late-14 E / D / H mass | .220 / .310 / .469 | .565 / .027 / .408 | Parent distractors are material on Hotpot but negligible on QASPER |
| E mass after offline D removal | .281 | .579 | Renormalization recovers .061 on Hotpot and only .014 on QASPER |
| Final hidden relative L2, last-14 / all | .541 / .869 | .602 / 1.245 | All-layer replay accumulates a much larger residual-path departure |

The Hotpot result supports both contextual mismatch and softmax scattering, but
neither is sufficient by itself. Native direct evidence has lower mean evidence
attention mass (`.120`) than oracle last-14 (`.220`) while producing the larger
likelihood gain. QASPER has almost no selected-parent distractor mass but still
fails under all-layer replay. The common signal is that direct evidence is
jointly contextualized with the question through the ordinary residual path,
whereas source-only memory is reintroduced as parallel K/V support. Repeating
that intervention in every layer compounds the mismatch.

The 32/64/128/256/512 encoding-context sweep is a one-example-per-dataset
diagnostic with final parent spans held fixed. Hotpot last-14 utility rises from
`+2.985` to `+3.548`, then saturates; QASPER moves in the opposite direction
despite higher K/V cosine. Encoding context therefore matters, but cosine
fidelity is not a monotone proxy for causal utility.

`oracle_gap_audit.json` contains the complete per-example receipts, including
annotation text and spans, selected identities, physical token positions,
per-layer materialization, per-head attention summaries, K/V fidelity, and
hidden-state divergence. CSV files hold the paper-facing aggregates. The
counterfactual removes D only from already-computed softmax support; it is not a
real partial-materialization experiment.

Reproduce from the repository root:

```powershell
python -m experiments.paper2_hf.qa.run_oracle_gap_audit --device cuda
```
