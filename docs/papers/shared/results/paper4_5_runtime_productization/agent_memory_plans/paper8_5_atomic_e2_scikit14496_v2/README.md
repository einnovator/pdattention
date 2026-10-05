# Paper 8.5 Atomic-E2 native-K/V handoff

This directory freezes the two repeat-qualified Atomic-E2 executions for
`scikit-learn__scikit-learn-14496`, episode 6 of a persistent session with five
completed prior issues.  The selector operated without evaluator task IDs or
explicit task-boundary signals.

## Logical result inherited from Paper 8.5

- Model: `qwen3-coder:30b-ctx131k`, Q4_K_M, temperature 0, top-k 20.
- FULL and Atomic-E2 both resolved the task in both repeats (2/2 versus 2/2).
- Calls were tied in aggregate (17 versus 17).
- Atomic-E2 retained 197,592 of 460,531 own-counterfactual tokens: 57.09%
  logical own saving.
- This is a strict subset of original records.  It contains no synthetic
  receipts, replacement summaries, or compacted text that would require
  re-encoding old history.

The Paper 8.5 source bundle is
`paper8_5_agent_memory/atomic_e2_two_identity_repeat_v2` at revision
`3cc48ab4`.

## Frozen files

Each repeat contains a selection fixture, a request replay, and its manifest:

- `r01_frozen_plan.jsonl`, `r01_request_replay.jsonl`, `r01_manifest.json`
- `r02_frozen_plan.jsonl`, `r02_request_replay.jsonl`, `r02_manifest.json`

The manifests bind the fixture and replay hashes.  Paper 4.5 must consume these
files without rerunning the selector.

## Qualification contract

Logical selected-token counts must remain engine-independent.  Each engine is
qualified separately for original-position sparse K/V residence, selected
history re-encoding (required: zero), K/V copies, consumer temporary bytes,
new suffix encoding, and lifecycle safety.  The 57.09% logical saving is not an
engine K/V or latency result; page/block rounding and engine implementation are
reported separately.

No cross-engine autonomous-agent or production-default claim follows from this
handoff.  The first runtime gate is exact realization of the frozen selected
subset.  That request-level gate now passes on llama.cpp, direct MLX,
SGLang-MLX, and vLLM-Metal.  HF/CUDA and autonomous execution remain open.

## llama.cpp strict-subset result

The patched Metal llama.cpp engine at revision
`e6e5d63dbc09b5ea6153822e39057f9b7de40eb0` passes both independent final
requests:

| Repeat | Source K/V | Selected K/V | Wire suffix | Visible saving | Resident omission | History re-encoded | K/V copied | Repeated continuation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| r01/request 8 | 28,503 | 12,641 | 306 | 55.06% | 55.65% | 0 | 0 B | exact `[3617, 46]` |
| r02/request 9 | 28,321 | 12,459 | 305 | 55.41% | 56.01% | 0 | 0 B | exact `[3617, 46]` |

Both lifecycle clean-up checks pass and neither plan contains materialized
replacement history.  This is a 2/2 strict original-position resident-K/V
mechanism result.  It is not an autonomous cross-engine accuracy result.

The M4 Pro was concurrently running a high-CPU virtual machine during these
runs.  Prime and wall-time fields are preserved in the raw evidence but are
quarantined from runtime-economics claims; the admissible measurements here are
correctness, token accounting, re-encoding, copy, and lifecycle state.

## Cross-engine strict-subset replication

The same two hash-bound frozen ledgers were consumed without retuning by three
additional engines.  Direct MLX and SGLang-MLX preserve the exact llama.cpp
token geometry.  vLLM-Metal uses complete 16-token scheduler pages, so its
source tokenizer geometry differs by at most seven tokens and selection adds
22 tokens in each repeat.  These are explicit page-rounding effects, not
selector changes.

| Engine / model | Repeat | Source K/V | Selected K/V | Suffix | Visible saving | History re-encoded | Initial selected-K/V copy | Same-subset result | Lifecycle |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| MLX-LM 0.31.3 / Qwen3-0.6B-4bit | r01 | 28,503 | 12,641 | 306 | 55.06% | 0 | 0 B | exact logits, `[151667,198]` | pass |
| MLX-LM 0.31.3 / Qwen3-0.6B-4bit | r02 | 28,321 | 12,459 | 305 | 55.41% | 0 | 0 B | exact logits, `[151667,198]` | pass |
| SGLang-MLX / Qwen3-0.6B-4bit | r01 | 28,503 | 12,641 | 306 | 55.06% | 0 | 0 B | exact logits, `[151667,198]` | pass |
| SGLang-MLX / Qwen3-0.6B-4bit | r02 | 28,321 | 12,459 | 305 | 55.41% | 0 | 0 B | exact logits, `[151667,198]` | pass |
| vLLM-Metal 0.29.0 / Qwen3-0.6B-4bit | r01 | 28,496 | 12,656 | 313 | 54.98% | 0 | 0 B | exact tokens, `[151667,198]` | pass |
| vLLM-Metal 0.29.0 / Qwen3-0.6B-4bit | r02 | 28,320 | 12,480 | 306 | 55.33% | 0 | 0 B | exact tokens, `[151667,198]` | pass |

MLX and SGLang-MLX additionally match the dense same-subset logits exactly.
Their measured fused-consumer peak deltas are 290,827--291,883 bytes, only
0.562--0.572% of one selected layer's K/V bytes; no full-selected-K/V-sized
temporary is observed.  vLLM-Metal attaches selected pages by non-owning alias
with zero active-memory delta.  Its offload/restore lifecycle necessarily
materializes the restored source (3.25--3.27 GB); that restore copy is reported
separately and is not described as zero-copy request attachment.

The MLX and SGLang rows ran on a 48-GB M4 Pro.  The admitted vLLM-Metal rows
use the exact pinned 0.29.0 package at source revision
`7390805822b2d7a208b09d55bd07b7572f727e20` on that same host.  Attempts to run
the 0.3.0 package on a 16-GB M5 were rejected before admission when the
requested scheduler reserve exceeded stable capacity; they are configuration
diagnostics, not negative engine results.  All elapsed-time fields remain
quarantined from runtime-economics claims.
