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
subset.  That request-level gate now passes on llama.cpp, HF/CUDA, direct MLX,
SGLang-MLX, and vLLM-Metal.  Autonomous HF execution remains open.

HF/CUDA now additionally qualifies both complete request sequences, not only
their final requests.  All 17/17 frozen requests pass.  Repeat 1 selects
90,923 of 217,819 historical K/V entries and encodes 4,288 new wire-suffix
tokens; repeat 2 selects 102,534 of 245,292 and encodes 4,108 suffix tokens.
Their visible-context omissions are 57.1328% and 57.2406%, respectively, only
0.1019 and 0.0890 percentage point above Paper 8.5's 57.0309% and 57.1516%
own logical omissions.  Combined, the engine selects 193,457 of 463,111
historical K/V entries, encodes 8,396 suffix tokens, and realizes 57.1898%
visible omission versus 57.0947% logical omission, a +0.0951-point difference.
Selected-history re-encoding, interval packing, and selected-K/V attachment
copies are all zero.  The maximum consumer temporary is 1,887,879,168 bytes;
lifecycle offload payloads remain separate.  These are frozen request-sequence
mechanism results, not autonomous solves, HBM-saving, or timing results.
Receipts and bound reducers are under
`engine_sequences/hf_cuda_qwen25_05b_gtx950m_r01_v1` and
`engine_sequences/hf_cuda_qwen25_05b_gtx950m_r02_v1`.

The fail-closed whole-sequence matrix is materialized in
`cross_engine_sequence_gate_r01.json` and
`cross_engine_sequence_gate_r02.json`.  Both currently admit HF/CUDA and name
the five still-missing required sequence summaries explicitly: llama.cpp
Metal, direct MLX, SGLang-MLX, vLLM-Metal, and vLLM-CUDA.  The reducer rejects
different fixture/replay hashes or per-request plan identities even if every
engine passes in isolation.  It also rejects selected-history re-encoding,
selection packing, or selected-K/V request-attachment copies, and reports
logical-token, resident-K/V, and visible-context omission separately.  The
gate artifacts remain negative until every declared engine consumes the same
complete request sequence; the 2/2 final-request table below does not satisfy
that stronger requirement.

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

The same two hash-bound frozen ledgers were consumed without retuning by four
additional engines.  HF/CUDA, direct MLX and SGLang-MLX preserve the exact
llama.cpp token geometry.  vLLM-Metal uses complete 16-token scheduler pages,
so its source tokenizer geometry differs by at most seven tokens and selection
adds 22 tokens in each repeat.  These are explicit page-rounding effects, not
selector changes.

| Engine / model | Repeat | Source K/V | Selected K/V | Suffix | Visible saving | History re-encoded | Initial selected-K/V copy | Same-subset result | Lifecycle |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| HF CUDA / Qwen2.5-0.5B-Instruct FP16 | r01 | 28,503 | 12,641 | 306 | 55.06% | 0 | 0 B | exact logits | pass |
| HF CUDA / Qwen2.5-0.5B-Instruct FP16 | r02 | 28,321 | 12,459 | 305 | 55.41% | 0 | 0 B | exact logits | pass |
| MLX-LM 0.31.3 / Qwen3-0.6B-4bit | r01 | 28,503 | 12,641 | 306 | 55.06% | 0 | 0 B | exact logits, `[151667,198]` | pass |
| MLX-LM 0.31.3 / Qwen3-0.6B-4bit | r02 | 28,321 | 12,459 | 305 | 55.41% | 0 | 0 B | exact logits, `[151667,198]` | pass |
| SGLang-MLX / Qwen3-0.6B-4bit | r01 | 28,503 | 12,641 | 306 | 55.06% | 0 | 0 B | exact logits, `[151667,198]` | pass |
| SGLang-MLX / Qwen3-0.6B-4bit | r02 | 28,321 | 12,459 | 305 | 55.41% | 0 | 0 B | exact logits, `[151667,198]` | pass |
| vLLM-Metal 0.29.0 / Qwen3-0.6B-4bit | r01 | 28,496 | 12,656 | 313 | 54.98% | 0 | 0 B | exact tokens, `[151667,198]` | pass |
| vLLM-Metal 0.29.0 / Qwen3-0.6B-4bit | r02 | 28,320 | 12,480 | 306 | 55.33% | 0 | 0 B | exact tokens, `[151667,198]` | pass |

HF, MLX and SGLang-MLX additionally match the dense same-subset logits exactly.
HF's streaming segmented CUDA attention reports 315,850,752--316,882,944
temporary bytes without interval packing or transient K/V copying.  The MLX
and SGLang measured fused-consumer peak deltas are 290,827--291,883 bytes, only
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
