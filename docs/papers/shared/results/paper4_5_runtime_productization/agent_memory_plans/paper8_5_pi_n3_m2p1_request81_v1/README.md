# Frozen Pi M2/P1 request 81

This directory is the first Paper 8.5 Pi request in the prospective persistent
N=3 campaign for which `frontier_dag_m2_heuristic_p1` actually excludes
history.  The policy abstains during Tasks 1--2; request 81 is the first model
request of Task 3 after one instruction epoch becomes older than the M=2
frontier.

The original campaign predated direct selective-request capture.  The full
logical request was therefore reconstructed from Pi's append-only native
session with
`experiments.paper8_5_agent_memory.reconstruct_pi_openai_request` at Paper 8.5
commit `41a2094f`.  Reconstruction is accepted only because the 162-message
payload exactly matches the frozen proxy identity
`0ad3e4de6aadbacb08af1d1483a08bf749e47a23d112e26fa6ee21f032c4222c`.
The source hashes and reconstructed-output hash are in
`reconstruction_manifest.json`; `export_manifest.json` binds that request,
the native action log, the selection trace, and the two engine fixtures.

The frozen Paper 8.5 row contains 31,404 message-content tokens, selects and
materializes 12,435, and excludes 18,969 tokens in 49 causal groups.  This is
60.40% logical saving for this request.  The Qwen3-Coder tokenizer audit maps
the same decision to:

- 39,987 full prompt tokens;
- 37,843 resident source-history tokens and a 2,144-token live wire tail;
- 13,235 selected resident K/V tokens plus the identical live tail;
- 38.46% realized prompt retention, or 61.54% token/K/V opportunity;
- two original-position physical intervals, `[0, 2319)` and
  `[26927, 37843)`.

These files prove logical identity and engine-token geometry only.  They do
not yet prove zero re-encoding, zero-copy attention, lifecycle safety, logit
equivalence, or runtime benefit.  Those gates must be produced by each native
Paper 4.5 engine against `request_replay.json` and `selection_fixture.json`.

## MLX lifecycle qualification

`mlx_qwen3_06b_lifecycle_selective.json` applies this frozen selection to the
corrected segmented MLX path using Qwen3-0.6B-4bit as a fast engine-mechanics
probe.  The engine-prefixed geometry is 37,784 source tokens, 13,418 selected
resident tokens, and a 2,144-token wire suffix: 35.51% historical retention
and 38.98% total realized retention.

The run qualifies the mechanism, not coding-agent model quality:

- zero selected-history text re-encoding and zero selection-pack bytes;
- exact same-subset, dense-engine-oracle, and restore logits (all reported
  maximum absolute deltas are `0.0`);
- no selected-K/V-sized consumer transient: 155,659 peak delta bytes versus
  54,960,128 bytes of selected layer K/V (`0.283%`);
- successful concurrent borrowing, cancellation, error cleanup, stale-fork
  rejection, eviction/offload, exact restoration, termination, and source
  tombstoning.

The corresponding Qwen3-Coder-30B autonomous task run and the same lifecycle
gate on HF, vLLM, SGLang, and llama.cpp remain separate required evidence.
