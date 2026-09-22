# Paper 8.5 selection-only E2 transfer fixture

This frozen cohort exports the repeat-qualified Paper 8.5
`persistent_instruction_epoch_retirement` policy for the first held-out
`django__django-15741` episode. The selector emits original records only:
`materialized_message_replacements` is empty in all nine requests.

The files `request_replay.jsonl` and `frozen_plan.jsonl` are identity joined by
`request_input_sha256`. `frozen_plan_manifest.json` records the cohort digests.
`llamacpp_request9_strict_v1.json` is the isolated request-9 mechanism gate.

## Autonomous llama.cpp controls

The compact evidence under `llama_evidence/` uses the same Qwen3-Coder-30B
Q4_K_M checkpoint, task, five-episode prefix, temperature zero, and official
SWE-bench grader.

| arm | official | calls | logical selected/full | selected-history re-encoding | selected-K/V copy |
|---|---:|---:|---:|---:|---:|
| plain full-history engine | 1/1 | 10 | 274,532/274,532 | n/a | n/a |
| native PRA-100 | 1/1 | 8 | 217,970/217,970 | 0 | 0 B |
| strict selection-only E2 | 0/1 | 20 | 372,180/586,360 (36.53% saving) | 0 | 0 B |

PRA-100 matches the first three tool commands and the first two complete
assistant responses, then follows a shorter successful trajectory. It therefore
passes task and zero-re-encoding/copy gates but not exact action-trajectory
parity. The strict selective arm is an admissible negative transfer result: it
uses selected original-position resident K/V on every model-visible request,
but doubles the plain call count and fails official grading. It must not be
promoted as a default policy.

Earlier diagnostic runs that generated request 1 from FULL history, or sent an
empty selected suffix, are quarantined and are not part of this evidence set.

## Direct MLX 14B request gate

The corrected first-visible-request gate was also run on MLX 0.32.2 / MLX-LM
0.31.3 with `mlx-community/Qwen3-14B-4bit` at revision
`a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4`. Canonical source capture is
non-generating and selection is no longer deferred, but the fused disjoint
consumer fails the identical-selected-subset numerical gate: maximum logit
delta is `0.5` against a predeclared `0.01` limit. Autonomous execution is
therefore not admitted. The compact negative evidence is under
`mlx14_request9_fused/`.

The first attempt on a 16 GB Apple host was quarantined after active swap grew
to 23.5 GB during canonical capture. The reported strict results come from the
48 GB host and are not swap-thrashing measurements. The matched long-context
eager control also reports `0.5 > 0.01`; its candidate and dense-reference
first-token IDs are both `3617`. The frozen discrepancy is therefore not
specific to the fused Metal kernel. The tolerance must not be relaxed merely
to promote either path.

A bounded 154-token control under `mlx14_short_control/` localizes the
discrepancy. Fused and eager disjoint consumers each match their
same-consumer, physically packed reference exactly (`0.0` logit delta), but
differ from MLX native dense attention by `1.375` and `1.1611328125`
respectively. Both preserve the four-token trajectory; a separately labeled
fused diagnostic preserves 16/16 tokens. Thus record selection and interval
addressing are exact in this control, while cross-consumer numerical parity is
not. The original `0.01` promotion gate remains in force.
