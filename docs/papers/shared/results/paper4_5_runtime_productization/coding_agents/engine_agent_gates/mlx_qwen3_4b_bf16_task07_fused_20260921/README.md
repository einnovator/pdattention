# MLX Qwen3-4B BF16 Task-7 frozen agent gate

This bundle records the first post-fix, same-endpoint mini-swe-agent triple for
locked baseline-success Task 7 (`django__django-14089`).  All arms use
Qwen3-4B BF16 revision
`1cfa9a7208912126459214e8b04321603b3df60c`, temperature 0, seed 0, the
`qwen3-stable-no-thinking` chat template, mini-swe-agent 2.4.6, and the official
SWE-bench 4.1.0 Docker grader.

## Admitted result

| Arm | Official result | Calls | Action trajectory | Patch | Mean realized retention | Selected-history re-encoding |
|---|---:|---:|---|---|---:|---:|
| Plain | 1/1 | 7 | reference | reference | n/a | n/a |
| PRA-100 | 1/1 | 7 | exact match | exact match | 100% | 0 tokens |
| Tail-90 | 1/1 | 5 | first divergence at action 3 | exact match | 93.81% | 0 tokens |

Tail-90 materializes 7,595 physical input tokens from 8,096 logical tokens, an
own-trajectory physical saving of 6.19%.  Against PRA-100's 12,093 physical
input tokens, paired physical workload falls 37.20%; 33.05 points come from the
shorter five-call trajectory.  This is one task, not an accuracy estimate or a
latency claim.

The direct sparse production path reports zero selected-history re-encoding,
zero selected-interval packing, and zero host-to-device bytes.  Its cumulative
`total_kv_copy_bytes` is not zero: PRA-100 reports 3.546 GB and tail-90 reports
3.423 GB.  Those totals include canonical suffix grafts and the qualification-
only packed same-subset reference run beside each request.  Peak whole-request
temporary allocation is 371.9 MB at PRA-100 and 626.5 MB at tail-90.  Selection
aliasing, validation-reference copying, canonical growth, and consumer scratch
therefore remain separate metrics.

## Correctness audit

The eager segmented MLX consumer failed the first real sparse request with a
same-subset maximum logit delta of 0.5 (limit 0.005).  The fused disjoint
consumer passed that request and the complete autonomous task.  The live gate
then matched all 7/7 full-retention turns and all 4/4 eligible sparse turns;
three earlier turns were explicitly ineligible because no old whole causal
group could yet be removed.  The sparse live gate re-encoded zero selected
tokens but its dense qualification reference physically packed the subset.

The initial clean triple was rerun after correcting a stale validator constant
that expected mini-swe-agent 2.4.0 even though the frozen campaign and installed
package use 2.4.6.  The manifests in this bundle all report
`exact_environment: true`.

## Files

- `gate.json`: reduced matched triple and gate decisions.
- `*_run_manifest.json`: immutable configuration and runtime identity.
- `*_official_result.json`: normalized official grader output.
- `*_request_telemetry.jsonl`: per-request logical/physical accounting.
- `live_gate_100.json`, `live_gate_090.json`: same-resident K/V equivalence
  gates bound to the exact chat-template digest.

Invalid infrastructure and eager-consumer attempts are quarantined outside the
admitted bundle and are not included in any denominator.
