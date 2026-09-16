# Paper 8.5 R4/M2/V2/P1 qualification

This bundle freezes the logical policy admitted to Paper 4.5's native-K/V
evaluation. It is not an engine-performance result.

| Execution | FULL | P1 | FULL calls | P1 calls | Paired saving |
|---|---:|---:|---:|---:|---:|
| 1 | 3/3 | 3/3 | 53 | 47 | 36.41% |
| 2 | 3/3 | 3/3 | 53 | 44 | 45.82% |

Across two executions of the same ordered Django--Pytest--Scikit-learn
sequence family, P1 preserves all six paired successes, saves a descriptive
mean 41.11% of cumulative message-content input, and reduces calls by 7.5 on
average. This passes the predeclared discovery gate (30--50% paired saving,
zero lost successes, no call increase), but it is not held-out-sequence or
population confirmation.

P1 adds one clean completed action--observation exemplar per prior typed
task/episode to the R4/M2/V2 progress spine. The runtime interface exposes this
as `recent_protocol_turns=1`; agent-neutral harnesses supply `memory_roles` and
`episode_id`, `task_id`, or `issue_id`. The narrow Bash/return-code inference is
only a mini-swe-agent compatibility fallback.

Paper 4.5 must now consume the frozen record identities from resident K/V and
measure selected-history re-encoding, K/V copy, temporary bytes, calls, and
official resolution. The logical Paper 8.5 numbers are not latency or cache
savings claims.

Evidence hashes:

- `execution_1_evidence.json`: `2160b24271317bb3c0d6e360e455902dc3830db51e10a0bdf09dc0b0a954262f`
- `execution_2_evidence.json`: `58a1af7ca9c6dd7ee83af13169f0c1ccea30d2d8c043ea8dc64ba63ca96ad61e`
