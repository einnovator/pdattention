# Paper 8.5 Easy-14 profile-transfer decision

This receipt freezes the logical-policy decision imported by Paper 4.5. The
atomic E1/E2/E3 names below are Paper 8.5 policy labels, not PRA engine-depth
labels. It is not an engine-speed result and it does not authorize Easy-14
native-K/V expansion.

The source campaign uses one persistent ordinary-role session, clean SWE-bench
workspaces, `qwen3-coder:30b`, temperature zero, and a verified 131,072-token
active context. The selector receives no evaluator task ID, repository label,
workspace identity, or explicit issue boundary.

| Arm | Observed issues | Official resolution | Calls | Materialized input | Decision |
|---|---:|---:|---:|---:|---|
| Persistent FULL | 14/14 | 10/14 | 207 | 8,902,105 | Agent-Full correctness control |
| Atomic E2 | 5/14 | 2/5 vs FULL 4/5 | 84 vs 85 | 1,016,037 vs 1,415,568 | Stopped; 28.22% raw but zero failure-aware saving |
| Atomic E3, no-intervention prefix | 4/14 | 2/4 vs FULL 3/4 | 126 vs 79 | 2,765,676 vs 1,215,410 | Stopped before first deletion |

Exact-prefix Task 5 uses the same saved four-episode FULL prefix and clean
workspace for two repeats per arm:

| Policy | Resolved | Calls | Full-history counterfactual | Materialized | Paired result |
|---|---:|---:|---:|---:|---|
| FULL | 1/2 | 8 | 263,178 | 263,178 | Reliability control |
| Atomic E3 | 1/2 | 9 | 298,777 | 219,442 | 16.62% paired saving; +1 call |
| Atomic E2 | 1/2 | 27 | 937,220 | 544,181 | -106.77% paired saving from trajectory inflation |

No reduced-history policy meets the predeclared 30--50% paired-saving,
zero-loss, no-call-increase gate. Therefore:

- Agent-Full is the only qualified profile.
- Agent-Quality maps to E3 only as a quarantined research candidate.
- Agent-Balanced/E2 is rejected as a default.
- Agent-Economy/E1 is unqualified because the less aggressive E2 already
  failed.

Paper 4.5 may use E2/E3 only as exact-plan diagnostic treatments. It must key
materialized identity by `selected_messages_sha256`, not the canonical
pre-selection request digest, and independently report official resolution,
calls, materialized tokens, selected-history re-encoding, K/V-copy bytes, and
consumer temporary bytes.

Source provenance:

- Paper 8.5 commit: `433bac19`
- implementation/reduction commit: `fe8d2643`
- campaign: `paper85-autonomous-easy14-atomic-profiles-ctx131k-v2`
- immutable remote root:
  `/Users/jorge.simao/git/rd/paper85-runs/easy14-atomic-profiles-ctx131k-v2-retry06`
