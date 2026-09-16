# Easy-14 context-qualified atomic-profile campaign

This bundle records the first Easy-14 persistent-session campaign for which the active `qwen3-coder:30b` runtime was fail-closed at a 131,072-token context. The ordered sequence uses clean SWE-bench workspaces but one continuous ordinary-role conversation. The policy receives no evaluator task ID, repository label, workspace identity, or explicit issue boundary.

## Primary campaign

| Arm | Observed issues | Official resolution | Calls | Selected input | Paired interpretation |
|---|---:|---:|---:|---:|---|
| Persistent FULL | 14/14 | 10/14 | 207 | 8,902,105 | Control |
| Atomic E2 | 5/14 | 2/5 vs FULL 4/5 | 84 vs 85 | 1,016,037 vs 1,415,568 | 28.22% raw; zero failure-aware saving; stopped after two lost FULL successes |
| Atomic E3 no-intervention prefix | 4/14 | 2/4 vs FULL 3/4 | 126 vs 79 | 2,765,676 vs 1,215,410 | Stopped before first deletion because FULL-equivalent replay had already diverged |

The largest FULL request is 85,801 message-content tokens, below the verified 131,072-token active context. E2 does not retire history in Tasks 2–3; the changed call counts in those tasks are therefore endpoint-repeatability controls rather than selection effects. E3 performs no deletion through Task 4, yet Tasks 3–4 both reach the 40-call limit. Its first intervention was consequently evaluated from the exact saved FULL prefix instead of the altered autonomous prefix.

## Exact-prefix Task-5 diagnosis

All rows start from the same four-episode FULL prefix and a clean `pytest-dev__pytest-7982` workspace.

| Policy | Repeats resolved | Calls | Full-history counterfactual | Materialized | Result |
|---|---:|---:|---:|---:|---|
| FULL | 1/2 | 8 | 263,178 | 263,178 | Reliability control |
| Atomic E3 | 1/2 | 9 | 298,777 | 219,442 | 26.55% gross, 16.62% paired saving; one extra call |
| Atomic E2 | 1/2 | 27 | 937,220 | 544,181 | 41.94% gross, but -106.77% paired saving from trajectory inflation |

No tested policy meets the predeclared 30–50% paired-saving, zero-loss, no-call-increase gate. E3 is the nondominated diagnostic point, not a production default. E2 demonstrates why per-request retention cannot substitute for realized-trajectory accounting.

The first FULL request is byte-identical across its two executions but produces different first actions and opposite official outcomes. Divergence identity is therefore computed from `selected_messages_sha256`; `request_input_sha256` hashes canonical pre-selection input and cannot establish model-visible equality after selection.

## Provenance

- campaign: `paper85-autonomous-easy14-atomic-profiles-ctx131k-v2`
- remote immutable root: `/Users/jorge.simao/git/rd/paper85-runs/easy14-atomic-profiles-ctx131k-v2-retry06`
- implementation commit used for final reduction: `fe8d2643`
- primary machine: `192.168.1.8`; model endpoint: `192.168.1.6:11435`
- full evidence, request traces, official grader reports, active-context probes, and immutable retry directories remain under the remote root
- `prefix_curves.csv` and `easy14_atomic_prefix_frontier.pdf` are descriptive stopped-prefix curves, not population confidence intervals

Failed 32K-context attempts, Docker-pull failures, interrupted Task-5 launches, and the stalled corrected spine-envelope retry are quarantined and excluded.
