# Exact-prefix-qualified Task 3 comparison

This bundle contains one strict, same-prefix comparison on
`scikit-learn__scikit-learn-12585`. Both executions start at episode 3 from
the same two-episode persistent FULL prefix:

- prefix SHA-256: `a30f35190dbbcb79f28f8465943286fc1c5d56e31b3e8951baa487d18e463431`;
- preceding issues: `sympy__sympy-16886` and `django__django-15277`;
- model: `qwen3-coder:30b`, revision
  `06c1097efce0431c2045fe7b2e5108366e43bee1b4603a7aded8f21689e90bca`;
- temperature `0`, top-p `1`, seed `0`, and 131,072-token served context.

| Arm | Official solve | Calls | Full-history counterfactual | Materialized | Own-trajectory saving | Paired saving vs FULL |
|---|---:|---:|---:|---:|---:|---:|
| FULL | 1 | 7 | 103,713 | 103,713 | 0.00% | 0.00% |
| M2/P1 | 1 | 7 | 104,474 | 92,434 | 11.52% | 10.88% |

The pair passed identity checks for pair ID, task, frozen prefix, logical
session, source workspace, model and tokenizer revisions, harness, scaffold,
and decoding parameters. FULL ran first and solved officially. M2/P1 preserves
the solve and call count.

This low-saving point is informative rather than a failed policy gate. Only two
prior issues exist at episode 3, and M2 keeps the two most recent genuine-user
instruction epochs. The policy therefore has little previous-task history to
retire. The result is consistent with the predeclared scaling mechanism: gross
saving should grow later in a persistent session while current-task history
remains protected.

This is one task identity and one paired repeat. `evidence.json` is the compact
ledger. Each arm directory contains its manifest, per-request selection trace,
trajectory, instrumentation checkpoints, official result, and autonomous
metrics.
