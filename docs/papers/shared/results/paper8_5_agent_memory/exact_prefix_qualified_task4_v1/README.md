# Exact-prefix-qualified Task 4 comparison

This bundle contains one strict, same-prefix comparison on
`pytest-dev__pytest-7982`. Both executions start at episode 4 from the same
three-episode persistent FULL prefix:

- prefix SHA-256: `75b0135f34db8dcb277c50cc700b5b996c112fbb15863dd8c71e77f18e8bf73f`;
- preceding issues: `sympy__sympy-16886`, `django__django-15277`, and
  `scikit-learn__scikit-learn-12585`;
- model: `qwen3-coder:30b`, revision
  `06c1097efce0431c2045fe7b2e5108366e43bee1b4603a7aded8f21689e90bca`;
- temperature `0`, top-p `1`, seed `0`, and 131,072-token served context.

| Arm | Official solve | Calls | Full-history counterfactual | Materialized | Own-trajectory saving | Paired saving vs FULL |
|---|---:|---:|---:|---:|---:|---:|
| FULL | 1 | 14 | 260,662 | 260,662 | 0.00% | 0.00% |
| M2/P1 | 1 | 6 | 104,705 | 49,667 | 52.56% | 80.95% |

The pair passed identity checks for pair ID, task, frozen prefix, logical
session, source workspace, model and tokenizer revisions, harness, scaffold,
and decoding parameters. FULL ran first and solved officially. M2/P1 then
preserved the official solve while using eight fewer calls.

Two savings denominators are reported because the action trajectories differ.
The 52.56% own-trajectory saving measures selection within M2/P1's six-call
trajectory. The 80.95% paired saving compares its materialized history with
the successful 14-call FULL control and therefore includes a favorable
trajectory-length effect. The paired number is valid for this task-level
outcome but must not be interpreted as a pure selector compression ratio or a
population expectation.

This is one task identity and one paired repeat. It is a strict additional
success for M2/P1, not yet evidence for a deployable default. Qualification
continues on other task identities with a successful exact-prefix FULL control
required before each treatment. The persistent exports currently report one
fewer receipt than commands; raw traces, trajectories, checkpoints, manifests,
and official grader outputs are retained so that this terminal discrepancy is
auditable.

`evidence.json` is the compact ledger. Each arm directory contains its manifest,
per-request selection trace, trajectory, instrumentation checkpoints, official
result, and autonomous metrics.
