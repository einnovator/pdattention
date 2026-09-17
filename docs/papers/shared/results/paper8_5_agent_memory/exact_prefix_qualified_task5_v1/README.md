# Exact-prefix-qualified Task 5 comparison

This bundle contains one strict, same-prefix comparison on
`django__django-16145`. All three executions start at episode 5 from the same
four-episode persistent FULL prefix:

- prefix SHA-256: `3dffd88030a654a228545b71f999e7e9bcf14a1154e362ed1fce13a456b9e4ed`;
- preceding issues: `sympy__sympy-16886`, `django__django-15277`,
  `scikit-learn__scikit-learn-12585`, and `pytest-dev__pytest-7982`;
- model: `qwen3-coder:30b`, revision
  `06c1097efce0431c2045fe7b2e5108366e43bee1b4603a7aded8f21689e90bca`;
- temperature `0`, top-p `1`, seed `0`, and 131,072-token served context.

| Arm | Official solve | Calls | Full-history counterfactual | Materialized | Own-trajectory saving | Paired saving vs FULL |
|---|---:|---:|---:|---:|---:|---:|
| FULL | 1 | 9 | 191,697 | 191,697 | 0.00% | 0.00% |
| M1/P1/W1 | 0 | 9 | 191,610 | 88,506 | 53.81% | 53.83% |
| M2/P1 | 1 | 9 | 194,161 | 97,780 | 49.64% | 48.99% |

The pair passed identity checks for pair ID, task, frozen prefix, logical
session, source workspace, model and tokenizer revisions, harness, scaffold,
and decoding parameters. FULL ran first and solved officially, so the W1 loss
is attributable to the treatment under the experiment contract rather than to
pre-existing multi-task interference.

The first useful mechanistic difference is current-task source localization.
FULL performs several targeted reads before one correct mutation. M1/P1/W1
reads the whole file and immediately edits the wrong default-address
assignment; later inspections do not repair the semantic mistake. M2/P1 keeps
two additional targeted address-processing reads. It makes one unsuccessful
edit, localizes the parsing block, applies the correct conversion, and solves
in the same nine calls as FULL.

This is a one-task, one-repeat result. It establishes a quality boundary and a
testable mechanism, not a population accuracy estimate or deployable default.
A promotion decision requires another successful exact-prefix FULL control
immediately before a repeated M2/P1 treatment and expansion to other task
identities. The persistent export currently reports eight receipts for nine
commands in each arm; the raw selection traces, trajectories, workspace
checkpoints, manifests, and official grader outputs are retained so that this
terminal-sidecar discrepancy remains auditable.

`evidence.json` is the compact ledger. Each arm directory contains its manifest,
per-request selection trace, trajectory, instrumentation checkpoints, official
result, and autonomous metrics.
