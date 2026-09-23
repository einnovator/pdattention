# Kilo declared-safe Task 1 pairing audit

This bundle records the first normalized Kilo 7.7.5 `FULL` versus
declared-safe H2/T4 tail-90 pair on `django__django-15277` with
`qwen3-coder:30b` at temperature zero.

Kilo inserts a wall-clock `Message time:` field into its model-visible
environment block. The harness now normalizes only that declared volatile
field before hashing and forwards a fixed replacement to both arms. After
normalization, both arms have the same initial request digest
(`8eaa9120...`) and identical generation controls.

Both executions resolve officially and produce the same 689-byte patch
(`8512d6bc...`). The candidate retires only whole records and saves 7.30% of
its own full-history counterfactual. It does **not** qualify for paired causal
economics: the first canonical tool action diverges at action 1, while
selection first changes model input at request 8. The candidate therefore
consumes 376,980 selected content tokens versus 146,013 in the paired FULL
trajectory, a descriptive paired saving of -158.18%.

The preselection divergence demonstrates backend/model trajectory variance
despite identical initial model-visible input. Accordingly,
`qualification.json` sets `causal_attribution_qualified` to false. This bundle
supports official capability preservation for one identity and a negative
economics result; it is not evidence that selection caused the trajectory
difference and is not promoted as a policy default.

Key files:

- `qualification.json`: strict pairing and promotion decision.
- `*_run_manifest.json`: immutable run identity and policy configuration.
- `*_event_summary.json`: Kilo-native action and tool-event totals.
- `*_official_report.json`: SWE-bench grader outcomes.
