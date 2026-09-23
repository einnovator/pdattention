# Corrected OpenHands H2/T4 tail-90, Task 2

This is the second officially graded OpenHands identity for the corrected,
declared-safe H2/T4 within-task policy. It uses OpenHands 1.49.2,
`qwen3-coder:30b` revision
`06c1097efce0431c2045fe7b2e5108366e43bee1b4603a7aded8f21689e90bca`,
the exact tokenizer, temperature 0, top-p 1, seed 0, disabled native
condensation, and the same `django__django-15368` image as its admitted FULL
control.

The candidate officially resolves the task in 26 actions versus FULL's 25.
It produces a different but resolving 671-byte patch. Selection begins at
request 8 and the first assistant-content divergence is response 9. It saves
9.12% against its own full-history counterfactual and 6.12% against the paired
FULL cumulative input.

Together with Task 1, corrected OpenHands H2/T4 now preserves 2/2 official
solves, uses 57 versus 58 actions, saves 27.42% against the aggregate candidate
counterfactual, and saves 33.50% against aggregate paired FULL input. The large
per-task range (35.35% versus 9.12% own saving) confirms that opportunity is a
workload property even when the agent, model, tokenizer, and policy are fixed.

This two-identity cohort is transfer evidence, not a population accuracy
estimate or production default. `qualification.json` is the compact decision
record; the run manifest, event summary, patch, and official report are kept
beside it.
