# Corrected OpenHands H2/T4 tail-90, Task 3

This is the third officially graded OpenHands identity for the corrected,
declared-safe H2/T4 within-task policy. It uses OpenHands 1.49.2,
`qwen3-coder:30b` revision
`06c1097efce0431c2045fe7b2e5108366e43bee1b4603a7aded8f21689e90bca`,
the exact tokenizer, temperature 0, top-p 1, seed 0, disabled native
condensation, and the same `scikit-learn__scikit-learn-13135` image in both
arms. A transient endpoint failure occurred before the FULL agent started; the
clean immutable retry is the only FULL execution scored here.

Both arms officially resolve the task. H2/T4 uses 17 actions versus FULL's 19,
saves 8.14% against its own full-history counterfactual, and saves 18.75%
against paired FULL cumulative input. Selection and assistant-content
divergence both begin at request 8. The two arms produce different but
resolving variants of the same sorted-centers fix.

Across OpenHands Tasks 1--3, corrected H2/T4 preserves 3/3 official solves,
uses 74 versus 77 actions, saves 22.67% against aggregate candidate-own
counterfactual input, and saves 29.76% against aggregate paired FULL input.
The fixed-policy per-task own-saving range is 8.14--35.35%, so workload
structure remains a major effect modifier.

This three-identity cohort qualifies cross-agent policy transfer but is not a
population accuracy estimate or production default.
