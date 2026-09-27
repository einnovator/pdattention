# Pi persistent-session N=5/N=6 evidence

This compact bundle records the first completed boundary-free Pi 0.75.3 /
Qwen3-Coder-30B M2/P1 execution through six ordered SWE-bench Verified
identities.  The full request traces remain immutable on the execution host;
their hashes and the hashes of every compact artifact are recorded in
`summary.json` and `n5_derivation_manifest.json`.

The exact matched five-task cohort resolves 3/5 under persistent FULL and 4/5
under M2/P1.  M2/P1 preserves all three FULL successes, loses none, and adds
one recovery diagnostic.  It uses 155 versus 152 calls, saves 40.92% against
its own trajectory, and saves 37.51% against paired FULL.  Charging two
discarded Task-5 attempts reduces paired saving to 29.05% and changes the call
delta from +3 to +27.

The completed six-task M2/P1 trajectory resolves 5/6 in 194 calls and
materializes 3,446,052 of 7,085,344 counterfactual full-history tokens, for
51.36% own saving.  It has no paired N=6 saving denominator: persistent FULL's
first Task-6 request contained 53,484 message-content tokens and produced no
response before the frozen timeout.  The only unresolved M2/P1 identity is
`django__django-15741`, which persistent FULL also fails.

Files:

- `n5_pair_reduction.json`: fail-closed matched-cohort reduction.
- `n5_control_report.json` and `n5_candidate_report.json`: official graders.
- `n5_derivation_manifest.json`: exact prefix and artifact identities.
- `n6_campaign_state.json`: completed autonomous six-task state.
- `n6_official_report.json`: normalized official 5/6 result.
- `n6_predictions.json`: the six graded patches.
- `n6_label_normalization_manifest.json`: proves that report-label
  normalization changed no patch.
- `summary.json`: compact headline and failure-aware accounting.

This is one ordered task-clustered execution, not a population accuracy
estimate and not yet a production-default qualification.
