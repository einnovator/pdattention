# Structured-evidence threshold screening

This directory records the adaptive threshold screen on the frozen successful
`scikit-learn__scikit-learn-13135` trajectory with `qwen3-coder:30b`.

The 256- and 512-token thresholds are one materialization equivalence class on
this task. Both retain 188,072 of 199,109 cumulative message-content tokens
(5.54% saving) at exactly the same 23 decisions. The only extra record entering
structured handling at threshold 256 is a 259-token observation for which the
structured selector retains every line. The no-op byte-identity rule therefore
restores its original content. All records that lose content are identical in
the two arms.

Despite equivalent model-visible histories, conservative action agreement is
63.6% at threshold 256 and 54.5% at threshold 512; exact-command agreement is
54.5% and 45.5%, respectively. This difference is endpoint repeat variability,
not a threshold-quality effect. The planned 1024/2048 continuation was stopped
because those values remain in the same effective class for this trajectory.
Future threshold campaigns must derive breakpoints from observed tool-result
sizes and deduplicate arms by the newly recorded `request_messages_sha256`.

Files:

- `structured_t256.json`: completed 23-decision arm;
- `comparison_equivalent_thresholds.{json,md}`: FULL, threshold 256, and the
  existing threshold-512 repeat reduced together.
