# Task 3 recent-frontier DAG M=2 repeat 2

- instance: `scikit-learn__scikit-learn-12585`
- historical paired FULL: resolved in 6 calls, 88,510 cumulative input tokens
- DAG M=2: official failure (`patch_apply_failed`) in 8 calls, 103,709 tokens
- paired saving: -17.17%
- own-trajectory pruning: 13.52%

The repeat again makes the correct one-line source mutation but submits prose
instead of a unified diff. Two concordant failures localize the defect to
submission-protocol behavior rather than task understanding. The P1 repair is
therefore evaluated separately and these failures remain in the primary
ledger.

The JSON metrics, official result, manifest, and trajectory are copied
verbatim from the immutable remote run directory.
