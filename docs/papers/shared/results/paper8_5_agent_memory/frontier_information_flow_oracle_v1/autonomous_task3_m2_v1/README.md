# Task 3 recent-frontier DAG M=2

- instance: `scikit-learn__scikit-learn-12585`
- historical paired FULL: resolved in 6 calls, 88,510 cumulative input tokens
- DAG M=2: official failure (`patch_apply_failed`) in 7 calls, 89,876 tokens
- paired saving: -1.54%
- own-trajectory pruning: 13.63%

The workspace patch contains the correct one-line source mutation, but the
agent submits explanatory prose rather than a unified git diff. The immediately
preceding failed episode contains the same malformed protocol example, while
the older successful protocol completion is retired. This motivates the P1
valid-protocol control edge; it does not reclassify this official failure.

The JSON metrics, official result, manifest, and trajectory are copied
verbatim from the immutable remote run directory.
