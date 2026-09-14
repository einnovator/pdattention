# Autonomous structured-observation Task 5 pilot

This bounded pilot runs `scikit-learn__scikit-learn-13135` with
mini-swe-agent, `qwen3-coder:30b`, temperature zero, and a 512-token threshold
for structured materialization of oversized tool observations. It excludes no
logical causal group.

| Arm | Official resolution | Calls | Materialized / trajectory-full tokens | Gross saving |
|---|---:|---:|---:|---:|
| FULL A | 1 | 23 | 197,796 / 197,796 | 0.00% |
| FULL B | 0 (empty submission) | 19 | 155,753 / 155,753 | 0.00% |
| Structured evidence | 1 | 22 | 159,482 / 169,850 | 6.10% |

The treatment is a promising single-task point: it resolves officially with
one fewer call than the successful FULL execution while saving 6.10% of its
own trajectory's model-visible message tokens. It is not a profile result. The
FULL repeats differ in both trajectory length and submission outcome; FULL B's
terminal workspace contains a nonempty 778-byte patch but its submitted patch
is empty. Therefore paired trajectory saving and a quality confidence interval
remain unavailable until this arm and both FULL controls are repeated on
multiple same-host task identities.

`request_selection.jsonl` preserves per-request selected/materialized counts;
`autonomous_metrics.json`, `official_result.json`, and `run_manifest.json`
preserve the primary outcomes and pairing fields. Large Docker/grader trees are
not duplicated in this compact publication bundle; their remote paths and
digests remain in the manifests.
