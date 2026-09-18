# Corrected MLX native transfer: Scikit-learn 14496

This bundle records the first autonomous Paper 8.5 prompt-pinned E2+F1C
transfer that passes the corrected Paper 4.5 MLX same-subset gate. It is a
single task identity at episode six after the same frozen five-episode prefix
used by the llama.cpp transfer. The reduced arm repeats exactly under a cold
engine restart. This is repeat-qualified single-identity evidence, not a
population accuracy estimate.

## Frozen identities

- Task: `scikit-learn__scikit-learn-14496`, SWE-bench Verified revision
  `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`.
- Model: `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` revision
  `6e302ea604ad9ab206367e2c501d1571023e7b6d`.
- Tokenizer digest/revision:
  `06c1097efce0431c2045fe7b2e5108366e43bee1b4603a7aded8f21689e90bca`.
- mini-swe-agent 2.4.6, temperature 0, top-p 1, seed 0.
- Paper 8.5 runner revision `679fb46f33dcde776a1015a93f388faa02d4ea90`.
- Paper 4.5 MLX runtime revision `4451c522`, MLX 0.32.2 and mlx-lm 0.31.3.
- Runtime contract: original-position disjoint live K/V, compact closure
  receipts encoded separately, and a positioned packed same-subset oracle.

## Result

| Arm | Official solve | Calls | Materialized message-content tokens | Completion tokens | Selected-history re-encoding | Selected-history K/V copy |
|---|---:|---:|---:|---:|---:|---:|
| PRA-100 | 1/1 | 9 | 238,938 | 689 | 0 | 0 B |
| E2+F1C | 1/1 | 12 | 192,762 | 883 | 0 | 0 B |
| E2+F1C cold repeat | 1/1 | 12 | 192,762 | 883 | 0 | 0 B |

All eleven post-bootstrap E2+F1C requests pass the same-subset gate with
maximum absolute logit delta exactly 0.0. The sparse requests expose 302,031
full source K/V tokens to the counterfactual and attend to 180,052 selected
resident tokens plus 1,056 newly encoded receipt tokens. Steady-state K/V
omission is therefore 40.04%. Including the one-time 26,367-token source
bootstrap gives 328,398 versus 207,475 K/V-workload tokens, or 36.82% omission.

The two E2+F1C executions have identical request-input, assistant-content, and
shell-command hashes on all twelve requests. Their raw HTTP-body hashes differ
only because that scope includes volatile response metadata. Both have maximum
same-subset logit delta 0.0 and identical aggregate token metrics.

The autonomous trajectory is therefore the limiting factor. Relative to the nine-call
PRA-100 control, E2+F1C saves 19.33% of materialized message-content input,
19.19% after adding completion tokens, and 15.23% of the selected-K/V plus
receipt workload. Thus MLX realizes the same approximately 40% within-request
physical opportunity as llama.cpp, but three extra calls reduce the paired
end-to-end benefit.

The first three shell commands match PRA-100. At command four, E2+F1C reads
the second affected source location before editing, while PRA-100 edits the
first location immediately. E2+F1C then uses an incorrectly escaped `sed`
substitution and spends two further calls diagnosing and repairing it. The
active task statement and all active-task turns are retained, so this is not
loss of current-task evidence. It is a behavioral-priming effect caused by
retiring detail from earlier instruction epochs. The exact cold repeat makes
the call penalty stable enough to justify one targeted prior-mutation-exemplar
treatment; broad policy sweeping remains unwarranted.

The immutable traces were emitted before a shell-classifier correction: a
diagnostic redirect such as `find ... 2>/dev/null` was labeled as a write.
That label was not consumed by E2+F1C, so official outcomes, selected record
IDs, token/K/V totals, hashes, and same-subset gates are unchanged. Raw
`assistant_operation` and action-count fields in these traces must not be used
as mutation evidence; the corrected classifier now distinguishes diagnostic
sinks from redirects that write workspace files.

## Quarantine ledger

Earlier directories are not pooled with this result. They exposed, in order,
nonresident replacement resources, an MLX two-pass SIMD-plane failure, a
transport-only failure, an invalid packed causal reference (delta 2.7627), an
eager disjoint reference with unsuitable numerical drift (delta 0.703125),
and a positioned oracle whose full/vector dispatch differed from the candidate
(delta 0.53125). Runtime revision `4451c522` aligns candidate and oracle
receipt prefill with MLX vector SDPA and is the first qualifying revision.

`e2f1c/`, `e2f1c_repeat2/`, and `pra100/` contain immutable manifests,
request-level selection and engine metrics, official grader results,
persistent-episode exports, and the complete mini-swe-agent trajectories.
`summary.json` is the reduced comparison.
