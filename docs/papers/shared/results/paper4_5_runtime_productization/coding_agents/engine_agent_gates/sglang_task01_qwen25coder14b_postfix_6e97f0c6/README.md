# Fresh SGLang-MLX autonomous Task 1 gate

This directory records the post-fix autonomous mini-swe-agent gate for
`django__django-15277` on the 48 GB M4 host.  All three runs use the same
Qwen2.5-Coder-14B-Instruct-4bit revision, temperature zero, sampling seed zero,
chat template, container image, and official SWE-bench grader.

| mode | official solve | calls | trajectory vs plain | physical input | selected history re-encoded | selected K/V copy | suffix-graft copy | consumer temporary peak |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| plain | 1/1 | 6 | reference | not instrumented | not applicable | not applicable | not applicable | not instrumented |
| PRA-100 task-aware | 1/1 | 6 | all six actions and patch exact | 8,920 / 8,920 | 0 tokens | 0 bytes | 0 bytes | 50,944,156 bytes |
| PRA-90 task-aware | 1/1 | 6 | first divergence at final response; patch exact | 8,725 / 8,920 (2.19% saving) | 0 tokens | 0 bytes | 17,498,112 bytes | 1,610,612,740 bytes |

The strict PRA-100 behavioral and resident-subset gates pass.  The reduced run
also preserves the official solve, calls, and patch, but must not be described
as globally zero-copy: selected history is neither encoded nor copied, while a
separate canonical-suffix graft copies 17.5 MB.  The large reduced-path
temporary allocation is a systems optimization target, not a selection-quality
failure.

The serving process used the source content committed as `6e97f0c6` for the
current-Transformers bootstrap alias.  The runner's historical
`engine_version` label remains `sglang-mlx-agent-02e53d01`; this discrepancy is
disclosed here rather than silently rewriting immutable run manifests.

Raw run manifests, trajectories, interaction histories, telemetry, and
official grader summaries are under `raw/`.  `summary.json` is the reduced
machine-readable comparison.
