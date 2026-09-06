# SWE-bench Lite-50

This frozen cohort is the second autonomous repository-fixing tier after the
SWE-bench Verified Easy frontier. It is sampled from the official SWE-bench
organization's Lite `test` split at revision
`b0dde1093fe417d83b7184254edf8199c1f0dff5`.

| Field | Value |
| --- | --- |
| Upstream | `SWE-bench/SWE-bench_Lite` |
| Source | https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite |
| License | MIT |
| Split | `test` |
| Official grader | SWE-bench Docker evaluation harness |

The builder sorts every task by
`SHA256("paper4.5-swebench-lite50-v1" + NUL + instance_id)` and takes the first
50. Selection uses no patch, grader-test, or model-outcome field. Regenerate the
identity card with:

```bash
python -m experiments.paper4_5_agent.build_swebench_lite_cohort
```

Lite-50 is not launched until the easier local baseline is admitted and its
matched PRA frontier is complete. Its smaller, mostly localized changes make it
useful as a bridge between the Easy cohort and the externally reported fixed-50
Verified cohort, but it does not replace multi-file context-demand tests.
