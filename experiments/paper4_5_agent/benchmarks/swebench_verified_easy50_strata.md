# Easy-50 conditional-retention protocol

The admitted direct no-PRA baseline partitions the frozen Easy-50 cohort before
any treatment result is inspected:

- `baseline_success14`: the 14 baseline-resolved tasks, used first to estimate
  retention and regression;
- `baseline_failure36`: the remaining 36 tasks, used only after the retention
  stage to estimate acquisition and persistent failure.

Treatment order is G11 PRA gateway plus native llama.cpp E2, Headroom 0.37.0,
then G10 Selected Context. All treatments retain Qwen3-Coder-30B Q4_K_M,
mini-swe-agent 2.4.6, the backticks scaffold, 32,768 context, 50 steps,
temperature zero, and the official SWE-bench 4.1.0 Docker grader.

Headroom is pinned to `headroom-ai==0.37.0` in its default production `cache`
mode. Memory, learning, output shaping, and remote telemetry are disabled.
Local request logs and `/stats` telemetry must be retained. Headroom-reported
compression is reported separately from model-server prompt/cache counters and
must not be relabeled as physical K/V savings.

Primary success-stratum metrics are retained/14 and regressed/14. Primary
failure-stratum metrics are acquired/36 and still-unsolved/36. Overall solved
count is secondary and is always decomposed as retained plus acquired.
