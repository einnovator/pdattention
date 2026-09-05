# Baseline reproduction report

PRA and gateway treatments remain locked until the matching no-PRA cell is `BASELINE_REPRODUCED`.

| Cell | Model | Harness | Published | Observed | Status | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| `qwen3-coder-30b-no-pra` | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | `mini-swe-agent` | 14.0% | - | SKIPPED | Requires the exact H100-class source environment; disabled on the current fleet. |
| `gemma4-31b-no-pra` | `google/gemma-4-31B-it` | `mini-swe-agent` | 38.0% | - | SKIPPED | Preferred treatment baseline; requires one H100 80GB or equivalent. |
