# Baseline reproduction report

PRA and gateway treatments remain locked until the matching no-PRA cell is `BASELINE_REPRODUCED`.

| Cell | Model | Harness | Target | Observed | Status | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| `easy50-no-pra` | `qwen3-coder:30b` | `mini-swe-agent` | 20.0%--80.0% | 28.0% | BASELINE_REPRODUCED | Official result satisfies the pinned local admission contract. |
