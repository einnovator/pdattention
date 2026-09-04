# Baseline reproduction report

PRA and gateway treatments remain locked until the matching no-PRA cell is `BASELINE_REPRODUCED`.

| Cell | Model | Harness | Published | Observed | Status | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| `fim7b-q4-no-pra-smoke` | `TIGER-Lab/FIM-7B` | `R2E-Gym` | 17.8% | 50.0% | BASELINE_ATTEMPTED | Configuration difference: engine=llama.cpp-not-vLLM; Configuration difference: quantization=Q4_K_M-not-BF16; Configuration difference: cohort=2-not-500; Configuration difference: context=32768-not-65536; Cohort size differs: observed 2, published 500. |
