# Canonical Evidence Matrix

This page compares the same task, hardware, engine, model, mode, and profile under **No PRA**, **PRA - No Adaptor**, and **PRA - Adaptor Bundle**. Values are absolute measurements; each delta is candidate minus No PRA. Missing data is never rendered as zero.

## Coverage by model, engine, mode, and profile

| Model | Engine | Mode | Profile | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Evidence tier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mlx-community/Qwen3-14B-4bit` | mlx | Native Memory | QUALITY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | ENGINE_QUALIFIED |
| `mlx-community/Qwen3-14B-4bit` | mlx | Native Memory | BALANCED | MEASURED (16) | MEASURED (16) | NOT_MEASURED | ENGINE_QUALIFIED |
| `mlx-community/Qwen3-14B-4bit` | mlx | Native Memory | ECONOMY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | ENGINE_QUALIFIED |
| `mlx-community/Qwen3-14B-4bit` | mlx | Native Memory | QASPER-LEARNED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | ENGINE_QUALIFIED |
| `mlx-community/Qwen3-32B-4bit` | mlx | Native Memory | QUALITY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | ENGINE_QUALIFIED |
| `mlx-community/Qwen3-32B-4bit` | mlx | Native Memory | BALANCED | MEASURED (16) | MEASURED (16) | NOT_MEASURED | ENGINE_QUALIFIED |
| `mlx-community/Qwen3-32B-4bit` | mlx | Native Memory | ECONOMY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | ENGINE_QUALIFIED |
| `mlx-community/Qwen3-8B-4bit` | mlx | Native Memory | QUALITY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | ENGINE_QUALIFIED |
| `mlx-community/Qwen3-8B-4bit` | mlx | Native Memory | BALANCED | MEASURED (16) | MEASURED (16) | NOT_MEASURED | ENGINE_QUALIFIED |
| `mlx-community/Qwen3-8B-4bit` | mlx | Native Memory | ECONOMY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | ENGINE_QUALIFIED |
| `Qwen/Qwen2.5-1.5B-Instruct` | hf | Native Memory | QUALITY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| `Qwen/Qwen2.5-1.5B-Instruct` | hf | Selected Context | BALANCED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| `Qwen/Qwen2.5-1.5B-Instruct` | hf | Native Memory | ECONOMY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| `Qwen/Qwen2.5-1.5B-Instruct` | hf | Native Memory | QASPER-LEARNED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | hf | Native Memory | QUALITY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | hf | Selected Context | BALANCED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | hf | Native Memory | ECONOMY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | hf | Native Memory | QASPER-LEARNED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| `mlx-community/Llama-3.1-8B-Instruct-4bit` | mlx | Native Memory | QUALITY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| `mlx-community/Llama-3.1-8B-Instruct-4bit` | mlx | Selected Context | BALANCED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| `mlx-community/Llama-3.1-8B-Instruct-4bit` | mlx | Native Memory | ECONOMY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| `mlx-community/Llama-3.1-8B-Instruct-4bit` | mlx | Native Memory | QASPER-LEARNED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| `mlx-community/Qwen3-4B-4bit` | mlx | Native Memory | QUALITY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| `mlx-community/Qwen3-4B-4bit` | mlx | Selected Context | BALANCED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| `mlx-community/Qwen3-4B-4bit` | mlx | Native Memory | ECONOMY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| `mlx-community/Qwen3-4B-4bit` | mlx | Native Memory | QASPER-LEARNED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| `mlx-community/gemma-3-1b-it-4bit` | mlx | Native Memory | QUALITY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| `mlx-community/gemma-3-1b-it-4bit` | mlx | Selected Context | BALANCED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| `mlx-community/gemma-3-1b-it-4bit` | mlx | Native Memory | ECONOMY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| `mlx-community/gemma-3-1b-it-4bit` | mlx | Native Memory | QASPER-LEARNED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| `Qwen/Qwen3-0.6B` | hf | Selected Context | QUALITY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | RESEARCH |
| `Qwen/Qwen3-0.6B` | hf | Selected Context | BALANCED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | RESEARCH |
| `Qwen/Qwen3-0.6B` | hf | Selected Context | ECONOMY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | RESEARCH |
| `Qwen/Qwen3-0.6B` | hf | Selected Context | QASPER-LEARNED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | RESEARCH |
| `mlx-community/Qwen3-4B-8bit` | mlx | Native Memory | QUALITY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | ENGINE_QUALIFIED |
| `mlx-community/Qwen3-4B-8bit` | mlx | Native Memory | BALANCED | MEASURED (16) | MEASURED (16) | NOT_MEASURED | ENGINE_QUALIFIED |
| `mlx-community/Qwen3-4B-8bit` | mlx | Native Memory | ECONOMY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | ENGINE_QUALIFIED |
| `mlx-community/Qwen3-4B-8bit` | mlx | Native Memory | QASPER-LEARNED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | ENGINE_QUALIFIED |
| `mlx-community/Qwen3-8B-8bit` | mlx | Native Memory | QUALITY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | ENGINE_QUALIFIED |
| `mlx-community/Qwen3-8B-8bit` | mlx | Native Memory | BALANCED | MEASURED (16) | MEASURED (16) | NOT_MEASURED | ENGINE_QUALIFIED |
| `mlx-community/Qwen3-8B-8bit` | mlx | Native Memory | ECONOMY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | ENGINE_QUALIFIED |
| `mlx-community/Qwen3-14B-8bit` | mlx | Native Memory | QUALITY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | ENGINE_QUALIFIED |
| `mlx-community/Qwen3-14B-8bit` | mlx | Native Memory | BALANCED | MEASURED (16) | MEASURED (16) | NOT_MEASURED | ENGINE_QUALIFIED |
| `mlx-community/Qwen3-14B-8bit` | mlx | Native Memory | ECONOMY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | ENGINE_QUALIFIED |
| `mlx-community/Qwen3-8B-6bit` | mlx | Native Memory | QUALITY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | ENGINE_QUALIFIED |
| `mlx-community/Qwen3-8B-6bit` | mlx | Native Memory | BALANCED | MEASURED (16) | MEASURED (16) | NOT_MEASURED | ENGINE_QUALIFIED |
| `mlx-community/Qwen3-8B-6bit` | mlx | Native Memory | ECONOMY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | ENGINE_QUALIFIED |
| `mlx-community/Llama-3.2-1B-Instruct-8bit` | mlx | Native Memory | QUALITY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | ENGINE_QUALIFIED |
| `mlx-community/Llama-3.2-1B-Instruct-8bit` | mlx | Native Memory | BALANCED | MEASURED (16) | MEASURED (16) | NOT_MEASURED | ENGINE_QUALIFIED |
| `mlx-community/Llama-3.2-1B-Instruct-8bit` | mlx | Native Memory | ECONOMY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | ENGINE_QUALIFIED |
| `mlx-community/gemma-3-1b-it-8bit` | mlx | Native Memory | QUALITY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| `mlx-community/gemma-3-1b-it-8bit` | mlx | Selected Context | BALANCED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| `mlx-community/gemma-3-1b-it-8bit` | mlx | Native Memory | ECONOMY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| `Qwen/Qwen2.5-1.5B-Instruct` | hf | Native Memory | QUALITY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| `Qwen/Qwen2.5-1.5B-Instruct` | hf | Selected Context | BALANCED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| `Qwen/Qwen2.5-1.5B-Instruct` | hf | Native Memory | ECONOMY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |

A `MEASURED (n)` cell reports the number of scalar metrics available for that exact condition. Detailed values follow only for measured records; profile rows without matched evidence remain explicit.

## Measured absolute values and deltas

### mlx-community/Qwen3-14B-4bit / mlx-lm / native-memory / balanced

Task: `combined`. Hardware: `Apple M4 Pro (Mac16,7), 48 GB`. Evidence: `ENGINE_QUALIFIED`.

#### Quality

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Token F1 | fraction | higher_is_better | 0.27746 | 0.27746 | NOT_MEASURED | +0 (+0.00%) | NOT_MEASURED |
| Exact Match | fraction | higher_is_better | 0 | 0 | NOT_MEASURED | +0 | NOT_MEASURED |
| Gold Answer Log Probability | log_probability | higher_is_better | -9.97839 | -9.97839 | NOT_MEASURED | +0 (-0.00%) | NOT_MEASURED |

#### Context

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Visible Tokens | token | lower_is_better | 315.533 | 34.2667 | NOT_MEASURED | -281.267 (-89.14%) | NOT_MEASURED |
| Selected Native K/V Tokens | token | neutral | 0 | 11250.7 | NOT_MEASURED | +11250.7 | NOT_MEASURED |

#### Serving

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| TTFT p50 (ms) | ms | lower_is_better | 210.724 | 210.373 | NOT_MEASURED | -0.351416 (-0.17%) | NOT_MEASURED |
| TTFT p95 (ms) | ms | lower_is_better | 339.229 | 338.285 | NOT_MEASURED | -0.94425 (-0.28%) | NOT_MEASURED |
| TTFT p99 (ms) | ms | lower_is_better | 339.229 | 338.285 | NOT_MEASURED | -0.94425 (-0.28%) | NOT_MEASURED |
| ITL p50 (ms) | ms | lower_is_better | 34.3822 | 35.2247 | NOT_MEASURED | +0.842512 (+2.45%) | NOT_MEASURED |
| ITL p95 (ms) | ms | lower_is_better | 34.6274 | 35.4551 | NOT_MEASURED | +0.827768 (+2.39%) | NOT_MEASURED |
| ITL p99 (ms) | ms | lower_is_better | 34.6274 | 35.4551 | NOT_MEASURED | +0.827768 (+2.39%) | NOT_MEASURED |
| Output Tokens Per Second | output_token/s | higher_is_better | 29.0911 | 28.4113 | NOT_MEASURED | -0.679798 (-2.34%) | NOT_MEASURED |
| Completion Latency Mean (ms) | ms | lower_is_better | 507.829 | 513.248 | NOT_MEASURED | +5.41944 (+1.07%) | NOT_MEASURED |

#### Resources

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Active Detail Bytes | byte | lower_is_better | 0 | 4.60827e+07 | NOT_MEASURED | +4.60827e+07 | NOT_MEASURED |
| Retained Detail Bytes | byte | lower_is_better | 0 | 4.60827e+07 | NOT_MEASURED | +4.60827e+07 | NOT_MEASURED |
| Peak Memory Bytes | byte | lower_is_better | 8.85034e+09 | 8.79468e+09 | NOT_MEASURED | -5.56564e+07 (-0.63%) | NOT_MEASURED |

### mlx-community/Qwen3-32B-4bit / mlx-lm / native-memory / balanced

Task: `combined`. Hardware: `Apple M4 Pro (Mac16,7), 48 GB`. Evidence: `ENGINE_QUALIFIED`.

#### Quality

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Token F1 | fraction | higher_is_better | 0.231164 | 0.231164 | NOT_MEASURED | +0 (+0.00%) | NOT_MEASURED |
| Exact Match | fraction | higher_is_better | 0 | 0 | NOT_MEASURED | +0 | NOT_MEASURED |
| Gold Answer Log Probability | log_probability | higher_is_better | -9.57946 | -9.57946 | NOT_MEASURED | +0 (-0.00%) | NOT_MEASURED |

#### Context

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Visible Tokens | token | lower_is_better | 315.533 | 34.2667 | NOT_MEASURED | -281.267 (-89.14%) | NOT_MEASURED |
| Selected Native K/V Tokens | token | neutral | 0 | 18001.1 | NOT_MEASURED | +18001.1 | NOT_MEASURED |

#### Serving

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| TTFT p50 (ms) | ms | lower_is_better | 524.92 | 490.204 | NOT_MEASURED | -34.7164 (-6.61%) | NOT_MEASURED |
| TTFT p95 (ms) | ms | lower_is_better | 799.757 | 797.258 | NOT_MEASURED | -2.49929 (-0.31%) | NOT_MEASURED |
| TTFT p99 (ms) | ms | lower_is_better | 799.757 | 797.258 | NOT_MEASURED | -2.49929 (-0.31%) | NOT_MEASURED |
| ITL p50 (ms) | ms | lower_is_better | 77.7091 | 79.0556 | NOT_MEASURED | +1.34651 (+1.73%) | NOT_MEASURED |
| ITL p95 (ms) | ms | lower_is_better | 78.2616 | 80.3438 | NOT_MEASURED | +2.08218 (+2.66%) | NOT_MEASURED |
| ITL p99 (ms) | ms | lower_is_better | 78.2616 | 80.3438 | NOT_MEASURED | +2.08218 (+2.66%) | NOT_MEASURED |
| Output Tokens Per Second | output_token/s | higher_is_better | 12.8734 | 12.6266 | NOT_MEASURED | -0.246821 (-1.92%) | NOT_MEASURED |
| Completion Latency Mean (ms) | ms | lower_is_better | 1177.04 | 1183.23 | NOT_MEASURED | +6.18768 (+0.53%) | NOT_MEASURED |

#### Resources

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Active Detail Bytes | byte | lower_is_better | 0 | 7.37324e+07 | NOT_MEASURED | +7.37324e+07 | NOT_MEASURED |
| Retained Detail Bytes | byte | lower_is_better | 0 | 7.37324e+07 | NOT_MEASURED | +7.37324e+07 | NOT_MEASURED |
| Peak Memory Bytes | byte | lower_is_better | 1.91537e+10 | 1.9058e+10 | NOT_MEASURED | -9.56826e+07 (-0.50%) | NOT_MEASURED |

### mlx-community/Qwen3-8B-4bit / mlx-lm / native-memory / balanced

Task: `combined`. Hardware: `Apple M4 Pro (Mac16,7), 48 GB`. Evidence: `ENGINE_QUALIFIED`.

#### Quality

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Token F1 | fraction | higher_is_better | 0.236455 | 0.236455 | NOT_MEASURED | +0 (+0.00%) | NOT_MEASURED |
| Exact Match | fraction | higher_is_better | 0 | 0 | NOT_MEASURED | +0 | NOT_MEASURED |
| Gold Answer Log Probability | log_probability | higher_is_better | -12.2676 | -12.2676 | NOT_MEASURED | +0 (-0.00%) | NOT_MEASURED |

#### Context

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Visible Tokens | token | lower_is_better | 315.533 | 34.2667 | NOT_MEASURED | -281.267 (-89.14%) | NOT_MEASURED |
| Selected Native K/V Tokens | token | neutral | 0 | 10125.6 | NOT_MEASURED | +10125.6 | NOT_MEASURED |

#### Serving

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| TTFT p50 (ms) | ms | lower_is_better | 115.775 | 115.061 | NOT_MEASURED | -0.713958 (-0.62%) | NOT_MEASURED |
| TTFT p95 (ms) | ms | lower_is_better | 183.02 | 231.031 | NOT_MEASURED | +48.0105 (+26.23%) | NOT_MEASURED |
| TTFT p99 (ms) | ms | lower_is_better | 183.02 | 231.031 | NOT_MEASURED | +48.0105 (+26.23%) | NOT_MEASURED |
| ITL p50 (ms) | ms | lower_is_better | 18.787 | 19.4877 | NOT_MEASURED | +0.700714 (+3.73%) | NOT_MEASURED |
| ITL p95 (ms) | ms | lower_is_better | 18.9825 | 20.5185 | NOT_MEASURED | +1.53595 (+8.09%) | NOT_MEASURED |
| ITL p99 (ms) | ms | lower_is_better | 18.9825 | 20.5185 | NOT_MEASURED | +1.53595 (+8.09%) | NOT_MEASURED |
| Output Tokens Per Second | output_token/s | higher_is_better | 53.3045 | 51.1062 | NOT_MEASURED | -2.19834 (-4.12%) | NOT_MEASURED |
| Completion Latency Mean (ms) | ms | lower_is_better | 275.946 | 284.088 | NOT_MEASURED | +8.14217 (+2.95%) | NOT_MEASURED |

#### Resources

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Active Detail Bytes | byte | lower_is_better | 0 | 4.14745e+07 | NOT_MEASURED | +4.14745e+07 | NOT_MEASURED |
| Retained Detail Bytes | byte | lower_is_better | 0 | 4.14745e+07 | NOT_MEASURED | +4.14745e+07 | NOT_MEASURED |
| Peak Memory Bytes | byte | lower_is_better | 5.18009e+09 | 5.13407e+09 | NOT_MEASURED | -4.60226e+07 (-0.89%) | NOT_MEASURED |

### mlx-community/Qwen3-4B-8bit / mlx-lm / native-memory / balanced

Task: `combined`. Hardware: `Apple M4 Pro (Mac16,7), 48 GB`. Evidence: `ENGINE_QUALIFIED`.

#### Quality

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Token F1 | fraction | higher_is_better | 0.131517 | 0.131517 | NEEDS_RUN | +0 (+0.00%) | NEEDS_RUN |
| Exact Match | fraction | higher_is_better | 0 | 0 | NEEDS_RUN | +0 | NEEDS_RUN |
| Gold Answer Log Probability | log_probability | higher_is_better | -20.1859 | -20.1859 | NEEDS_RUN | +0 (-0.00%) | NEEDS_RUN |

#### Context

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Visible Tokens | token | lower_is_better | 396.317 | 33.5167 | NEEDS_RUN | -362.8 (-91.54%) | NEEDS_RUN |
| Selected Native K/V Tokens | token | neutral | 0 | 362.8 | NEEDS_RUN | +362.8 | NEEDS_RUN |

#### Serving

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| TTFT p50 (ms) | ms | lower_is_better | 112.525 | 102.135 | NEEDS_RUN | -10.3901 (-9.23%) | NEEDS_RUN |
| TTFT p95 (ms) | ms | lower_is_better | 130.114 | 115.365 | NEEDS_RUN | -14.749 (-11.34%) | NEEDS_RUN |
| TTFT p99 (ms) | ms | lower_is_better | 136.158 | 127.489 | NEEDS_RUN | -8.66854 (-6.37%) | NEEDS_RUN |
| ITL p50 (ms) | ms | lower_is_better | 19.1811 | 20.0532 | NEEDS_RUN | +0.872125 (+4.55%) | NEEDS_RUN |
| ITL p95 (ms) | ms | lower_is_better | 20.4041 | 20.9611 | NEEDS_RUN | +0.556967 (+2.73%) | NEEDS_RUN |
| ITL p99 (ms) | ms | lower_is_better | 20.6432 | 22.1909 | NEEDS_RUN | +1.5477 (+7.50%) | NEEDS_RUN |
| Output Tokens Per Second | output_token/s | higher_is_better | 43.666 | 42.8374 | NEEDS_RUN | -0.8286 (-1.90%) | NEEDS_RUN |
| Completion Latency Mean (ms) | ms | lower_is_better | 550.365 | 561.034 | NEEDS_RUN | +10.6692 (+1.94%) | NEEDS_RUN |

#### Resources

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Active Detail Bytes | byte | lower_is_better | 0 | 5.3497e+07 | NEEDS_RUN | +5.3497e+07 | NEEDS_RUN |
| Retained Detail Bytes | byte | lower_is_better | 0 | 5.3497e+07 | NEEDS_RUN | +5.3497e+07 | NEEDS_RUN |

#### Routing

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Evidence Recall | fraction | higher_is_better | 0.615972 | 0.615972 | NEEDS_RUN | +0 (+0.00%) | NEEDS_RUN |

### mlx-community/Qwen3-8B-8bit / mlx-lm / native-memory / balanced

Task: `combined`. Hardware: `Apple M4 Pro (Mac16,7), 48 GB`. Evidence: `ENGINE_QUALIFIED`.

#### Quality

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Token F1 | fraction | higher_is_better | 0.144011 | 0.144011 | NO_QUALIFIED_ADAPTER | +0 (+0.00%) | NO_QUALIFIED_ADAPTER |
| Exact Match | fraction | higher_is_better | 0 | 0 | NO_QUALIFIED_ADAPTER | +0 | NO_QUALIFIED_ADAPTER |
| Gold Answer Log Probability | log_probability | higher_is_better | -19.1076 | -19.1076 | NO_QUALIFIED_ADAPTER | +0 (-0.00%) | NO_QUALIFIED_ADAPTER |

#### Context

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Visible Tokens | token | lower_is_better | 396.317 | 33.5167 | NO_QUALIFIED_ADAPTER | -362.8 (-91.54%) | NO_QUALIFIED_ADAPTER |
| Selected Native K/V Tokens | token | neutral | 0 | 362.8 | NO_QUALIFIED_ADAPTER | +362.8 | NO_QUALIFIED_ADAPTER |

#### Serving

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| TTFT p50 (ms) | ms | lower_is_better | 186.881 | 176.211 | NO_QUALIFIED_ADAPTER | -10.6696 (-5.71%) | NO_QUALIFIED_ADAPTER |
| TTFT p95 (ms) | ms | lower_is_better | 232.033 | 208.936 | NO_QUALIFIED_ADAPTER | -23.0971 (-9.95%) | NO_QUALIFIED_ADAPTER |
| TTFT p99 (ms) | ms | lower_is_better | 249.496 | 212.698 | NO_QUALIFIED_ADAPTER | -36.7977 (-14.75%) | NO_QUALIFIED_ADAPTER |
| ITL p50 (ms) | ms | lower_is_better | 34.8123 | 35.6742 | NO_QUALIFIED_ADAPTER | +0.861843 (+2.48%) | NO_QUALIFIED_ADAPTER |
| ITL p95 (ms) | ms | lower_is_better | 36.4766 | 36.8092 | NO_QUALIFIED_ADAPTER | +0.332536 (+0.91%) | NO_QUALIFIED_ADAPTER |
| ITL p99 (ms) | ms | lower_is_better | 36.8347 | 37.1582 | NO_QUALIFIED_ADAPTER | +0.323467 (+0.88%) | NO_QUALIFIED_ADAPTER |
| Output Tokens Per Second | output_token/s | higher_is_better | 24.4728 | 24.3286 | NO_QUALIFIED_ADAPTER | -0.144183 (-0.59%) | NO_QUALIFIED_ADAPTER |
| Completion Latency Mean (ms) | ms | lower_is_better | 981.993 | 987.449 | NO_QUALIFIED_ADAPTER | +5.4559 (+0.56%) | NO_QUALIFIED_ADAPTER |

#### Resources

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Active Detail Bytes | byte | lower_is_better | 0 | 5.3497e+07 | NO_QUALIFIED_ADAPTER | +5.3497e+07 | NO_QUALIFIED_ADAPTER |
| Retained Detail Bytes | byte | lower_is_better | 0 | 5.3497e+07 | NO_QUALIFIED_ADAPTER | +5.3497e+07 | NO_QUALIFIED_ADAPTER |

#### Routing

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Evidence Recall | fraction | higher_is_better | 0.615972 | 0.615972 | NO_QUALIFIED_ADAPTER | +0 (+0.00%) | NO_QUALIFIED_ADAPTER |

### mlx-community/Qwen3-14B-8bit / mlx-lm / native-memory / balanced

Task: `combined`. Hardware: `Apple M4 Pro (Mac16,7), 48 GB`. Evidence: `ENGINE_QUALIFIED`.

#### Quality

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Token F1 | fraction | higher_is_better | 0.149402 | 0.149402 | NO_QUALIFIED_ADAPTER | +0 (+0.00%) | NO_QUALIFIED_ADAPTER |
| Exact Match | fraction | higher_is_better | 0 | 0 | NO_QUALIFIED_ADAPTER | +0 | NO_QUALIFIED_ADAPTER |
| Gold Answer Log Probability | log_probability | higher_is_better | -17.4831 | -17.4831 | NO_QUALIFIED_ADAPTER | +0 (-0.00%) | NO_QUALIFIED_ADAPTER |

#### Context

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Visible Tokens | token | lower_is_better | 396.317 | 33.5167 | NO_QUALIFIED_ADAPTER | -362.8 (-91.54%) | NO_QUALIFIED_ADAPTER |
| Selected Native K/V Tokens | token | neutral | 0 | 362.8 | NO_QUALIFIED_ADAPTER | +362.8 | NO_QUALIFIED_ADAPTER |

#### Serving

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| TTFT p50 (ms) | ms | lower_is_better | 335.664 | 327.575 | NO_QUALIFIED_ADAPTER | -8.08944 (-2.41%) | NO_QUALIFIED_ADAPTER |
| TTFT p95 (ms) | ms | lower_is_better | 414.035 | 393.91 | NO_QUALIFIED_ADAPTER | -20.1256 (-4.86%) | NO_QUALIFIED_ADAPTER |
| TTFT p99 (ms) | ms | lower_is_better | 430.039 | 406.098 | NO_QUALIFIED_ADAPTER | -23.9403 (-5.57%) | NO_QUALIFIED_ADAPTER |
| ITL p50 (ms) | ms | lower_is_better | 63.2739 | 63.7036 | NO_QUALIFIED_ADAPTER | +0.429721 (+0.68%) | NO_QUALIFIED_ADAPTER |
| ITL p95 (ms) | ms | lower_is_better | 64.9334 | 65.6904 | NO_QUALIFIED_ADAPTER | +0.75698 (+1.17%) | NO_QUALIFIED_ADAPTER |
| ITL p99 (ms) | ms | lower_is_better | 65.2931 | 66.3538 | NO_QUALIFIED_ADAPTER | +1.06065 (+1.62%) | NO_QUALIFIED_ADAPTER |
| Output Tokens Per Second | output_token/s | higher_is_better | 13.3873 | 13.3466 | NO_QUALIFIED_ADAPTER | -0.0407081 (-0.30%) | NO_QUALIFIED_ADAPTER |
| Completion Latency Mean (ms) | ms | lower_is_better | 1795.02 | 1800.26 | NO_QUALIFIED_ADAPTER | +5.23693 (+0.29%) | NO_QUALIFIED_ADAPTER |

#### Resources

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Active Detail Bytes | byte | lower_is_better | 0 | 5.94412e+07 | NO_QUALIFIED_ADAPTER | +5.94412e+07 | NO_QUALIFIED_ADAPTER |
| Retained Detail Bytes | byte | lower_is_better | 0 | 5.94412e+07 | NO_QUALIFIED_ADAPTER | +5.94412e+07 | NO_QUALIFIED_ADAPTER |

#### Routing

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Evidence Recall | fraction | higher_is_better | 0.615972 | 0.615972 | NO_QUALIFIED_ADAPTER | +0 (+0.00%) | NO_QUALIFIED_ADAPTER |

### mlx-community/Qwen3-8B-6bit / mlx-lm / native-memory / balanced

Task: `combined`. Hardware: `Apple M4 Pro (Mac16,7), 48 GB`. Evidence: `ENGINE_QUALIFIED`.

#### Quality

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Token F1 | fraction | higher_is_better | 0.158232 | 0.158232 | NO_QUALIFIED_ADAPTER | +0 (+0.00%) | NO_QUALIFIED_ADAPTER |
| Exact Match | fraction | higher_is_better | 0 | 0 | NO_QUALIFIED_ADAPTER | +0 | NO_QUALIFIED_ADAPTER |
| Gold Answer Log Probability | log_probability | higher_is_better | -18.8568 | -18.8568 | NO_QUALIFIED_ADAPTER | +0 (-0.00%) | NO_QUALIFIED_ADAPTER |

#### Context

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Visible Tokens | token | lower_is_better | 396.317 | 33.5167 | NO_QUALIFIED_ADAPTER | -362.8 (-91.54%) | NO_QUALIFIED_ADAPTER |
| Selected Native K/V Tokens | token | neutral | 0 | 362.8 | NO_QUALIFIED_ADAPTER | +362.8 | NO_QUALIFIED_ADAPTER |

#### Serving

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| TTFT p50 (ms) | ms | lower_is_better | 174.612 | 175.757 | NO_QUALIFIED_ADAPTER | +1.14525 (+0.66%) | NO_QUALIFIED_ADAPTER |
| TTFT p95 (ms) | ms | lower_is_better | 224.646 | 203.595 | NO_QUALIFIED_ADAPTER | -21.0517 (-9.37%) | NO_QUALIFIED_ADAPTER |
| TTFT p99 (ms) | ms | lower_is_better | 227.007 | 219.186 | NO_QUALIFIED_ADAPTER | -7.82108 (-3.45%) | NO_QUALIFIED_ADAPTER |
| ITL p50 (ms) | ms | lower_is_better | 27.1323 | 28.1637 | NO_QUALIFIED_ADAPTER | +1.03139 (+3.80%) | NO_QUALIFIED_ADAPTER |
| ITL p95 (ms) | ms | lower_is_better | 28.9412 | 29.379 | NO_QUALIFIED_ADAPTER | +0.437819 (+1.51%) | NO_QUALIFIED_ADAPTER |
| ITL p99 (ms) | ms | lower_is_better | 29.4701 | 31.0573 | NO_QUALIFIED_ADAPTER | +1.58718 (+5.39%) | NO_QUALIFIED_ADAPTER |
| Output Tokens Per Second | output_token/s | higher_is_better | 29.8239 | 29.3964 | NO_QUALIFIED_ADAPTER | -0.427475 (-1.43%) | NO_QUALIFIED_ADAPTER |
| Completion Latency Mean (ms) | ms | lower_is_better | 806.747 | 818.315 | NO_QUALIFIED_ADAPTER | +11.5686 (+1.43%) | NO_QUALIFIED_ADAPTER |

#### Resources

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Active Detail Bytes | byte | lower_is_better | 0 | 5.3497e+07 | NO_QUALIFIED_ADAPTER | +5.3497e+07 | NO_QUALIFIED_ADAPTER |
| Retained Detail Bytes | byte | lower_is_better | 0 | 5.3497e+07 | NO_QUALIFIED_ADAPTER | +5.3497e+07 | NO_QUALIFIED_ADAPTER |

#### Routing

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Evidence Recall | fraction | higher_is_better | 0.615972 | 0.615972 | NO_QUALIFIED_ADAPTER | +0 (+0.00%) | NO_QUALIFIED_ADAPTER |

### mlx-community/Llama-3.2-1B-Instruct-8bit / mlx-lm / native-memory / balanced

Task: `combined`. Hardware: `Apple M4 Pro (Mac16,7), 48 GB`. Evidence: `ENGINE_QUALIFIED`.

#### Quality

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Token F1 | fraction | higher_is_better | 0.125357 | 0.125357 | NO_QUALIFIED_ADAPTER | +0 (+0.00%) | NO_QUALIFIED_ADAPTER |
| Exact Match | fraction | higher_is_better | 0 | 0 | NO_QUALIFIED_ADAPTER | +0 | NO_QUALIFIED_ADAPTER |
| Gold Answer Log Probability | log_probability | higher_is_better | -15.0841 | -15.0841 | NO_QUALIFIED_ADAPTER | +0 (-0.00%) | NO_QUALIFIED_ADAPTER |

#### Context

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Visible Tokens | token | lower_is_better | 390.867 | 33.25 | NO_QUALIFIED_ADAPTER | -357.617 (-91.49%) | NO_QUALIFIED_ADAPTER |
| Selected Native K/V Tokens | token | neutral | 0 | 357.617 | NO_QUALIFIED_ADAPTER | +357.617 | NO_QUALIFIED_ADAPTER |

#### Serving

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| TTFT p50 (ms) | ms | lower_is_better | 31.8044 | 29.6227 | NO_QUALIFIED_ADAPTER | -2.18167 (-6.86%) | NO_QUALIFIED_ADAPTER |
| TTFT p95 (ms) | ms | lower_is_better | 36.076 | 31.4482 | NO_QUALIFIED_ADAPTER | -4.62779 (-12.83%) | NO_QUALIFIED_ADAPTER |
| TTFT p99 (ms) | ms | lower_is_better | 87.1491 | 92.627 | NO_QUALIFIED_ADAPTER | +5.47783 (+6.29%) | NO_QUALIFIED_ADAPTER |
| ITL p50 (ms) | ms | lower_is_better | 6.13011 | 6.26277 | NO_QUALIFIED_ADAPTER | +0.132659 (+2.16%) | NO_QUALIFIED_ADAPTER |
| ITL p95 (ms) | ms | lower_is_better | 6.66693 | 6.42713 | NO_QUALIFIED_ADAPTER | -0.239795 (-3.60%) | NO_QUALIFIED_ADAPTER |
| ITL p99 (ms) | ms | lower_is_better | 7.20762 | 7.41378 | NO_QUALIFIED_ADAPTER | +0.206161 (+2.86%) | NO_QUALIFIED_ADAPTER |
| Output Tokens Per Second | output_token/s | higher_is_better | 138.47 | 138.864 | NO_QUALIFIED_ADAPTER | +0.394598 (+0.28%) | NO_QUALIFIED_ADAPTER |
| Completion Latency Mean (ms) | ms | lower_is_better | 173.813 | 173.392 | NO_QUALIFIED_ADAPTER | -0.421136 (-0.24%) | NO_QUALIFIED_ADAPTER |

#### Resources

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Active Detail Bytes | byte | lower_is_better | 0 | 1.17184e+07 | NO_QUALIFIED_ADAPTER | +1.17184e+07 | NO_QUALIFIED_ADAPTER |
| Retained Detail Bytes | byte | lower_is_better | 0 | 1.17184e+07 | NO_QUALIFIED_ADAPTER | +1.17184e+07 | NO_QUALIFIED_ADAPTER |

#### Routing

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Evidence Recall | fraction | higher_is_better | 0.615972 | 0.615972 | NO_QUALIFIED_ADAPTER | +0 (+0.00%) | NO_QUALIFIED_ADAPTER |

### mlx-community/gemma-3-1b-it-8bit / mlx-lm / native-memory / balanced

Task: `combined`. Hardware: `Apple M4 Pro (Mac16,7), 48 GB`. Evidence: `ENGINE_QUALIFIED`.

#### Quality

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Token F1 | fraction | higher_is_better | 0.0735174 | 0.0699096 | NO_QUALIFIED_ADAPTER | -0.0036078 (-4.91%) | NO_QUALIFIED_ADAPTER |
| Exact Match | fraction | higher_is_better | 0 | 0 | NO_QUALIFIED_ADAPTER | +0 | NO_QUALIFIED_ADAPTER |
| Gold Answer Log Probability | log_probability | higher_is_better | -23.0353 | -20.6684 | NO_QUALIFIED_ADAPTER | +2.36697 (-10.28%) | NO_QUALIFIED_ADAPTER |

#### Context

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Visible Tokens | token | lower_is_better | 395.317 | 35.1667 | NO_QUALIFIED_ADAPTER | -360.15 (-91.10%) | NO_QUALIFIED_ADAPTER |
| Selected Native K/V Tokens | token | neutral | 0 | 360.15 | NO_QUALIFIED_ADAPTER | +360.15 | NO_QUALIFIED_ADAPTER |

#### Serving

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| TTFT p50 (ms) | ms | lower_is_better | 32.6564 | 31.0044 | NO_QUALIFIED_ADAPTER | -1.65196 (-5.06%) | NO_QUALIFIED_ADAPTER |
| TTFT p95 (ms) | ms | lower_is_better | 39.1253 | 46.6731 | NO_QUALIFIED_ADAPTER | +7.54779 (+19.29%) | NO_QUALIFIED_ADAPTER |
| TTFT p99 (ms) | ms | lower_is_better | 46.3028 | 58.7391 | NO_QUALIFIED_ADAPTER | +12.4364 (+26.86%) | NO_QUALIFIED_ADAPTER |
| ITL p50 (ms) | ms | lower_is_better | 5.91003 | 6.36293 | NO_QUALIFIED_ADAPTER | +0.452904 (+7.66%) | NO_QUALIFIED_ADAPTER |
| ITL p95 (ms) | ms | lower_is_better | 6.44007 | 7.00265 | NO_QUALIFIED_ADAPTER | +0.562576 (+8.74%) | NO_QUALIFIED_ADAPTER |
| ITL p99 (ms) | ms | lower_is_better | 7.41584 | 7.40068 | NO_QUALIFIED_ADAPTER | -0.0151522 (-0.20%) | NO_QUALIFIED_ADAPTER |
| Output Tokens Per Second | output_token/s | higher_is_better | 141.269 | 133.394 | NO_QUALIFIED_ADAPTER | -7.87498 (-5.57%) | NO_QUALIFIED_ADAPTER |
| Completion Latency Mean (ms) | ms | lower_is_better | 170.376 | 180.421 | NO_QUALIFIED_ADAPTER | +10.0445 (+5.90%) | NO_QUALIFIED_ADAPTER |

#### Resources

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Active Detail Bytes | byte | lower_is_better | 0 | 9.58863e+06 | NO_QUALIFIED_ADAPTER | +9.58863e+06 | NO_QUALIFIED_ADAPTER |
| Retained Detail Bytes | byte | lower_is_better | 0 | 9.58863e+06 | NO_QUALIFIED_ADAPTER | +9.58863e+06 | NO_QUALIFIED_ADAPTER |

#### Routing

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Evidence Recall | fraction | higher_is_better | 0.615972 | 0.615972 | NO_QUALIFIED_ADAPTER | +0 (+0.00%) | NO_QUALIFIED_ADAPTER |

### Qwen/Qwen2.5-1.5B-Instruct / huggingface_eager / native-memory / balanced

Task: `combined`. Hardware: `NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB`. Evidence: `ENGINE_QUALIFIED`.

#### Quality

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Token F1 | fraction | higher_is_better | 0.399111 | 0.0178571 | NO_QUALIFIED_ADAPTER | -0.381254 (-95.53%) | NO_QUALIFIED_ADAPTER |
| Exact Match | fraction | higher_is_better | 0.333333 | 0 | NO_QUALIFIED_ADAPTER | -0.333333 (-100.00%) | NO_QUALIFIED_ADAPTER |

#### Context

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Visible Tokens | token | lower_is_better | 404.6 | 33.6 | NO_QUALIFIED_ADAPTER | -371 (-91.70%) | NO_QUALIFIED_ADAPTER |
| Selected Native K/V Tokens | token | neutral | 0 | 371 | NO_QUALIFIED_ADAPTER | +371 | NO_QUALIFIED_ADAPTER |

#### Serving

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Output Tokens Per Second | output_token/s | higher_is_better | 2.30576 | 2.09692 | NO_QUALIFIED_ADAPTER | -0.208833 (-9.06%) | NO_QUALIFIED_ADAPTER |
| Completion Latency Mean (ms) | ms | lower_is_better | 3772.26 | 11448.4 | NO_QUALIFIED_ADAPTER | +7676.19 (+203.49%) | NO_QUALIFIED_ADAPTER |

#### Resources

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Active Detail Bytes | byte | lower_is_better | 0 | 1.06373e+07 | NO_QUALIFIED_ADAPTER | +1.06373e+07 | NO_QUALIFIED_ADAPTER |
| Retained Detail Bytes | byte | lower_is_better | 0 | 1.06373e+07 | NO_QUALIFIED_ADAPTER | +1.06373e+07 | NO_QUALIFIED_ADAPTER |

#### Routing

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Evidence Recall | fraction | higher_is_better | 0.75 | 0.75 | NO_QUALIFIED_ADAPTER | +0 (+0.00%) | NO_QUALIFIED_ADAPTER |


## Interpretation

The adaptor-bundle column is intentionally distinct from generic PRA. A published bundle may contain only structural mapping and profile metadata, or may include an opt-in learned router. A bundle cell becomes measured only when the immutable bundle revision was resolved during the run.

Routing-only recall is reported in each model card's research diagnostics and does not substitute for answer quality, TTFT, ITL, throughput, or memory measurements.
