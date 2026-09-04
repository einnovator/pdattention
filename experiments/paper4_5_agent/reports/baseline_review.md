# No-PRA baseline review

## Decision

No existing model/agent row is eligible for a PRA efficacy comparison. The
gateway fixture rows establish API and tool compatibility only; the official
Terminal-Bench rows establish negative no-PRA diagnostics.

| Harness / model | Official cohort | Result | Disposition |
| --- | --- | ---: | --- |
| OpenCode / Qwen3-14B | Terminal-Bench 2.1, five frozen tasks | 0/5 | `BLOCKED`: quality floor |
| OpenCode / Qwen2.5-Coder-7B | Terminal-Bench 2.1, two tasks | 0/2 | `BLOCKED`: tool calls rendered as text |
| OpenCode / Qwen3-Coder-30B-A3B | Terminal-Bench 2.1, one task | 0/1 | `BLOCKED`: 4/6 verifier checks is not task success |
| Pi / Qwen3-14B | Terminal-Bench 2.1, one task | 0/1 | `BLOCKED`: quality floor |
| OpenHands / Qwen3-14B | Terminal-Bench 2.1, one task | 0/1 | `BLOCKED`: continuation loop |

The primary next target is FIM-14B with the published R2E-Gym/SWE-bench
Verified configuration (`29.20%`, three-seed mean). The available RTX 5060
Laptop has only 8 GiB VRAM, so it cannot host the BF16 14B model. The 48 GiB
M4 can host it but does not provide the published vLLM/CUDA engine. Therefore
an Apple/quantized result must remain `BASELINE_ATTEMPTED`.

The first infrastructure qualification uses FIM-7B Q4_K_M through llama.cpp
on the M4, the unmodified R2E-Gym agent on the NVIDIA/WSL host, and official
SWE-bench Docker grading. It deliberately runs without the PRA gateway. Its
engine, precision, and smoke-cohort differences are recorded in the receipt,
so it cannot unlock gateway or PRA conditions.

That qualification completed on September 4, 2026. The official SWE-bench
grader resolved `django__django-13821` and did not resolve
`sympy__sympy-19954`: `1/2` (`50.0%`, Wilson 95% interval `9.5-90.5%`) with
zero grader errors. The resolved task terminated naturally after 31 model
calls; the unresolved task reached the 40-step limit. Together they made 72
model calls, consumed 1,123,772 cumulative prompt tokens and 11,878 output
tokens, and took 607.4 seconds in the agent trajectories. These are useful
path and workload measurements, not an estimate of population-level accuracy.
The campaign records the result as `BASELINE_ATTEMPTED`; PRA interpretation
remains locked.
