# Subagent Context Reuse Benchmark (SCRB)

SCRB separates ordinary harness memoization from PRA payload reuse and native
K/V reuse. The first experiment is an event-driven controlled simulation with
direct checks against the Paper 9 runtime implementation. Its injected cost
model is recorded in `summary.json`; values are not live model or engine
latencies.

The scaling grid contains five deterministic workload seeds, shared records of
2K/8K/32K/64K tokens, fan-out 2/4/8/16, and reuse probabilities from 0.25 to
1.0. It compares isolated children, harness memoization, PRA without visibility,
PRA payload reuse, and compatible native K/V reuse.

The safety experiment compares coarse invalidation, resource-level selective
invalidation, and deliberately unsafe reuse. The completed-child experiment
compares summary-only return, full transcript copying, transcript replay, and
PRA descendant routing.

Run from the repository root:

```powershell
$env:PYTHONPATH = "src"
python experiments/paper9_subagents/run_scrb.py `
  --output docs/papers/shared/results/paper9_subagents/scrb_v1
```

Live-engine and natural-source phases remain separate from SCRB. They must not
reuse the controlled simulator's latency numbers as measurements, and the
tracked-source cohort is not an autonomous coding-agent success benchmark.

The tracked-source natural workload exercises the real callback scheduler and
completed-child selectors without claiming autonomous coding success:

```powershell
$env:PYTHONPATH = "src"
python experiments/paper9_subagents/run_natural_repository_workload.py
```

On Apple Silicon, the live MLX benchmark freezes source/query tokens and
compares one-shot text, host split-prefill, memoized text re-prefill, typed
record re-prefill, and immutable native K/V reuse:

```bash
PYTHONPATH=src python experiments/paper9_subagents/run_mlx_live_reuse.py \
  --shared-tokens 512,2048,8192,32768 \
  --fanouts 1,2,4,8,16 \
  --seeds 11,23,37,71,101 \
  --output docs/papers/shared/results/paper9_subagents/mlx_live_v1
```

An instrumentation-only audit can be run against Paper 4.5 trajectory exports:

```powershell
python experiments/paper9_subagents/analyze_agent_traces.py `
  --input D:\git\rd\pdattention-paper4\docs\papers\shared\results\paper4_5_runtime_productization\coding_agents `
  --output docs/papers/shared/results/paper9_subagents/natural_trace_opportunity.json
```

That artifact is deliberately labeled as a single-agent opportunity audit. It
does not claim that cross-agent reuse occurred in those historical runs.
