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

The v2 cohort preserves the original whole-record lexical and learned baselines
and adds a zero-model fielded BM25 policy over resource identity, leading
documentation, and bounded source windows. Its output is written to
`natural_repository_v2`; the tracked `natural_repository_v1` artifact remains
the immutable 60% baseline.

The autonomous repository campaign crosses sequential/parallel scheduling with
isolated/completed-peer context. Model children choose their own read-only
repository tools; the harness records path accuracy, model tokens, routed peer
records, and logical versus physical tool calls:

```bash
PYTHONPATH=src python -m experiments.paper9_subagents.run_autonomous_repository_campaign \
  --repo . --output docs/papers/shared/results/paper9_subagents/autonomous_repository_v1/seeds/seed11 \
  --model qwen3-coder:30b --seeds 11 --max-workers 4 --max-steps 7
```

Seeds 11, 23, and 37 were checkpointed independently and combined with
`aggregate_autonomous_campaign.py`. The aggregate records the M4 host, Ollama
and model versions, base repository revision, source hashes, paired speedups,
and paired selection-versus-consumption deltas.

The frozen router-transfer cohort fits the learned baseline only on Paper 9's
development repository, then evaluates unchanged policies on two unrelated
repositories pinned by commit and file hash:

```powershell
$env:PYTHONPATH = "src"
python -m experiments.paper9_subagents.run_router_transfer `
  --repo dynaspike=D:\git\rd\dynaspike `
  --repo cognitive_coprocessors=D:\git\rd\cognitive_coprocessors
```

On Apple Silicon, the live MLX benchmark freezes source/query tokens and
compares one-shot text, host split-prefill, memoized text re-prefill, typed
record re-prefill, and immutable native K/V reuse. The host, memoized, and
typed-record controls all use the same serving-style split-prefill path. The
one-shot condition is diagnostic only because MLX kernel/chunking choices can
change finite-precision logits even when the token sequence is identical:

```bash
PYTHONPATH=src python experiments/paper9_subagents/run_mlx_live_reuse.py \
  --shared-tokens 512,2048,8192,32768 \
  --fanouts 1,2,4,8,16 \
  --seeds 11,23,37,71,101 \
  --output docs/papers/shared/results/paper9_subagents/mlx_live_m4_final
```

For 32K sources, omit the diagnostic one-shot arm and restrict the measured
fan-out when device memory is constrained:

```bash
PYTHONPATH=src python experiments/paper9_subagents/run_mlx_live_reuse.py \
  --shared-tokens 32768 --fanouts 1,2,4 \
  --seeds 11,23,37,71,101 --omit-one-shot \
  --output docs/papers/shared/results/paper9_subagents/mlx_32
```

The current public MLX cache interface forms a transient concatenated
shared-plus-local attention view. The retained native arrays are shared across
children, but this benchmark must not be described as segmented zero-copy
attention. `run_mlx_generation_parity.py` separately checks greedy output
parity, and `analyze_mlx_cross_host.py` compares the same protocol across
memory-capacity regimes.

An instrumentation-only audit can be run against Paper 4.5 trajectory exports:

```powershell
python experiments/paper9_subagents/analyze_agent_traces.py `
  --input D:\git\rd\pdattention-paper4\docs\papers\shared\results\paper4_5_runtime_productization\coding_agents `
  --output docs/papers/shared/results/paper9_subagents/natural_trace_opportunity.json
```

That artifact is deliberately labeled as a single-agent opportunity audit. It
does not claim that cross-agent reuse occurred in those historical runs.
