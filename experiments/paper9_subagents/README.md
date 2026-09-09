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

The follow-up consumption factorial freezes the 16 fielded-BM25 top-1 records
from that transfer artifact, including its two routing misses. Every model and
arm receives byte-identical selected text. Only presentation changes: direct
injection, provenance/attribution-aware presentation, or mandatory read-only
tool verification against the embedded pinned candidate snapshot. Output is
checkpointed per model, condition, repository, and query. The run manifest
captures each Ollama model digest before execution, refuses an incompatible
resume, and verifies that those digests remain unchanged through completion:

```bash
PYTHONPATH=src:. python -m experiments.paper9_subagents.run_consumption_factorial \
  --manifest experiments/paper9_subagents/benchmarks/consumption_factorial_v1.json \
  --output .runs/consumption_factorial_v1 \
  --base-url http://127.0.0.1:11435 \
  --models qwen3-coder:30b,qwen3:14b,gemma3:4b-it-qat
```

The direct-MLX lifecycle follow-up uses session-scoped state identities and
exercises simultaneous model-forward requests, cooperative cancellation before
model submission, scoped session termination, inactive LRU eviction, and
device-to-host-to-device state transitions. The supplied Hub revision is
resolved to a concrete snapshot before MLX-LM loads it. The experiment must be
described as direct model-forward evidence, not HTTP cancellation or mid-kernel
preemption:

```bash
PYTHONPATH=src:. python -m experiments.paper9_subagents.run_mlx_lifecycle \
  --model mlx-community/Qwen3-0.6B-4bit --revision 73e3e38d \
  --shared-tokens 2048 --repetitions 5 \
  --output .runs/mlx_lifecycle_m5_v1
```

The remote launchers preserve other campaigns: `run_mlx_lifecycle_medium.sh`
waits for the Paper 3.3 and Paper 4.5 controllers on the M5, while
`run_mlx_lifecycle_bigmac_after_factorial.sh` waits for the digest-pinned
consumption factorial to finish successfully on the M4. Running both produces
the primary result sooner and retains a cross-host replication.
The Big Mac factorial launcher uses the same explicit Python 3.12 environment
as its lifecycle run; do not fall back to the macOS system Python.

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
