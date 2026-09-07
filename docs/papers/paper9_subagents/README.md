# Paper 9: Selectively Permeable Subagent Contexts

Paper 9 extends the Paper 8 typed-record agent substrate from one agent stream
to a lineage graph of parent and child streams. Concrete tool semantics remain
in harness adapters; the PRA runtime sees only typed records, visibility,
generic effects, normalized resource identities, causal order, and native-state
compatibility receipts.

## Current capability boundary

| Capability | Status |
| --- | --- |
| Spawn and inspect child streams | Supported |
| Stop and resume children | Supported |
| Typed lifecycle and reuse-decision records | Supported |
| Ancestor and completed-descendant visibility | Experimental |
| Exact payload/tool-result reuse | Experimental |
| Resource-level invalidation | Experimental |
| External validator hooks | Experimental |
| Compatible native K/V callback | Experimental |
| Sequential/parallel callback scheduling | Experimental |
| Explicit DAG joins and peer visibility | Experimental |
| Validity-first lexical/learned/oracle descendant routing | Experimental |
| Live MLX native K/V reuse | Experimental benchmark |
| Cross-session memory | Out of scope; Paper 10 |

The safe default is a hard boundary: ancestor and descendant visibility are
`none`, `UNKNOWN` effects cannot be reused, and mutable external resources need
an explicit validator.

## Source map

- `src/pra_hf/subagent_context.py`: generic lineage, visibility, consistency,
  validity, reuse, and native-state receipts.
- `src/pra_hf/subagent_harness.py`: conventional harness lifecycle and concrete
  execution callbacks, including bounded sequential/parallel fan-out.
- `src/pra_hf/subagent_routing.py`: validity-first completed-child selectors.
- `src/pra_hf/subagent_mlx_native.py`: narrow live MLX native-state port.
- `tests/test_subagent_context.py`: correctness and isolation contracts.
- `experiments/paper9_subagents/`: controlled SCRB experiments and summaries.

The callback scheduler intentionally accepts an arbitrary child runner instead
of embedding an LLM loop. A harness integration wraps its existing child loop:

```python
results = harness.run_subagents(
    parent_agent_uuid,
    child_specs,
    lambda child, pra: existing_agent_runner(child, pra),
    parallel=True,
    max_workers=8,
)
```

Run the natural tracked-source cohort locally:

```powershell
$env:PYTHONPATH = "src"
python experiments/paper9_subagents/run_natural_repository_workload.py
```

Run the live MLX prefill/native-reuse surface on Apple Silicon:

```bash
PYTHONPATH=src python experiments/paper9_subagents/run_mlx_live_reuse.py \
  --shared-tokens 512,2048,8192,32768 \
  --fanouts 1,2,4,8,16 \
  --seeds 11,23,37,71,101 \
  --output docs/papers/shared/results/paper9_subagents/mlx_live_m4_final
```

## Current evidence

The five-seed tracked-source cohort executes both sequential and parallel
schedulers. Each run makes 35 logical file reads but only 13 physical reads;
the remaining 22 are valid ancestor reuses. Explicit two-parent DAG joins and
completed-only peer visibility are observed in every run. On held-out
descendant queries, lexical and learned top-1 routing each recover 60% of the
relevant records, while oracle top-1 recovers 100%. The current learned linear
ranker therefore does not close the selection gap.

The live MLX sweep uses immutable Qwen3-0.6B-4bit revision `73e3e38d`. On an
M4 Pro with 48 GB, fan-out 16 yields `5.03x`, `8.74x`, and `11.69x` amortized
speedup for 512, 2K, and 8K shared records. At 32K and fan-out four, the same
host yields `3.89x`; a 16 GB M5 yields `0.48x` despite exact logits
and the same 75% physical-token reduction. The retained K/V is shared, but the
public MLX cache seam still constructs transient concatenated attention views.
The result is therefore a residency-sensitive implementation boundary, not a
claim that native reuse is uniformly faster.

A separate five-seed, 16-token greedy generation check at 8K matches all five
host split-prefill sequences exactly with zero observed logit delta. These are
direct model-forward measurements, not HTTP TTFT or autonomous-agent quality.
