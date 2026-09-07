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
  --output docs/papers/shared/results/paper9_subagents/mlx_live_v1
```
