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
| Parallel scheduling | Experimental harness responsibility |
| General DAG and peer visibility | Not implemented |
| Cross-session memory | Out of scope; Paper 10 |

The safe default is a hard boundary: ancestor and descendant visibility are
`none`, `UNKNOWN` effects cannot be reused, and mutable external resources need
an explicit validator.

## Source map

- `src/pra_hf/subagent_context.py`: generic lineage, visibility, consistency,
  validity, reuse, and native-state receipts.
- `src/pra_hf/subagent_harness.py`: conventional harness lifecycle and concrete
  execution callbacks.
- `tests/test_subagent_context.py`: correctness and isolation contracts.
- `experiments/paper9_subagents/`: controlled SCRB experiments and summaries.

