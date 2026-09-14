# Paper 8.5 matched retirement factorial

Evidence class: frozen one-decision policy comparison; not autonomous task
accuracy.

At the same three oracle decisions, this directory compares:

- `whole_record_h1h3`: H1+H3 exclusion realized by deleting complete causal
  groups;
- `payload_stub_h1h3`: identical negative decisions with model-visible
  observation receipts when the tokenizer size gate permits them;
- `matched_recency_to_whole`: matched token-tail recency constrained not to
  exceed the whole-record arm's materialized-token count;
- `matched_recency_to_stub`: the same recency control constrained by the
  receipt arm.

The recency ceiling rounds down when the next atomic retained unit does not fit,
and the unused budget is reported in each artifact. A receipt arm may retain
FULL when its receipt is not smaller than the original payload; this is a
fail-closed size-gate outcome, not a hidden zero-saving success.

Fresh FULL probes for Tasks 2 and 3 are included as
`full_contemporaneous.json`. Task 1's corresponding probe is
`../oracle_headroom_v1/task01_d24/full_c.json`. Exact command is compared with
the contemporaneous FULL probe in the paper; semantic transition is separately
compared with the pinned repeat-qualified FULL reference.

