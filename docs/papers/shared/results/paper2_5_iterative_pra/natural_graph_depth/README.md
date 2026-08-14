# MuSiQue and 2Wiki dataset audit

This directory records the frozen natural-graph cohort used by Paper 2.5. Raw
archives and extracted datasets remain regenerable, ignored local inputs.

## MuSiQue

- Official source: <https://github.com/StonyBrookNLP/musique>
- Release: v1.0 MuSiQue-Ans development split, CC BY 4.0.
- Development rows: 2,417 (1,252 two-hop; 760 three-hop; 405 four-hop).
- The adapter preserves decomposition rows verbatim. Only explicit `#n`
  references become annotated graph edges.

## 2WikiMultiHopQA

- Official source: <https://github.com/Alab-NII/2wikimultihop>
- Release: April 7, 2021 segmentation-fixed archive, Apache-2.0.
- Development rows: 12,576. The official test labels are empty, so a stable
  identity hash partitions official dev into calibration and held-out subsets.
- Supporting facts locate source sentences. Entity-ID joins define edges when
  available; normalized lexical joins are marked as derived rather than ground
  truth. Full-dev mapping is 30,911/31,120
  (99.33%); unresolved rows are excluded.

## Frozen cohort

The cohort has 84 examples (42
validation, 42 held out). It contains 36
MuSiQue examples balanced over true depth and 48 2Wiki examples balanced over
question type. `selected_raw_annotations.jsonl` keeps source annotations separate
from the later tokenizer/chunk mapping artifacts.

## Frozen results

The validation-selected oracle-root condition is `K=6`, `B=16`, `H=4` at
128-token parents. Held-out node/complete recovery is 0.981/0.944 on MuSiQue
and 0.979/0.958 on 2Wiki, but the condition activates a counterfactual 51.3%
and 77.0% of source tokens. Under `B=6`, completion is 0.444 and 0.917.

The strict held-out 2Wiki path set contains 25 novel-parent transitions from
17 paths. Transition R@4/R@6/R@8 is 0.720/0.880/1.000; complete-path survival
is 0.588/0.824/1.000. MuSiQue oracle-parent count rises from 4.08 at D=2 to
7.67 at D=4, but recovery saturates at search depth two. This demonstrates a
dense local evidence neighborhood, not faithful four-step traversal.

`natural_graph_features.pt` is ignored and regenerable. All CSV/JSON mapping,
transition, search, facet, routed-root, timing, and plot artifacts are tracked.
