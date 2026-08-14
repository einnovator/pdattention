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

## Cross-dataset granularity

The extension keeps contextual source encoding at 256 tokens and varies the
zero-overlap search-parent size over 16, 32, 64, 128, and 256 tokens. The
prior `K=6`, `B=16`, `H=4` operating point remains frozen; `K=8` is a ceiling.
At 16 tokens, selected source falls to 6.3% MuSiQue and 17.2% 2Wiki, while
complete recovery falls to 0.500/0.583 and 2Wiki preserved edge R@6 falls to
0.400. At 128 tokens, the values remain 0.944/0.958 and 0.880. The runner
asserts exact reproduction of the canonical 2Wiki transition and preserved-
path curves before writing artifacts.

MuSiQue D=4 recall saturates at H=2 even for 16/32-token search nodes. The
result is classified B/C/E: fine-edge representation collapses; shallow native
shortcut saturation persists; and fine parents substantially improve payload
localization at lower quality. The exact dense implementation is diagnostic,
not a serving-speed claim.

## Multiscale query audit

At fixed 128-token parents, every valid stride-one question span of width
1/2/4/8/16 plus the global contextual state is scored from one query encoding.
The evaluation-only best-facet root R@4 ceiling is 0.878 MuSiQue and 0.950
2Wiki, versus 0.422/0.608 for the frozen routed root. A bounded executable
Top-1-per-facet union with one shared four-root budget reaches only 0.411/0.475.
Useful query views therefore exist, but max aggregation cannot select them
without also promoting distractors.

Regenerate both gates with:

```text
python -m experiments.paper2_5_iterative_pra.run_natural_graph_depth --device cuda
python -m experiments.paper2_5_iterative_pra.run_natural_multiscale_query_audit --device cuda
```

`natural_graph_features.pt` and `natural_multiscale_query_facet_cache.pt` are
ignored and regenerable. Their manifests/hashes and all compact CSV/JSON
mapping, transition, search, facet, routed-root, timing, and plot artifacts are
tracked.
