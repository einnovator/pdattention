# Paper 1.5 claim ledger

Updated 2026-09-12 for the P0 evidence pass reviewed at commit
`e68615b4663c363938dac9e837ea167957cae5dc`.

## Supported headline claims

| Claim | Evidence | Status and permitted wording |
| --- | --- | --- |
| Source-relative offsets remove layer-0 reset error in 30/30 paired comparisons and reduce final-layer K error in 30/30. | `validation/positional_mechanism_offset_validation.json` and `.csv` | Measured result. State the two capacities, three position mechanisms, and five seeds; do not call these independent model replications. |
| WikiText representation improvement does not imply next-token-loss improvement. | `validation/wikitext_position_validation.json`, `wikitext_summary.csv` | Measured negative result: representation improves in 30/30 pairs, loss in 5/30. |
| Centered subgists improve agreement with the defined native-Q/K chunk-ranking surrogate without improving target-identity recall/request-fraction area. | `pooling_geometry/pooling_geometry.json`, `pooling_geometry_publication.csv` | Measured controlled dissociation: `.794 -> .963` surrogate rank fidelity and `.6510 -> .6516` normalized target-identity area for `G=1 -> 8`. Never describe `.963` as attention-output fidelity or the area as ROC-AUC. |
| A supervised 32-D projection ranks the programmatic target identity well while resembling the Q/K surrogate poorly. | `learned_routing/learned_routing_geometry.json`, `learned_routing_aggregate.csv` | Measured controlled result: `.971` normalized target-identity recall/request-fraction area and `.178` surrogate rank fidelity. This is not evidence of general semantic relevance or downstream answer improvement. |
| Fixed selected identities preserve native K, V, and attention output exactly. | `pooling_geometry/pooling_geometry.json`, `learned_routing/learned_routing_geometry.json` | Software invariant for identical selected identities. Keep separate from selection quality and full attention-operator approximation. |
| Pretrained query-position restart is much more disruptive than the tested unfused segmented path. | `mac_long_context/qwen3_8b_long_context.json`, `qwen3_14b_long_context.json`, `qwen3_32b_long_context.json`, and `mlx_long_context_summary.json` | Fifteen unique 8K questions per model and three 8B/32K questions from one quantized Qwen family. The concatenated-cache rows in `mlx_long_context_summary.json` are byte-identical shared evidence with the standalone native-K/V manuscript, not an independent replication. |

## Shortcut and split audit

The read-only audit is stored in
`../shared/results/paper1_5_rope/learned_routing/shortcut_audit.json`.
For both 64-example datasets, every explicit `AnswerCode yes|no` anchor is at the source start
and its target is candidate `part-1`. The 51/13 training/held-out split has no HotpotQA
source-row overlap. QASPER has one paper shared across train and held out through two different
questions. The experiment contains a shuffled-label control but no position-only, random
low-dimensional, anchor-only lexical, randomized-location, or document-disjoint QASPER
control. Accordingly, it supports supervised target-identity detection in this construction;
it does not resolve which cue the projection learned.

## Deferred experiment backlog

- Randomize target location while keeping candidate content and labels otherwise matched.
- Remove or randomize the explicit answer-code cue and test an anchor-only lexical baseline.
- Add a position-only baseline and an untrained random low-dimensional projection.
- Make the QASPER split paper-disjoint and evaluate transfer to unseen source documents.
- Replace programmatic answer-code identities with ordinary question-conditioned relevance
  labels before making a broad semantic-retrieval claim.
- Compare the mean-head/max-token surrogate with per-head and attention-mass-based targets.
- Evaluate whether selector replacement improves downstream answers with identical payload and
  consumption policy.

These are future experiments, not evidence claimed by the present manuscript.
