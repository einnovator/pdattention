# Proposed publication review — Paper 1.5

Reviewed 2026-09-11. Branch `research/paper1-5-rope`, fetched and equal to `origin/research/paper1-5-rope` at `e68615b4663c363938dac9e837ea167957cae5dc`. Source: [paper.tex](paper.tex), including the September 1 pretrained long-context controls. This is a source-based review with selected artifact inspection, not a rerun or a PDF layout audit. Line anchors refer to this snapshot.

## Assessment and independent contribution

**Major revision, but one of the clearest independent scientific stories in the set.** The valuable result is the crossed dissociation: improving approximation of a native Q/K ranking does not improve annotated-target retrieval, while a supervised routing metric improves target retrieval without resembling that ranking. Offsets, overlap, and RoPE storage provide a concrete framework for distinguishing coordinate error from missing context.

The latest paper is substantially stronger than a narrow “RoPE support” implementation note. Preserve the negative WikiText result and the pretrained reset control. The main risk is presenting an intentionally simplified anchor-detection task as general semantic retrieval, or presenting an engineering correctness check as a new positional principle.

**Proposed central claim:** “Coordinate continuity, contextual fidelity, native Q/K ranking, and evidence selection are separable properties of retrieved memory; improving one does not guarantee improvement in the others.”

## Priority revisions

### P0 — Rename and delimit “semantic AUC” in the headline

**Locations:** abstract 37–50; pooling 552–572; learned routing 688–744; limitations 930–945.

The .971 value is a normalized sparse recall–fraction area, not ROC-AUC. The positive label is an explicit answer-code anchor; it is not a general semantic-relevance judgment. These qualifications are present later but disappear from the headline, where readers will easily infer a different task and metric.

Call it “target-identity recall–fraction AUC” or define the exact integral, normalization, fraction grid, ceiling rule, and aggregation at first use. Use actual selected fractions alongside nominal cutoffs. Fifteen candidates mean that the nominal 5% and 10% points are 6.7% and 13.3% physically.

Distinguish two supported claims: (a) Q/K-rank fidelity and annotated-target ranking dissociate in this controlled benchmark; (b) a learned metric detects the programmatic evidence identity. Broad semantic retrieval needs harder evidence. The minimal stronger test removes explicit answer-code cues, randomizes evidence placement, and evaluates on held-out source documents with ordinary question-conditioned labels. If that test is deferred, retain the controlled result with narrower wording.

### P0 — Test the fixed-location shortcut before attributing relevance learning

**Locations:** encoding-context ablation 478–484; learned protocol 688–697; shuffled-label interpretation 739–744.

The evidence is causally first in relevant probes. Attention-input hidden states are not generally position independent: they may carry positional information through earlier layers even when the current projection omits RoPE. A supervised metric may learn anchor syntax or source location. A negative-label control trained deliberately on non-evidence proves sensitivity to supervision, but does not exclude these shortcuts.

Document the evidence-location distribution and source-document overlap between train/test. Add, if not already available, a position-only baseline, random low-dimensional projection, anchor-only lexical baseline, and randomized-location evaluation. Do not automatically launch a large training campaign: first inspect the generator and split artifacts. If shortcuts remain untested, use “routing without an explicit positional rotation” instead of “position-independent semantic geometry.”

### P0 — Bound what the native-Q/K fidelity metric measures

**Locations:** equation for `s_native`, 552–564; interpretation 608–646.

The target maximizes over tokens after averaging heads. Actual multi-head attention normalizes within each head and combines values; it is not this scalar ranking. Therefore .963 fidelity is fidelity to a defined chunk-ranking surrogate, not .963 fidelity of attention outputs or a proof of approximating the full attention operator.

Define the head/token reducer beside every headline use of “native-attention fidelity.” Retain fixed-identity payload parity as a separate software invariant. An optional sensitivity using per-head ranking or attention-mass-based targets would strengthen generality, but accurate naming is the immediate fix. Clarify that the learned router did not establish improved downstream answering in these experiments.

### P1 — State RoPE assumptions and avoid absolute rebinding claims

**Locations:** minimal algebra 329–366; storage choices 318–325; retrieval placement 381–402.

The translation identity assumes a fixed frequency schedule and unchanged underlying unrotated Q/K. State those assumptions, including the extent of the rotary dimensions and numerical tolerance. It does not prove invariance of a whole contextual encoder under arbitrary translation, and length-dependent frequency changes require separate treatment.

Post-RoPE storage is not fundamentally incapable of rebinding: with original coordinates and the same rotation convention it can be inverse-rotated/re-rotated or transformed by a relative rotation. Pre-RoPE storage makes binding direct; it is not the only possible representation. Preserve the practical timing comparison without turning it into an impossibility claim.

MiniPIC already uses position-encoding-free K/V and user-controlled reuse; SemPIC studies learned cache construction while retaining the standard K/V interface. Make the contribution the controlled empirical separation, not priority for unrotated keys. [MiniPIC](https://arxiv.org/abs/2606.13126), [SemPIC](https://arxiv.org/abs/2607.28069).

### P1 — Clarify the pretrained experiment and numerical attribution

**Locations:** pretrained controls 818–867; generated position-geometry table.

This addition tests a query-coordinate restart with fixed source K/V at 8K/32K, not a complete replication of the controlled block-reset or learned-router experiment. Keep those interventions distinct. Preserved-coordinate sequence agreement is .667/.933/.667 at 8K, so “correct coordinates” should not become “exact execution.” Report the first-logit metric together with generated sequence changes and task scores.

The explanation that all residual error comes from unfused reduction needs a matched implementation control at the same full-source condition; an exact selected-text concatenated-cache test is useful but is not automatically a control for full 8K source segmentation. Inspect the matched receipts before making an exclusive numerical-cause attribution. Otherwise write “consistent with numerical differences in the segmented implementation.”

The `mac_long_context/mlx_long_context_summary.json` here is byte-identical to Paper 1's `mac_context_dilution` summary. Explicitly disclose the shared campaign and question identities. Position/reset analysis belongs here; dilution/payload scaling belongs to Paper 1. Shared data can support distinct questions, but it is not independent replication across papers.

### P1 — Improve inferential precision

Thirty comparisons are six conditions over five seeds, not 30 independent model replications. Likewise, 20 dataset–tier–seed pooling conditions share experimental structure. Preserve paired effect sizes and seed-level intervals; add question/document uncertainty where source-level observations exist. Do not infer absence of a practically relevant distance effect solely from a near-zero grand mean that may average opposite effects. Define a meaningful equivalence margin if claiming negligible effect; otherwise report heterogeneous near-range effects.

“Five levels form a dependency” is too strong: supervised retrieval can succeed despite poor native-Q/K fidelity, which is the paper's own result. Describe a diagnostic decomposition instead of a necessary causal chain.

## Readability and proposed organization

Reduce eight contribution bullets to three: the positional contract; controlled geometry/utility dissociation; pretrained coordinate control. Lead with the source-position-100 versus block-position-zero example. Introduce one experiment/protocol table before the results, rather than making readers reconstruct tiers and tasks from successive sections.

Recommended sequence: worked example -> definitions and assumptions -> matched factorial/protocol -> offsets/context -> pooling/routing dissociation -> pretrained scope -> implications/limits. Put early distance sweeps and storage microbenchmarks in supporting material unless needed for a central claim.

Suggested headline wording: “Increasing subgist resolution improves agreement with our native Q/K chunk-ranking target, but leaves target-identity retrieval nearly unchanged. A supervised projection shows the opposite pattern on the same controlled labels.”

## Publication boundary and Sol handoff

This paper can stand independently by owning coordinate/context/ranking separation. Paper 1 supplies a cited implementation background, not a prerequisite. Paper 2 should cite the controlled insight and contribute pretrained transfer rather than repeat these main tables. The independent reader needs the PRA attention equation and cache contract here, but not the series roadmap.

Suggested audience: a mechanistic ML journal or neural-computation journal. TMLR is a credible alternate destination for a carefully bounded empirical separation; its stated criteria prioritize supported claims and useful knowledge. [Criteria](https://jmlr.org/tmlr/acceptance-criteria.html). Do not choose a venue on the assumption that negative task-quality results are intrinsically unpublishable.

**Sol pass:** first fix metric names, dependence language, positional assumptions, and shared-campaign attribution. Audit generator/location shortcuts and the exact split count. Then decide whether a small cue-free evaluation is needed for the desired semantic claim; otherwise narrow it. Preserve all negative rows and finish with a source-to-claim checklist, separate experiment backlog, and rendered manuscript review.

Source preflight found no missing literal inputs, unresolved cited keys, undefined cross-reference labels, or duplicate labels. The generated pretrained table and shared JSON were inspected. This is not complete numerical or layout validation.
