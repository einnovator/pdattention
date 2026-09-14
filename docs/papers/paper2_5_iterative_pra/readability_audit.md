# Paper 2.5 Readability Audit

## Page Structure

- Final PDF: 73 pages total, with an 18-page reviewer-facing main narrative.
- The appendix begins on page 19 and preserves the chronological gate record, negative results,
  implementation maps, external judgment audit, and validation gates.
- The main paper lives in `main_associative_memory.tex`; `paper.tex` retains the common preamble and
  archival appendices. This separation makes the scientific argument inspectable without deleting
  the experiment history.

## Main Narrative

The main paper now follows the causal decomposition requested for the associative-memory framing:

1. PRA as a probe of transformer associative memory;
2. receptive field and memory topology;
3. iterative traversal under a fixed physical K/V budget;
4. controlled activation and the oracle memory intervention;
5. layerwise consumption and preservation;
6. pretrained validation across root entry, edge geometry, granularity, and output behavior;
7. related work, discussion, limitations, explicit follow-up boundaries, and conclusion.

The final reviewer patch adds a compact "When Does Iteration Help?" analysis before the causal
diagnosis, a dataset-routing geometry table in pretrained validation, and a parameter-directionality
table in discussion. The main text explicitly distinguishes measurements available after one-shot
retrieval from outcomes visible only after iteration.

The former gate sequence was moved behind the appendix transition. It preserves projection
correction, parent and local semantic discovery, native-Q/K controls, oracle rank and competition,
protected-root policies, query facets, static and dynamic grounding, terminal-threshold failures,
granularity surfaces, cross-dataset path diagnostics, layerwise topology, systems accounting,
implementation maps, and validation gates.

## Headline Evidence

- Controlled receptive-field matrix: `controlled_local_sa_v6/receptive_field_topology_summary.csv`.
- Matched one-shot/iterative analysis: `controlled_local_sa_v6/iterative_matched_budget_seed_stats.csv`.
- Frozen-consumer causal controls: `controlled_local_sa_v6/causal_memory_seed_summary.csv`.
- Traversal-to-answer joins: `controlled_local_sa_v6/traversal_to_use_rows.csv`.
- Final 59-versus-341 rows and miss-conditioned feature summaries:
  `final_reviewer_patch/iteration_benefit_59_vs_341.csv` and
  `final_reviewer_patch/iteration_benefit_feature_summary.csv`.
- Grouped pre-decision predictability diagnostic:
  `final_reviewer_patch/iteration_benefit_predictability.json`.
- Cross-dataset reviewer synthesis: `final_reviewer_patch/dataset_routing_geometry_summary.csv`.
- Attention, residual, and consumer-layer traces: `controlled_local_sa_v6/mechanistic/`.
- Exact synthetic indirect discovery: `local_associative_closure/gate2_local_results.json`.
- 2Wiki edge and path fidelity: `natural_graph_depth/natural_graph_depth_results.json`.
- Granularity and conceptual payload: `natural_graph_depth/cross_dataset_granularity.csv`.
- Layer-by-granularity topology: `layerwise_graph/layer_granularity_summary.csv`.
- Oracle versus executable root routing: `natural_graph_depth/routing_ceiling_table.csv`.
- Frozen native-K/V output behavior: `output_validation/gate3_output_summary.csv` and
  `output_validation/gate3_output_analysis.json`.

The complete value-level mapping is in `claim_to_artifact_audit.md`.

## Render Verification

- Built with `pdflatex`, `bibtex`, and two final `pdflatex` passes on August 16, 2026.
- `pdfinfo` reports 73 US-letter pages and approximately 2.1 MiB output.
- LaTeX reports no fatal errors, undefined references or citations, package errors, or overfull boxes.
  Remaining underfull-box notices occur in narrow bibliography and code-map paragraphs and do not
  clip content.
- Visually reinspected reviewer-facing pages 1, 9--13, and 15--18 at 130 dpi, including the abstract,
  59-versus-341 analysis, grouped predictability result, dataset geometry, judge distinction,
  parameter directions, scope handoffs, conclusion, and appendix boundary. The previously inspected
  topology, trajectory, causal-pipeline,
  and implementation-map content is unchanged. Headings, equations, tables,
  figures, captions, references, and page numbers are legible; no overlap, clipping, black boxes, or
  missing glyphs were found.

## Terminology Changes

- The causal sequence is **potential topology**, **root activation**, **associative traversal**,
  **controlled activation**, **consumption**, and **preservation**. These stages are not conflated.
- **Closure** remains only where it names a historical experiment, code artifact, or exact set
  operation in the appendix.
- **Root routing**, **native successor**, **native graph**, **annotated/task graph**, **path
  survival**, and **minimum native recovery depth** name separate measured quantities.
- **Conceptual selected memory** is distinguished from **native K/V materialization**.
- **Oracle memory** is a matched causal capacity control, not an executable routing method.

## Reviewer Risks Still Open

- Natural cohorts are diagnostic and cover one frozen 0.6B model family.
- Oracle-root and oracle-evidence controls are ceilings, not deployable policies.
- High complete recovery still requires broad conceptual source activation.
- Fine nodes reduce conceptual payload but damage local native-edge fidelity.
- MuSiQue generation is limited by the frozen backbone even with full direct context.
- The 2Wiki output deltas have confidence intervals spanning zero; the paper does not claim an
  end-to-end quality gain.
- The controlled causal cohort has 16 paired examples per checkpoint and five seeds per receptive
  field. Oracle effects are directionally consistent, but the minimum exact two-sided five-seed sign
  result is `p=.0625`; the paper treats this as a diagnostic ceiling, not a powered architecture claim.
- Iterative traversal improves margins strongly on path-improved model-example units, but the
  current policy produces too few such units to establish an aggregate accuracy improvement.
- The primary pre-decision stump is descriptive, label free, and not a deployable retry policy.
  Grouped held-out balanced accuracy is .638, while precision is .368 with a broad [.221,.529]
  identity-bootstrap interval; all estimates rest on 16 independent task identities. A stronger
  .839 query-length sensitivity is explicitly identified as a synthetic generator shortcut.
- The causal receptive-field comparison belongs to the controlled 25-model family. Qwen
  layer/chunk diagnostics are directional architecture evidence, not a causal pretrained
  LocalSA-versus-GlobalSA comparison.
- Ordinary selected memory remains distractor dominated; controlled activation, rather than search
  depth alone, is the primary unresolved mechanism.
- GPT-5.6 Sol supplies one complete, control-qualified external judgment. Claude covers 51.7% of
  pairs and fails calibration, leaving evaluator robustness unresolved.
- Sparse-band superiority to all-layer injection is a placement result; it is not evidence that
  balanced sparse PRA substantially outperforms one-shot or native output.
- Synchronized TTFT/TPOT and total generation latency are measured per example; production
  serving, concurrency, and optimized graph search remain future systems work.
