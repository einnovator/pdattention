# Paper 2.5 Readability Audit

## Page Structure

- Previous frozen PDF: 69 pages total, with 15 reviewer-facing main-text pages.
- Outcome-B diagnostic PDF: 81 pages total; the chronological experiment record remains in the
  appendices, while the controlled causal diagnosis now occupies pages 8--12 of the main text.
- The added pages preserve the full historical record and add the five-condition causal intervention,
  traversal-to-use analysis, consumer-layer profile, and reproducibility artifacts. No negative result
  was removed to shorten the paper.

## Main Narrative

The main paper now follows the final scientific decomposition:

1. inherited PRA/HF setting;
2. bounded associative discovery;
3. dataset roles and frozen evaluation protocol;
4. focused comparison with long-context, retrieval, and K/V-selection work;
5. controlled traversal-to-answer causal diagnosis;
6. results organized by scientific question;
7. discussion, limitations, handoff, and conclusion.

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

- Built with `pdflatex`, `bibtex`, and two final `pdflatex` passes on August 15, 2026.
- `pdfinfo` reports 81 US-letter pages and a 2.29 MiB output.
- LaTeX reports no fatal errors or overfull boxes. Remaining underfull-box notices occur in narrow
  bibliography and code-map paragraphs and do not clip content.
- Visually inspected the title page, causal-diagnosis pages 8--12, and final bibliography page 81 at
  130 dpi. Headings, equations, tables, figures, captions, references, and page numbers are legible;
  no overlap, clipping, black boxes, or missing glyphs were found.

## Terminology Changes

- The reviewer-facing mechanism is called **bounded associative discovery** or **bounded search**.
  **Closure** remains only where it names a historical experiment, code artifact, or exact set
  operation in the appendix.
- **Root routing**, **native successor**, **native graph**, **annotated/task graph**, **path
  survival**, and **minimum native recovery depth** name separate measured quantities.
- **Graph contraction/shortcut** does not imply shorter reasoning.
- **Conceptual selected memory** is distinguished from **native K/V materialization**.
- **Oracle evaluation** is distinguished from **oracle-free discovery** and executable selection.

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
- GPT-5.6 Sol supplies one complete, control-qualified external judgment. Claude covers 51.7% of
  pairs and fails calibration, leaving evaluator robustness unresolved.
- Synchronized TTFT/TPOT and total generation latency are measured per example; production
  serving, concurrency, and optimized graph search remain future systems work.
