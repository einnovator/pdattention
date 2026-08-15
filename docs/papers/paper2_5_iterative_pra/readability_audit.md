# Paper 2.5 Readability Audit

## Page Structure

- Previous PDF: 66 pages total; the first appendix began on page 58, leaving 57 main-text pages.
- Reorganized final PDF: 69 pages total; the chronological appendix begins on page 16, leaving 15
  reviewer-facing main-text pages.
- The three added total pages are attributable to the final Gate-3 output evidence, external-judge
  results, focused related work, and provenance material. No prior negative experiment was deleted
  to reduce page count.

## Main Narrative

The main paper now follows the final scientific decomposition:

1. inherited PRA/HF setting;
2. bounded associative discovery;
3. dataset roles and frozen evaluation protocol;
4. focused comparison with long-context, retrieval, and K/V-selection work;
5. results organized by scientific question;
6. discussion, limitations, handoff, and conclusion.

The former gate sequence was moved behind the appendix transition. It preserves projection
correction, parent and local semantic discovery, native-Q/K controls, oracle rank and competition,
protected-root policies, query facets, static and dynamic grounding, terminal-threshold failures,
granularity surfaces, cross-dataset path diagnostics, layerwise topology, systems accounting,
implementation maps, and validation gates.

## Headline Evidence

- Exact synthetic indirect discovery: `local_associative_closure/gate2_local_results.json`.
- 2Wiki edge and path fidelity: `natural_graph_depth/natural_graph_depth_results.json`.
- Granularity and conceptual payload: `natural_graph_depth/cross_dataset_granularity.csv`.
- Layer-by-granularity topology: `layerwise_graph/layer_granularity_summary.csv`.
- Oracle versus executable root routing: `natural_graph_depth/routing_ceiling_table.csv`.
- Frozen native-K/V output behavior: `output_validation/gate3_output_summary.csv` and
  `output_validation/gate3_output_analysis.json`.

The complete value-level mapping is in `claim_to_artifact_audit.md`.

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
- GPT-5.6 Sol supplies one complete, control-qualified external judgment. Claude covers 51.7% of
  pairs and fails calibration, leaving evaluator robustness unresolved.
- Synchronized TTFT/TPOT and total generation latency are measured per example; production
  serving, concurrency, and optimized graph search remain future systems work.
