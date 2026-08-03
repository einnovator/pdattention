# AGENTS.md - Review and Extension Instructions

## Mission

Turn `pra_survey.tex` into a publication-quality educational review and PRAttention comparison framework. Preserve its seven-category organization and its central distinction between naming, selection, materialization, fusion, and causal evaluation.

## Non-negotiable scientific rules

1. Do not inflate PRA novelty. Clearly distinguish prior mechanisms from the PRA package: explicit URI identity, naming/materialization separation, layer-specific reusable KV objects, progressive triggering, and counterfactual evaluation.
2. Never claim superiority from the existing tiny WikiText-2 pilot. Preserve the caveats: two seeds, generated reference-conditioned continuation, same width rather than parameter matching, no measured cache reuse, and small shuffled-reference deltas.
3. Verify every quantitative statement from the primary paper. Prefer arXiv, proceedings, publisher pages, and official repositories.
4. Distinguish architecture claims, systems claims, benchmark claims, and hypotheses.
5. For PageIndex and other rapidly evolving systems, record repository commit/tag, model, prompts, dataset version, judge, and cost. Do not repeat marketing claims as settled results.
6. Treat attention maps as diagnostics, not causal explanations. Prefer removal, substitution, shuffled-reference, and mediation-style tests.

## Required expansion work

### A. PRA section

- Import exact definitions and diagrams from the latest Paper 0 and Paper 1 sources.
- Add a table of all preliminary runs, seed-level metrics, parameter counts, trainable parameters, and uncertainty.
- Add the full reference-data construction algorithm and leakage controls.
- Add complexity analysis for resolve, encode, cache, route, and cross-attend stages.

### B. One-to-two pages per representative proposal

For every major method include:

1. motivation and core insight;
2. architecture and data flow;
3. mathematical formalism;
4. pseudocode;
5. runnable minimal PyTorch sketch;
6. diagram;
7. original datasets, models, baselines, metrics, and headline results;
8. ablations and failure modes;
9. paper trail: precursor work, same-author follow-ups, and downstream systems;
10. precise relation to PRA;
11. an experiment adapted for direct comparison with PRA.

Priority methods:

- Transformer-XL, Compressive Transformer, RMT, Infini-attention;
- Longformer, BigBird, Performer;
- Memory Networks, DNC, Memformer, Memorizing Transformer;
- kNN-LM, REALM, RAG, RETRO;
- Unlimiformer, PagedAttention, StreamingLLM, H2O, Scissorhands, SnapKV;
- Hierarchical Attention Networks, RAPTOR, MemWalker, PageIndex, GraphRAG;
- ReAct, Self-Ask, IRCoT, Toolformer, WebGPT, recursive language models;
- PaperTrail as provenance/HCI rather than a context architecture.

### C. Experimental synthesis

Create reproducible experiment specifications for:

- equal token, parameter, FLOP, latency, and memory budgets;
- valid/disabled/shuffled/irrelevant/oracle reference conditions;
- local-sufficient versus reference-required subsets;
- context length and distance sweeps;
- object count, object size, and distractor sweeps;
- cold cache versus warm cache versus cross-user shared cache;
- stale-version and invalidation tests;
- structured document navigation and multi-hop QA;
- open-web discovery requiring agent escalation;
- claim-level citation and provenance evaluation.

### D. Diagrams

Maintain the one-page landscape diagram. Add consistent TikZ diagrams for each architecture. Use a shared visual grammar:

- gray: local token compute;
- blue: retrieval and external corpora;
- green: learned memory;
- purple: structural indexes;
- red: agents/tools;
- orange: PRA and proposed hybrids.

### E. Bibliography

Move the embedded bibliography to `references.bib`. Add DOI and live HTML/arXiv links. Check author names, year, venue, title capitalization, and version. Add access dates only for evolving repositories.

## Build and QA

Run:

```bash
latexmk -pdf -interaction=nonstopmode survey.tex
```

Then render the PDF to images and inspect every page for:

- overfull tables or code blocks;
- clipped landscape diagrams;
- broken hyperlinks;
- missing glyphs;
- orphan headings;
- inconsistent captions;
- citations that do not support adjacent claims.

Fail the task if LaTeX emits unresolved references/citations or if any page has visible overlap/clipping.

## Suggested paper split

If the manuscript exceeds a practical tutorial length, preserve one canonical source but generate two outputs:

1. `tutorial-survey.tex`: neutral field tutorial and taxonomy;
2. `pra-comparison-agenda.tex`: PRA-centered synthesis, experimental doctrine, and design extensions.

## Definition of done

The review is submission-ready only when each major architecture has been checked against the primary source, all headline result claims are tabulated with conditions, code sketches execute in tests, diagrams are visually verified, and PRA comparisons are framed as testable hypotheses rather than conclusions.
