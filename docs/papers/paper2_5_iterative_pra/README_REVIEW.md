# Proposed publication review — Paper 2.5

Reviewed 2026-09-11 against fetched branch `research/paper2-5-iter-gist` at `e81d434501db5e67dcb8e2b6043f6b798f7b7688`, equal to its origin tracking ref. Entry: [paper.tex](paper.tex); substantive article: [main_associative_memory.tex](main_associative_memory.tex). This review includes the latest normalized-budget addition. It uses source, selected experiment code, and stored outcome rows; models were not rerun. Line anchors below refer to the substantive article at this commit.

## Assessment and independent contribution

**Major revision.** The most interesting result is a tradeoff: local receptive fields preserve useful associative structure, yet better memory traversal does not reliably translate into better answers. The controlled topology experiment and oracle-consumption comparisons can support a coherent independent paper. A general claim that iterative PRA improves reasoning is not established.

**Proposed central claim:** “Local receptive fields alter the accessibility of multi-step relations. Budgeted iterative retrieval sometimes improves path recovery, but aggregate answer gains remain unresolved, and oracle controls expose a separate consumption limitation.” Keep the controlled 25-model study central. Pretrained document retrieval should test the boundary of that conclusion, rather than become a second HF integration paper.

## Priority revisions

### P0 — Put aggregate benefit and harm beside the positive subgroup

**Locations:** abstract; causal analysis from approximately 338; path-improved analysis from approximately 406.

The 59 path-improved observations and their mean margin gain of +2.164 are informative, but they are selected after observing the treatment outcome. They cannot establish an overall gain or causal mediation by improved traversal.

Recalculation from stored `traversal_to_use_rows.csv` gives the following descriptive results across 400 model-example rows:

| Outcome group | Rows | Mean answer-accuracy change | Mean margin change |
|---|---:|---:|---:|
| Improved path recovery | 59 | +0.1017 | +2.1639 |
| Unchanged path recovery | 293 | −0.0273 | +0.2901 |
| Worse path recovery | 48 | −0.2917 | −1.5852 |
| All rows | 400 | −0.0400 | +0.3414 |

Mean path gain across all rows is +0.0275, or 2.75 percentage points. These are recomputed point estimates, not new significance tests. Report the full distribution and overall comparison before subgroup interpretation. Preserve the positive finding as a conditional association. State plainly that the answer-accuracy point estimate in this mechanistic cohort is four percentage points lower under iteration.

**Completion criterion:** the abstract, principal table, discussion, and conclusion agree about aggregate versus subgroup results and distinguish this cohort from the larger controlled test sets.

### P0 — Respect the number of independent examples

The 400 rows reuse **16 unique example identities** across 25 trained models; they are not 400 independently sampled tasks. `experiments/paper2_5_iterative_pra/summarize_outcome_b.py`, function `bootstrap_ci` around lines 65–73, resamples the supplied values independently. An interval over the selected model-example rows therefore needs stronger qualification than a generic bootstrap label.

Recompute question-identity-clustered uncertainty for paired effects and provide sensitivity to model seed/window. Treat window as an experimental condition, and explain what population each interval describes. With only 16 task identities, show per-identity results and avoid precise population claims. Do not replace seed-level intervals for the separate 512-example evaluation with this cohort's intervals. Document whether subgroup membership is held fixed or recomputed during resampling; either version remains a selected-subgroup analysis.

### P0 — Separate iteration from its injection schedule

**Locations:** iterative method from 271 and controlled comparison near 300.

One-shot selection places four facts at one layer, while iteration places one at each of layers 0, 2, 4, and 5. Equal total layer-token state does not isolate an evolving-query search effect: timing, exposure depth, and selection all change.

First describe the comparison accurately as two complete schedules. If retaining a claim specifically about iterative query updating, add a narrow matched-schedule control: precompute selections from a static query and inject them on the same layer schedule, with the same no-repeat policy and budget. Compare that with evolving-query selection. A preselected/oracle schedule can separately diagnose timing. Otherwise narrow the mechanism attribution without running new experiments.

Add two short algorithm boxes distinguishing controlled residual-dependent selection from pretrained graph traversal over a compact memory. Both are iterative, but they do not demonstrate an identical mechanism.

### P0 — Do not describe the retry gate as deployable on its evaluated eligibility set

**Locations:** retry analysis following the path-improved subsection.

The decision stump uses observable features and leave-one-example-out evaluation, but evaluation starts with 248 examples where one-shot retrieval missed the gold path. Gold-free features do not make that eligibility decision available at inference. The reported balanced accuracy of .638 is conditional on an oracle-defined subset.

Call this a conditional diagnostic. For an executable policy claim, define eligibility from pre-retry observable state, evaluate all examples, and include unnecessary retries, regressions, and extra retrieval cost. Keep hyperparameter selection inside the appropriate training folds. This distinction is more important than adding a more complex classifier.

### P0 — Correct the evidence-use interpretation

**Locations:** causal table around 362; layer dynamics from 504; attention diagram around 540.

The oracle attention masses are approximately .270 evidence and .301 distractor, with the remainder native. The diagram's “EVIDENCE-DOMINATED ATTENTION” label is inconsistent with those values. Use “increased evidence attention” and report all categories.

Oracle selection raises accuracy from approximately .140 to .398 in the relevant comparison, demonstrating room for better selection under the tested consumer. It does not imply reliable reasoning. Likewise, a declining intermediate logit-lens margin is not direct proof that usable information was erased. Use “reduced intermediate answer decodability,” unless a causal intervention or calibrated readout supports the stronger interpretation.

### P1 — Integrate and repair the normalized-budget section

**Location:** approximately 603–650.

Define an evidence set `E`, selected set `S`, and candidate count `n`; use `|S|/n`, `|S∩E|/|E|`, and `|S∩E|/|S|`. The current notation mixes set intersections with variables introduced as counts and omits cardinalities.

At the 10% budget, HotpotQA mean recall changes .207 to .202 and complete recovery .013 to .000; QASPER recall changes .477 to .466 and complete recovery .213 to .200. Present this as an unresolved or unfavorable comparison at that setting. Explain that the oracle obeys a common cap but often executes a smaller selection; “matched budget” does not mean equal realized cost. Identify the 80 example-seed observations per dataset and their unique identities, and connect this cohort explicitly to the earlier pretrained experiments. Do not imply that one normalized panel establishes an improved Pareto frontier.

### P1 — Turn the article into a scientific argument instead of a campaign history

Suggested order: question and contribution; controlled task and topology; two retrieval schedules; aggregate results; oracle and content controls; limits of transfer to pretrained models; implications. Move chronological experiment additions from the entry file's appendix into a clearly indexed supplement, retaining anything needed for reproduction.

Use the existing configuration detail: six layers, width 96, four heads, five seeds, training/validation/test sizes, and update count. Add a compact experiment manifest connecting each table to its cohort and artifact. Avoid letting “400,” “25 models,” and “five seeds” appear as interchangeable sample sizes.

The pretrained results should remain candid: for 2Wiki, the displayed balanced/broad iterative F1 values (.313/.271) do not beat one-shot (.354), and broad retrieval requests most of the source. MuSiQue at zero even with full context is a consumer limitation in that setting. These outcomes can strengthen the paper's diagnostic value when presented without a positive-performance headline.

## Readability and reviewer interest

Suggested title direction: **“Associative Memory Topology and the Limits of Iterative Retrieval in Transformers.”**

Suggested opening: “Retrieving the next relevant fact is only part of multi-step reasoning. The model must also preserve and use that fact. We study these two requirements separately in controlled transformers and frozen pretrained models.”

Explain edge recall, complete-path recovery, answer accuracy, and answer margin together before any results. Replace internal campaign labels with descriptive experiment names. Use one worked chain example to show the source order, causal mask, retrieved facts, and prediction target. The reader should understand why local contextualization can preserve associations without consulting another PRA paper.

## Independent publication boundary and journal direction

Own the **topology–traversal–consumption relationship**. Cite Paper 1 for the broader PRA mechanism and Paper 2 for integration background; restate the minimal mathematics and configuration here. Leave position-frame fidelity to Paper 1.5 and representation/index selection to Paper 2.6. Shared tasks or artifacts need explicit attribution and a clear statement of the new analysis.

A neural-network mechanisms or learning-methodology journal is the natural audience. [Machine Learning's official scope](https://link.springer.com/journal/10994/aims-and-scope) includes methodology and requires clearly supported, reproducible contributions; this is a plausible fit if the learning/topology result leads. Coordinate venue allocation with Paper 1.5 rather than assuming different journals are necessary for scientific independence. This is a fit assessment, not a submission-readiness prediction.

## Sol handoff and acceptance checklist

1. Work on this branch and compare HEAD with the reviewed commit before using line anchors.
2. Correct aggregate/subgroup framing, sample-unit descriptions, attention label, notation, and retry eligibility using existing evidence first.
3. Recompute clustered paired summaries from stored rows and preserve scripts, inputs, and outputs. Never substitute unrun estimates.
4. Decide whether to narrow the iteration claim or add the matched-schedule control; list proposed runs separately from completed evidence.
5. Rewrite around the independent question and attach every numerical claim to a source artifact and cohort.
6. Compile the article and supplement, inspect figures/tables, and return a concise change log plus unresolved author decisions.

The source preflight found no unresolved literal inputs, citation keys, references, or duplicate labels in the expanded entry sources. This does not certify mathematical correctness, all artifact claims, or PDF layout. This review proposes revisions; it does not modify the manuscript or authorize invented results.
