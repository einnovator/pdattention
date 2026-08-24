# Paper 2.8 Agent Instructions: QK-Response Distilled Memory Gists

## Goal
Build and evaluate Paper 2.8 as a narrow continuation of PRA Papers 2.5 and 2.6. Do not reopen generic multi-gist, local-gist closure, or memory graph traversal as if they were new. The new hypothesis is: a PRA gist should approximate the frozen transformer's full key-stream query response, not reconstruct semantic clusters or generate a natural-language summary.

## Required repo context
Use the latest PRA code and artifacts from the pdattention repository. Reuse the same tokenizer/model revision, frozen query identities, chunking conventions, evidence annotations, and natural cohorts used by Papers 2.5/2.6 wherever possible. Include Paper 2.7 query-graph rows only as controls, not as the main mechanism.

## Primary research question
For a chunk key stream K_C and model-generated queries q, can m=2,4,8 compact native key landmarks preserve the full-K teacher ranking better than mean gists and prior multi-gist/prototype controls, at the same materialized K/V budget?

## Core definitions
- Teacher score S*(q,C): computed from full token-level keys in the chunk.
- Candidate teacher functions: max, top-r mean, log-sum-exp, attention-mass proxy.
- Compressed score S_hat(q,C): computed from selected or generated m-key summaries.
- Main objective: preserve chunk ranking and top-k selection induced by full-K, not reconstruct K vectors.

## Datasets
Use the same style as Papers 2.5/2.6:
- Controlled local bridge dataset.
- Synthetic QK teacher suite.
- HotpotQA.
- QASPER.
- 2WikiMultiHopQA.
- MuSiQue.

Initial natural cohort may match the Paper 2.7 frozen small cohort: 16 QASPER, 16 HotpotQA, 24 2WikiMultiHopQA, 18 MuSiQue. Scale only after a positive gate.

## Required baselines
Do not compare only against mean. Include:
- mean key gist;
- last-token key gist;
- random native key subset;
- farthest-first native key subset;
- historical prototype / k-means / SOM / hybrid multi-gist rows where available;
- Paper 2.5 parent/local/native-QK closure controls where available;
- Paper 2.6 semantic-only, BM25/lexical, and hybrid rows;
- Paper 2.7 graph query facets as a query-side control;
- full-K oracle ranking, clearly marked non-deployable;
- greedy QK-optimal native landmark oracle.

## Metrics
Report both native response-preservation metrics and Paper 2.5/2.6 routing metrics.

Native response metrics:
- score RMSE/MAE;
- Spearman/Kendall chunk-rank correlation;
- teacher top-k overlap for k=1,2,4,8;
- KL between teacher and compressed chunk distributions;
- calibration bins;
- head agreement / landmark overlap across heads.

Routing metrics:
- any-evidence recall;
- chain completion;
- exact identity when feasible;
- evidence coverage;
- evidence recall R_E, precision P_E, coverage C_E;
- MRR;
- active memory fraction K/N;
- evidence-normalized overhead K/E;
- requested token budget;
- materialized K/V tokens;
- semantic comparisons;
- native dots;
- wall-clock time;
- paired bootstrap 95% confidence intervals.

## Experimental gates
G0 - Reproduce Paper 2.6 inherited rows with compressor disabled.
G1 - Verify full-K teacher is a meaningful upper-bound router on controlled data and at least one natural dataset.
G2 - Greedy QK-optimal landmarks with m<=8 must beat mean and historical multi-gist controls on teacher top-4 overlap and at least one natural retrieval metric without increasing materialized K/V.
G3 - Train a tiny native landmark selector. It must recover at least 80% of the oracle gain and show paired positive recall or chain-completion delta on at least one natural dataset, with no regression below -0.02 elsewhere.
G4 - Test learned synthetic slots only if G3 passes.
G5 - Test streaming recurrent key memory only if the offline compressor passes.

Stop at the first failed gate and write the negative boundary clearly.

## Implementation tasks
1. Add feature extraction for per-layer/per-head Q and K streams, preserving token spans and evidence group IDs.
2. Add full-K teacher score computation: max, top-r mean, log-sum-exp, attention-mass proxy.
3. Add greedy QK landmark oracle for m in {1,2,4,8}.
4. Add no-training selectors: random, last, farthest-first.
5. Add tiny learned selector with soft/Gumbel top-k for training and hard top-k for inference.
6. Add optional slot compressor only after oracle/selector gates pass.
7. Add evaluation scripts that emit row-level CSVs compatible with Paper 2.5/2.6 audits.
8. Add changed-selection audits for wins and losses.
9. Add reproducibility manifests: model revision, tokenizer revision, dataset rows, seeds, cache hashes, script command lines.
10. Update paper_2_8.tex with actual results and keep claims matched to artifacts.

## Expected tables
- Teacher preservation by method, m, chunk size, teacher function.
- Natural retrieval by dataset and method at matched four-chunk budget.
- Budget curves at 10%, 20%, 30%, 40% where inherited rows exist.
- Cost table: compression ms, comparisons, native dots, materialized tokens.
- Head/layer ablation table.
- Gate status table.

## Claim discipline
Avoid saying "summarization" unless explicitly distinguishing it from routing-summary construction. Avoid claiming graph-KV novelty. If graph clustering is used, describe it only as a control or scaffold. The main claim is addressing-geometry distillation.
