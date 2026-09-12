# Proposed publication review — Paper 2

Reviewed 2026-09-11. Branch `research/paper2-hf`, fetched and equal to `origin/research/paper2-hf` at `139b4dc1229ad1d7af6f1486e70ddc99704ebd55`. Source: [paper.tex](paper.tex), including the September 8 MoE/hybrid-family qualification update and larger-model appendix. Source-based review with targeted artifact/protocol inspection; no model rerun or PDF layout certification. Anchors refer to this snapshot.

## Assessment and independent contribution

**Major revision; credible independent integration paper.** The strongest contribution is the explicit pretrained contract: routing returns identities, materialization preserves layer-native payload, family-specific attention semantics remain intact, and the disabled path delegates exactly to the original model. The separation between discovery and consumer adaptation is useful. The current main-text compression has improved the narrative considerably.

The scientific quality claims are less secure than the software-contract claims. Headline routing curves mix cohorts, the decisive QASPER polarity comparison has eight question identities, and disabled-path equality is guaranteed largely by control flow. Preserve the exactness result, but do not let it imply general memory-active preservation or broad QA benefit.

**Proposed primary claim:** “A shared native-K/V interface can be retrofitted to several pretrained decoder families; exact disabled behavior and active-memory utility require separate tests, and the latter depends on evidence selection and consumption depth.”

## Priority revisions

### P0 — Separate measured checkpoints, structural qualification, and plans

**Locations:** abstract 25–54; thin adapter section; portability 836–858; larger-model appendix 1849–1878.

Keep a single evidence-tier table near the front: loaded checkpoint + parity; active-memory functional test; tiny configuration test only; specification/planned integration. The latest text correctly excludes structural Mistral 3/GPT-OSS and Qwen3.8 planning from the three-checkpoint comparison. Preserve that distinction and simplify the abstract so it is not a compatibility release note.

The MLX 8B–32B/MoE experiment is a separate backend, selected-evidence protocol, and consumer-depth study. It does not validate every HF adapter or the learned router at scale. Give exact model/backend/quantization/provenance and disclose overlap with Papers 1 and 1.5. Do not aggregate 60 model–question executions as 60 independent questions.

### P0 — Make the routing frontier a paired comparison

**Locations:** representation table 540–557; recall–sparsity table/figure 604–629; selected-seed identity recall 632–638.

The manuscript openly notes different cohorts, which is good, but still visually presents one frontier and bold “best” values. A 16-example historical baseline cannot establish a paired improvement over a 32-identity learned-router test. The matched R@3 comparison at 586–589 is better evidence.

Recompute parameter-free baseline rankings on the same frozen test identities if their features are available; no backbone training should be necessary. Otherwise split historical and matched tables and remove comparative frontier language for unmatched curves. Report any-, identity-, and all-evidence recall on the same selected seed/cohort, and distinguish requested parent fraction from the fraction surviving physical admission. The .956 any-evidence result does not establish nearly complete evidence retrieval at 6–7% active K/V.

### P0 — Put the eight-question QASPER denominator and majority control beside the headline

**Locations:** main positive result 793–829; polarity details 1547–1631.

The 72.5% is an optimization mean over five seeds on eight fixed identities. It is not 40 independent test questions. The test-majority baseline is 62.5%, equal to the frozen oracle polarity result. That does not erase the adaptation signal, but it changes the scale of the claim. Report numerator/denominator, label balance, per-class results, and paired question-level uncertainty beside .250/.625/.725.

Do not imply the sequence routed-frozen -> oracle-frozen -> residual-routed rows identify additive router and consumer effects. They change different conditions. Add a compact factorial table when existing results permit: frozen/adapted by disabled/routed/oracle/wrong memory. Distinguish training on oracle Hotpot memory from the routed-QASPER-trained residual, whose F1 and polarity move differently.

Repeated tuning and diagnostics on the same test questions make the current evidence exploratory at program level, even where each individual run used disjoint validation. A new untouched cohort is the most valuable additional experiment for a broad positive QA claim. Without it, describe the result as a small held-out diagnostic and keep the exactness contribution primary.

### P0 — Repair behavioral-evaluation interpretation

**Locations:** main table 803–818; behavioral protocol 2331–2424.

The 2.5%/22.5% entries need an explicit comparator, denominator, judge identity, tie rate, and whether “negative” means worse than the comparator. The main table presents a stronger apparent positive result than the full judge-specific table: frozen PRA versus no context has negative mean quality on QASPER, while residual versus frozen is positive. Those are different contrasts.

The package gives judges prompt and answers, with no listed source evidence or gold answer. Therefore it can measure judged equivalence/plausibility/preference, but cannot independently establish factual correctness on source-dependent questions. One judge under-scores identical answers, and the corruption control can accidentally improve a weak answer. Keep these failures visible in the main assessment; perfect A/B reversal does not compensate for them.

Request human evidence-grounded adjudication of all eight QASPER identities and a small stratified Hotpot subset, using natural stopping budgets and a defined rubric. This is a proposed follow-up, not work to fabricate automatically. Preserve original judge files, use new package IDs, and keep eight-token and 32-token instruments distinct. Phrase automated judging as supplementary evidence until then.

### P1 — Narrow “oracle-path bug ruled out” and distinguish units

**Locations:** abstract 38–42; oracle-gap table 732–748; depth tables 699–713 and 962–990.

Exact recomputation, span coverage, and layer execution exclude the tested payload/identity failures. They do not exclude every possible mask, query-position, native-frame, or contextualization error. Replace blanket “bug rejected” with the specific invariants that passed. The later paragraph already recognizes representation differences; retain its caution about attention mass not being a scalar causal explanation.

The main depth-table caption says “gold-sequence Delta log p,” whereas the appendix defines and reports mean gold-token log probability. Use “nats per gold token” consistently for the +3.641/−.766 experiment. The adaptation ladder uses sequence log probability and must retain a different symbol/label. Likewise, separate source tokens, per-layer active K/V, layer-summed token states, transfer bytes, and whole-process peak allocation.

### P1 — Strengthen active-path preservation and the closest baseline

No-memory parity follows from bypassing the adapter and is a valuable regression guarantee. It does not test ordinary tasks while irrelevant memory is present or a memory gate erroneously opens. Report false activation, irrelevant-memory effects, and preservation with PRA enabled before saying ordinary language is preserved broadly.

The decisive integration comparison is identical selected evidence as text versus native K/V, with source layout, position, consumer layers, and compute disclosed. Show it as a comparison of representations and consumer paths, not a general RAG victory. Discuss [RetrievalAttention](https://arxiv.org/abs/2409.10516) and [CacheBlend](https://arxiv.org/abs/2405.16444) as direct neighbors. Existing layer sweeps vary total layer-token intervention cost, so they identify schedule dependence rather than an isolated depth law.

## Readability and independent publication

Keep the main story in three parts: contract -> matched routing -> controlled consumption. Replace numbered-paper dependencies in the introduction and conclusion with short ordinary citations and a one-paragraph scope boundary. The eight-stage vocabulary is useful as a table, but repeating the entire chain in the abstract, introduction, results, and conclusion obscures the empirical finding.

Suggested wording: “On eight QASPER test questions, the residual adapter improves mean polarity accuracy across five optimization seeds. This result motivates a larger replication; exact no-memory behavior is separately guaranteed by the wrapper's bypass path.”

This paper should own family integration, active-path diagnostics, and conditional adaptation. It should cite Paper 1.5 for the controlled geometry result rather than importing its headline numbers into every introduction. Paper 2.5's graph search and Paper 2.6's channel choices remain separate contributions.

Suggested audience: ML software/architecture. A software-centered **JMLR MLOSS** version is a possible route if the reusable implementation becomes the main contribution and the package satisfies that track's requirements; MLOSS explicitly covers substantial ML implementations and toolboxes. [Official MLOSS page](https://jmlr.org/mloss/). A full scientific version can instead target a research journal, but the positive task claims need stronger independent evidence. These are alternative editorial directions, not two submissions of substantially the same contribution.

## Sol handoff and completion criteria

First correct units, main-table denominators, judge contrasts, cohort comparisons, and evidence tiers. Build a manifest for every headline: exact source artifact, split, unique questions, seeds, metric, memory mode, generation budget, and baseline. Then revise the abstract around what remains supported. Propose a fresh QASPER confirmation separately; do not add more adapters merely to increase the contribution count. Finish with a concise change log and all unresolved empirical requests.

Source preflight found no missing literal inputs, unresolved cited keys, undefined cross-reference labels, or duplicate labels. This does not certify adapter correctness or full numerical/PDF reproducibility.
