# Proposed publication review — Paper 0

Reviewed 2026-09-11. Branch: `main`; fetched `origin/main` and confirmed local/remote equality at `b50249700e5e764ef009599c6c79219a4cdb2da5`. Manuscript: [paper.tex](paper.tex); last manuscript change: `ce4cb5b8` (2026-08-28). Locations below refer to this snapshot. This is a source-based scientific/editorial review, with selected evidence checks, not a rerun of the research or a PDF layout certification. Existing manuscript and README are preserved.

## Assessment and independent contribution

**Major editorial revision; plausible independent position article.** The useful thesis is that naming, encoding, discovery, admission, and consumption should be separate interfaces, with explicit resource and correctness contracts. The virtual-memory analogy becomes scientifically interesting where it breaks: relevance-based access is fallible, missing context changes representations, and loading correct evidence does not guarantee useful output.

The manuscript currently also acts as a repository roadmap, implementation status report, tutorial, application prospectus, and historical experiment digest. That breadth obscures the argument and creates the largest overlap risk with the five empirical manuscripts and the separate survey. An independent journal article should persuade a reader of a specific architectural position without requiring interest in the author's publication program.

**Proposed central claim:** “Addressable external memory should be evaluated through separate contracts for retrieval, contextual state, active attention, and task utility; keeping these contracts distinct exposes failure modes that context length alone does not measure.”

Keep the counterfactual evaluation doctrine, the distinction between logical and physical memory, the limitations of the paging analogy, and the explicit admission that production economics remain unestablished.

## Priority revisions

### P0 — Convert the living program document into a fixed scholarly argument

**Locations:** abstract, lines 47–69; “Living Architecture Status,” 617 onward; program roadmap, 675 onward; historical evidence, 1290 onward; conclusion, 1654 onward.

The abstract describes a “mature” implementation gateway and lists results from several companions. The roadmap devotes multiple tables and a diagram to working paper numbers. These are useful project documentation, but they make the article look dependent on an unpublished series.

Move detailed status inventories, paper-number roadmaps, release claims, and chronological experiment tables to a versioned online supplement. Retain one compact evidence table in the article with columns: architectural proposition, strongest supporting observation, counterexample, unresolved test. Identify reused evidence explicitly as companion results, not original experimental contributions here. Do not relabel measured numbers “illustrative” merely to satisfy an outdated local instruction; distinguish actual measurements from conceptual examples truthfully.

**Done when:** the abstract, introduction, and conclusion make a complete argument without “Paper 1/2/3.5” as prerequisites; the article remains understandable if every companion is unavailable.

### P0 — Make the memory bound mathematically auditable

**Locations:** implementation invariants, 477–509; GPU footprint, 550–560; three granularities, 562–571.

The target GPU equation has no explicit index-memory term, even though the text admits index growth. Hiding a corpus-dependent index in `M_runtime` makes the intended bounded-memory signature difficult to test. Add separately named routing/index, resident detail, active attention/workspace, direct history, and transfer-buffer terms; distinguish concurrently resident tensors from a sum of layer traffic.

The inequality `T_materialized <= T_requested <= T_logical` needs a common unit: unique source positions at one layer. Physical duplicates, padding, and layer-summed K/V can violate that ordering without any implementation error. State that restriction immediately. Replace `G_encode != G_search != G_materialize` with “independently configurable”; useful settings can have equal granularities.

**Done when:** every equation specifies whether it counts unique positions, padded positions, layer-token states, bytes, peak allocation, or cumulative work.

### P0 — Separate semantic retrieval from address resolution

**Locations:** URI memory, 186 onward; “Where the Analogy Breaks,” 861–875; conceptual computation, 335 onward.

URI-to-object resolution can be deterministic; query-to-useful-object selection can be learned or heuristic. Calling PRA translation categorically “learned and content-addressed” merges those operations. Define exact resolution, semantic selection, and speculative expansion separately. Also state when published state already contains the entire source versus when a true miss triggers encoding. Eagerly publishing all K/V followed by selective transfer is not evidence that cold source encoding is demand driven.

### P1 — Strengthen the competing explanation

The most serious alternative is that existing retrieval, prefix caching, and query-aware K/V selection already provide the useful working-set behavior, while URI naming is an application interface. Confront that argument directly. RetrievalAttention already combines vector retrieval with selective attention over stored K/V; the difference must concern the precise identity, lifecycle, contextualization, or admission contract, not generic external sparse memory. See [RetrievalAttention](https://arxiv.org/abs/2409.10516). CacheBlend already addresses contextual mismatch when reusing cached chunks; contextual encoding is therefore an important design requirement, not automatically a new discovery. See [CacheBlend](https://arxiv.org/abs/2405.16444).

Add a short “When PRA is unlikely to help” paragraph: low reuse, dense evidence dependence, cheap text retrieval, frequently changing sources, and routing/transfer dominating saved attention. Use an amortization equation with cold publication cost divided by actual reuse count. This gives reviewers a reason to trust the position.

### P1 — Make falsification proportionate and selection honest

**Locations:** scaling proposal, 877–895; profile calibration, 1028–1047; falsification, 1631 onward.

Separate rejection of a particular router or workload from rejection of the entire architecture. Failure of a learned selector to beat an oracle is not a sensible falsification criterion; use a feasible matched-budget baseline. A failed 8M-token experiment would narrow a scaling claim, not invalidate exact native-K/V transport.

“Quality-max selects the held-out quality optimum” should say validation-selected optimum, followed by a separate untouched test. The later calibration list already gestures toward that distinction; make terminology consistent.

## Suggested readable structure

1. A concrete selective-memory problem and one clear thesis.
2. Minimal architecture and an end-to-end worked example.
3. Four or five contracts and failure cases.
4. Comparison with the closest existing memory interfaces.
5. What current evidence supports and does not support.
6. Cost model, decisive experiments, and limitations.

Retain two representative applications rather than five broad application subsections. Keep embodied/cognitive analogies only where they yield a measurable prediction. Explain “gist” as a small routing vector at first use. Avoid “effectively unbounded” in headline text; “logical memory larger than the active attention budget” is both clearer and supported.

**Example opening:** “A language model may need access to a repository while using only a few functions to answer one question. Increasing its context window expands access, but does not specify which information should become active or when. We argue for an explicit memory interface that separates those decisions.”

## Publication boundary and journal direction

This paper should own the architectural thesis and evaluation framework. Paper 1 owns measured bounded transport; Paper 1.5 owns positional and routing geometry; Paper 2 owns pretrained integration; Papers 2.5/2.6 own traversal and retrieval-channel experiments. Summarize their evidence briefly with stable citations and disclose shared material. Do not repeat their principal result tables as this paper's independent contribution.

My tentative journal direction is **Artificial Intelligence Review**, as a critical architectural perspective/commentary rather than a second comprehensive survey. Its stated scope includes critical evaluations, tutorials, surveys, and commentary; this is a scope match, not an acceptance forecast. [Official scope](https://link.springer.com/journal/10462/aims-and-scope). If the separate PRA survey targets that venue, reserve different article purposes and avoid duplicating its taxonomy.

## Instructions for Sol's next pass

Complete the P0 textual/mathematical changes first, using existing evidence. Produce a revised abstract, a reduced main-text outline, and a claim-to-source table before expanding experiments. Freeze the status date and give each retained companion result a retrievable revision. Preserve negative observations. Treat the 32K–8M experiment as a proposed program, not required work for this position article. Finish with a change log mapping each review item to the revised section and an explicit list of unresolved evidence gaps.

Source preflight found no missing literal TeX inputs, unresolved citation keys, undefined `ref`/`eqref`/`pageref` labels, or duplicate labels. This check does not establish that citation targets support the claims or that the PDF renders correctly.
