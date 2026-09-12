# Proposed publication review — Paper 2.6

Reviewed 2026-09-11 against fetched branch `hybrid-pra` at `e30510cb691feffa243d55c14a2a39cef21cef50`, equal to `origin/hybrid-pra`. Current source: [paper2_6_v4.tex](paper2_6_v4.tex). The latest paper includes fresh two-model Gate B, indexed retrieval, and native-K/V consumption; this is not a review of the earlier discovery-only version. Review includes selected stored JSON/CSV results, but no inference rerun or PDF layout certification. Line anchors refer to this commit.

## Assessment and independent contribution

**Major revision, with a stronger empirical core in the latest version.** The useful result is that representation rankings vary by task and that better evidence recall does not guarantee better frozen-model consumption. Sparse index measurements add a systems contribution. Static or adaptive hybrid retrieval does not consistently beat the strongest individual channel, so “hybrid PRA improves QA” would be the wrong headline.

**Proposed main claim:** “Under a bounded native-K/V interface, lexical and contextual retrieval channels expose different evidence. Their retrieval rankings transfer across two small model families, while downstream likelihood gains remain task dependent.” Index efficiency is a second claim with its own measurement boundary.

## Priority revisions

### P0 — Make evidence-first source construction part of every main-result claim

**Locations:** abstract; Gate B at 130–137; limitations at 327 onward.

Gate B prepends annotated evidence strings, appends the original source, truncates to 1,536 tokens, and chunks the result. The draft discloses this, which is good, but the abstract still reads like ordinary benchmark retrieval. Gold-free scoring does not make the complete pipeline annotation independent. Source construction can change evidence location, prevalence, duplication, and lexical overlap.

Describe the main setting as an annotation-constructed mechanism benchmark in the abstract and primary table caption. Add a cheap first-four-chunks baseline: front-loading evidence makes this essential. Record whether prepended evidence also survives in the original-source portion and how duplicate evidence affects scoring.

If retaining claims about ordinary document retrieval, add an original-order source condition without gold-dependent construction, with a declared policy for examples whose evidence falls beyond the token limit. Randomized evidence location and a deduplicated version would isolate the shortcut more directly. Otherwise keep the present controlled scope and avoid implying unchanged HotpotQA/QASPER evaluation.

### P0 — Quantify channel differences on the same held-out identities

**Locations:** main channel table at 139–154.

The repeated ordering across Qwen and SmolLM is interesting: BM25 leads HotpotQA (.665/.662), and approximate matching leads QASPER (.747/.783). But both models use the same 50 held-out question identities per dataset, so this is model replication on a shared sample, not two independent task samples.

Report paired question-level intervals for the key differences: winning channel versus semantic, hybrid versus the best fixed channel, and adaptive versus its frozen comparator. Explain validation-only selection of those comparisons and distinguish descriptive rankings from statistically resolved differences. Five selector seeds measure optimization variability, not five new datasets.

The adaptive channel falls below the best fixed channel in all four displayed model/dataset panels. An architecture may expose multiple representations without having demonstrated a successful runtime selector. Describe the learned/adaptive result as a limitation and make state-dependent selection a hypothesis where appropriate.

### P0 — Show actual routed-content controls, not only oracle controls

**Locations:** consumption at 163–203; `paired_causal_effects.csv` and `end_to_end_findings.json`.

The draft appropriately separates retrieval, likelihood, and generated QA. Strengthen that distinction with the comparisons already available. For Qwen HotpotQA, semantic memory improves mean answer log-probability over disabled memory by about .744, and over irrelevant and shuffled memory by about .426 and .410, with the stored paired intervals excluding zero. These routed-content comparisons more directly support a content-specific effect than oracle-versus-irrelevant alone.

However, semantic-minus-disabled generated F1 is only about +.00894, with interval [−.02971, .04589]. It does not establish a generated-answer improvement. BM25 has the best evidence recall but a smaller likelihood gain than semantic retrieval; its shuffled-memory contrast is also less resolved. This mismatch is scientifically interesting and should be a main result.

For QASPER, approximate-channel likelihood differences from disabled memory remain unresolved, even though oracle beats irrelevant memory. State all three conclusions distinctly: evidence can be found, an idealized payload can have an effect, and the tested routed method has not shown a reliable answer benefit there. Give the gold-answer likelihood normalization and treatment of multiple accepted answers explicitly.

### P0 — Bound the indexed-retrieval claim to what was tested

**Locations:** index at 109–128 and detailed index appendix.

The prototype preserves exhaustive Top-4 on every measured paired query while scoring roughly 10% of candidates. A bounded union of postings plus a reserved semantic slice does not, by itself, prove exact Top-4 preservation for arbitrary queries or corpora. Phrase this as empirical parity on the suite unless a valid bound or complete fallback provides a general guarantee.

Describe every candidate-generation cost, including the semantic reserve. Separate cold build, warm lookup, updates, memory/object overhead, and final candidate scoring. The reported 68.4× mean routing speedup and 283–300× exact-channel speedup are against the measured Python exhaustive implementation; they are not speedups over a production search engine or end-to-end model inference.

If generalizing parity, add queries with low lexical overlap, paraphrases, typo-heavy rare terms, ties, and relevant items outside the reserved pool. Compare indexing efficiency to an appropriate optimized lexical baseline before claiming a general systems advantage. Fuzzy routing at roughly half a second remains a meaningful limitation, already acknowledged by the draft.

### P1 — Separate the four cohorts and explain changed winners

**Locations:** Gate B at 132; breadth at 205; root/successor analysis at 235–269; interaction at 271 onward.

Add one cohort table with unique identities, validation/test use, source construction, models, seeds, and purpose. Preserve the serialized disjointness audit. The old QASPER panel favors exact matching whereas fresh Gate B favors approximate matching; these are different cohorts/settings and should not be read as the same regime yielding interchangeable numbers.

The eight-example-per-dataset interaction grid is descriptive despite its many configurations. Report identity counts prominently and avoid converting a large number of conditions into a large sample claim. Root and successor comparisons also need a distinction between oracle roots, observed best channels, and an executable policy. “Different channels win different subtasks” does not establish that a deployed selector can choose those channels reliably. For unordered evidence collections, explain what successor recovery means before calling it reasoning traversal.

### P1 — Repair units and baseline fairness

**Locations:** metric definitions at 47–80; budget/cost discussion near 190–203.

Use set cardinalities consistently: selected fraction `|S|/N`, recall `|S∩E|/|E|`, and precision `|S∩E|/|S|`. State how empty evidence and duplicate spans are handled.

Audit the claimed 127.8–128 realized tokens across channels: stored summaries include approximately 127.32 for semantic and 127.62 for exact. Minor differences need accurate reporting rather than a fictitious identical realized payload. Trace the reported active MiB to model K/V head counts, head dimensions, dtype, layer count, and actual allocations; distinguish selected token identities from layer-token states and decimal MB from MiB. “Disabled has no active K/V” should mean no external admitted memory, not no native decoder cache.

The direct-context F1/runtime point is a separate operating point. It cannot establish a quality-matched speedup. Show the quality-cost pairs and distinguish source encoding/index amortization from per-query generation.

Explain that BM25 introduces lexical term extraction even when using model-token-derived decoded text. “No second NLP vocabulary” should not obscure how terms, document frequencies, and approximate matches are constructed. A strong learned semantic retriever would be a useful comparator if making claims about lexical versus semantic retrieval generally; raw contextual gists alone do not represent all semantic retrieval methods.

## Readability and reviewer interest

Suggested title direction: **“Retrieval Representation and Native-State Consumption Under a Bounded Memory Budget.”**

Shorten the abstract to the question, controlled setting, two decisive results, and limitation. Replace “Gate B” in reader-facing headings with “Held-out retrieval and consumption evaluation”; retain the campaign identifier in artifact metadata. Lead with a simple example where exact identity, approximate spelling, and contextual similarity rank different memory objects. Then explain that ranking and useful consumption are separate tests.

Keep the index and consumption interfaces visually distinct. Avoid asking one paragraph to argue representation heterogeneity, runtime policy, asymptotic scalability, and QA benefit simultaneously. The candid retrieval-to-consumption counterexample is more persuasive than an unsupported universal-score claim.

## Independent publication boundary and journal direction

Own **representation choice, sparse discovery, and the controlled handoff to a fixed materializer**. Paper 2 owns model integration and compatibility; Paper 2.5 owns topology and iterative traversal; Paper 1.5 owns position-frame fidelity. Include a self-contained PRA interface and cite the companion work for broader context. Attribute reused datasets, code, and legacy panels explicitly.

The likely audience is information retrieval or empirical AI systems. Venue fit is provisional. The former Information Retrieval Journal is now **Discover Computing**, with a broader computing scope; use the current title if considering it, rather than planning a submission to the historical journal name. See its [official scope](https://link.springer.com/journal/10791/aims-and-scope). A specialized retrieval venue would require a particularly strong novelty comparison and conventional retrieval baselines. Do not choose a venue solely to keep all six journal names different.

## Sol handoff and acceptance checklist

1. Confirm this branch and source version; do not revise a stale `paper.tex` from another worktree.
2. Correct setting, metric notation, realized budgets, and causal/QA wording from existing artifacts first.
3. Promote routed-content controls and paired channel comparisons; attach every table to a cohort and artifact.
4. Choose between a narrowly scoped mechanism article and new original-order/prefix-baseline evidence for broader retrieval claims. Keep proposed runs explicitly unperformed until artifacts exist.
5. Constrain index parity and speedup claims to the verified suite and baseline implementation.
6. Rewrite the main narrative, compile, inspect tables/figures, and return unresolved claims and author decisions separately.

Literal input/citation/reference/label preflight found no unresolved items in the expanded source. Bibliographic relevance, all experimental claims, statistical validity, and rendered layout still require the targeted checks above. The manuscript itself has not been edited by this review.
