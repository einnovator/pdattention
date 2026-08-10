# AGENTS-PAPERS-REVIEW.md — Deep Editorial, Pedagogical, Reproducibility, and Citation Pass

## Mission

Perform a thorough editorial and scientific-communication pass over PRA Paper 1 and Paper 0.

Primary goals:

- strengthen the abstracts, especially with the newest bounded-context results;
- make both papers easier to read and easier to follow;
- improve pedagogical value without reducing rigor;
- explain clearly what each experiment tests, what the result means, and why it matters;
- make the discovery path visible rather than presenting the final architecture as if it appeared fully formed;
- improve replication/reimplementation detail;
- audit references and add clickable final/preprint links where available;
- keep the mathematics sound but avoid pedantry;
- keep claims cautious and precise;
- make the papers engaging enough that a technically capable reader who followed *Attention Is All You Need* can follow PRA without specialist background.

The intended reading experience is:

> problem → mechanism → unexpected failure → diagnosis → refinement → scaling → systems bottleneck → optimization → bounded logical context → implications.

Do not turn the papers into marketing material. Preserve negative results, limitations, and uncertainty.

---

## 1. Rewrite and strengthen Paper 1 abstract

The abstract should answer, compactly:

1. What is PRA?
2. What problem does it address?
3. What is the core mechanism?
4. What are the strongest controlled results?
5. What important systems results are demonstrated?
6. What is not yet claimed?

Include the newest bounded-context finding.

Use the current result artifacts and exact values.

The abstract should communicate that PRA separates logical context size from the maximum context processed by the base model in any one native operation.

A suitable current result to include, if still exact in the artifacts:

> Under a hard 32-token native-operation limit, PRA used a 184-token displaced prompt head without violations; at 192 logical tokens, routed loss was 1.0567 versus 1.2287 for truncation and 1.7089 for independent block encoding.

Also include only the most informative subset of:

- historical/native-KV slicing fixing high-split fragmentation;
- scaling through 256 addressable units;
- exact tensorized routing speedup;
- warm packed-index routing if stable;
- CPU-resident K/V if still a headline result;
- `#__head`;
- materialization budgeting;
- streaming rollover.

Do not overload the abstract with every number.

Retain the qualification that controlled answer-code probes are not unrestricted QA.

---

## 2. Paper 1 should read as a discovery story

Review section order. Prefer approximately:

```text
Why long context is expensive
→ PRA intuition
→ references / routing / native-KV memory
→ first controlled probes
→ fragmentation failure
→ diagnosis
→ encoding granularity vs routing granularity
→ historical/native slicing
→ recovery and scaling
→ top-k / gist sensitivity
→ Python/CUDA routing bottleneck
→ exact tensorized routing
→ packed-index reuse
→ GPU-residency problem
→ CPU-resident K/V
→ logical context beyond native limit
→ bounded encoding
→ materialization budget
→ streaming rollover
→ implications / limitations
```

Reorganize if the current ordering interrupts this progression.

Do not preserve structure merely because it already exists.

---

## 3. Paper 0 should remain conceptual

Paper 0 is the architecture/position paper.

Its job is to explain:

- the central idea;
- why progressive disclosure of native K/V is useful;
- direct context vs addressable memory;
- logical context vs native model context;
- relationship to RAG, sparse attention, KV selection/compression, and recurrent/external memory;
- what current controlled evidence supports;
- what remains unproven.

Keep it shorter and more conceptual than Paper 1.

Do not duplicate Paper 1 tables and implementation detail.

Do a full stale-claim pass.

---

## 4. Make the discovery path explicit

Where scientifically useful, explain:

- initial assumption;
- surprising result/failure;
- diagnosis;
- architectural change;
- follow-up result.

### Fragmentation example

Make the lesson explicit:

**A routing chunk is an addressable retrieval unit, not necessarily an appropriate independent encoding unit.**

Explain that high-split degradation revealed contextualization fragmentation, motivating separation of:

**encoding granularity** and **routing granularity**.

### Routing-speed example

Explain that the first router demonstrated sparse selection but was dominated by Python/CUDA orchestration, not memory attention.

Then explain that packed tensor scoring + `torch.topk` preserved exact semantics while removing that implementation artifact.

### Native-context example

Explain that real pretrained models impose a hard per-operation context limit, so logical context must be separated from bounded encoding and materialization.

---

## 5. Use restrained bold emphasis

Highlight key ideas/results throughout the papers.

Good examples:

- **Encoding granularity and routing granularity are distinct.**
- **Routing gists are selectors, not the transported memory.**
- **Selected chunks materialize full native token-level K/V.**
- **Active K/V is not the same as total resident K/V.**
- **Exact tensorized routing preserves routing semantics.**
- **Logical context can exceed the base model's native context.**
- **No underlying model operation may exceed the configured native limit.**
- **Overlap-aware bounded encoding mattered more than the hard limit itself in the controlled experiment.**
- **Materialization budget is an independent quality/compute control.**

Do not over-bold.

Aim for one or two meaningful bold statements per major subsection where useful.

---

## 6. Frame every major experiment

Before each important experiment/table/figure, add one or two sentences answering:

> What question is this experiment testing?

After it, explain:

> What does this result mean?

Example:

> This experiment tests whether the high-split failure is primarily a routing-capacity failure or a contextualization failure caused by independently encoding tiny fragments.

After the result:

> The recovery under historical/native slicing indicates that the severe degradation was primarily an encoding-context artifact rather than a fundamental inability to route among many addressable units.

This is a high-priority edit.

---

## 7. Separate observation, interpretation, and claim

For major findings, prose should distinguish:

```text
Observation — what was measured.
Interpretation — what mechanism best explains it.
Claim — what conclusion is justified.
```

Literal labels are optional.

Avoid jumping from benchmark values directly to broad capability claims.

---

## 8. Improve figures and tables pedagogically

Each important table/figure should have:

1. a useful caption;
2. a sentence before it explaining why the comparison matters;
3. a paragraph after it identifying the main pattern.

Do not force the reader to infer the conclusion from raw columns.

Use plain-language summaries such as:

> Routed loss remains nearly flat as logical context grows under overlap-aware bounded encoding, while truncation and independent block encoding degrade substantially.

Then cite exact values.

---

## 9. Elevate the bounded-context result

Make clear that the newest experiment separates three effects:

### Hard native limit
The base model is restricted to a strict maximum native operation size.

### Encoding fragmentation
Long source/history must be encoded in bounded blocks; how context is preserved across boundaries matters.

### Materialization budget
Routing may find relevant memory, but too small a materialization allowance may still limit recovery.

Explain the current result clearly:

- the hard 32-token limit itself did not cause a large routed-quality collapse;
- overlap-aware bounded encoding retained strong performance;
- independent block encoding was substantially worse;
- increasing materialization budget improved recall/loss.

This should be one of Paper 1's headline conceptual findings.

---

## 10. Explain the context scales simply

Use:

\[
L_{logical} \gg L_{model,max} \ge L_{encode} > L_{route}.
\]

Define each term immediately and give a concrete example.

Also explain:

\[
L_{direct} + L_{materialized} + L_{reserve} \le L_{model,max}.
\]

Use the notation because it clarifies architecture, not for ceremony.

---

## 11. Add or improve the main architecture diagram

Paper 1 should have one clear diagram showing:

```text
logical source / long prompt
        ↓
model-safe encoding blocks
        ↓
native K/V
        ↓
smaller routing chunks
        ↓
gists / packed index
        ↓
exact routing
        ↓
materialization budgeter
        ↓
selected native K/V
        ↓
direct + memory attention
```

Also consider a compact streaming diagram:

```text
direct tail grows
→ oldest tokens roll into #__head
→ direct window stays bounded
→ old content remains retrievable
```

Keep diagrams simple and functional.

---

## 12. Mathematical rigor without pedantry

Review every equation.

Keep equations that:

- define PRA;
- clarify tensor shapes;
- establish context/budget invariants;
- distinguish routing from attention;
- make implementation reproducible.

Simplify or remove equations that only restate obvious code.

For every nontrivial equation:

- define symbols immediately;
- give shapes where useful;
- add one plain-language explanation.

Never require the reader to search several pages backward for a symbol.

---

## 13. Tensor-shape pass

Paper 1 should expose enough shapes for reimplementation.

At minimum:

```text
query
gist matrix
routing score matrix
selected chunk indices
native K/V
direct K/V
materialized memory K/V
attention logits
```

Use consistent notation.

Do not document every reshape.

---

## 14. Replication / reimplementation audit

A competent PyTorch researcher should be able to implement a minimal PRA version from the paper plus the referenced source.

Ensure the paper covers:

1. reference parsing/resolution;
2. bounded encoding blocks;
3. routing-chunk slicing;
4. native K/V capture;
5. gist construction;
6. packed gist index;
7. exact tensorized routing;
8. hierarchical reference/chunk selection;
9. materialization budgeting;
10. CPU/GPU residency;
11. direct-memory attention composition;
12. `#__head`;
13. streaming rollover;
14. metrics.

Add concise pseudocode where needed.

---

## 15. Improve stable line-numbered pseudocode

Use compact PyTorch-style pseudocode with stable pseudo line numbers.

Recommended moving-part subsections:

```text
A. source preparation
B. bounded encoding
C. native-KV slicing
D. gist construction
E. packed index
F. exact routing
G. budgeted selection
H. materialization
I. PRA attention
J. streaming rollover
```

For each:

- concise code;
- tensor shapes;
- explanatory prose.

Do not paste the whole repository implementation.

---

## 16. Source-file/function references

Where practical, include actual implementation references.

Prefer:

```text
src/pra_torch/memory.py::ClassOrFunction
src/pra_torch/attention.py::Function
```

and current line ranges if stable.

Before finalizing, verify line numbers against current HEAD.

Do not leave stale line references.

Function/class names are more robust than line numbers; use both when possible.

---

## 17. Add one minimal end-to-end PRA algorithm

Include one compact algorithm summarizing inference:

```text
1. keep recent prompt tail direct
2. move overflow to #__head
3. partition long sources into bounded encoding blocks
4. encode native K/V
5. slice into routing chunks
6. build gists/index
7. score candidates
8. obtain ranked chunk candidates
9. enforce materialization budget
10. materialize/transfer selected K/V
11. combine with direct attention
12. during generation, roll expired direct context into #__head
```

It should fit on about one page or less.

---

## 18. Explain exact tensorized routing clearly

Explain the old bottleneck:

```text
Python candidate loop
→ many tiny CUDA calls
→ .item() / .cpu() synchronization
→ full ranking serialization
```

Then the new path:

```text
packed gist tensor
→ matrix scoring
→ GPU aggregation
→ torch.topk
→ selected result transfer
```

Highlight:

**The optimization changes how the same exact scores are computed and selected; it does not introduce approximate retrieval.**

---

## 19. Explain warm vs cold routing

If present, define:

```text
cold routing = index construction + query
warm routing = query against persistent packed index
```

Explain why warm mode matters for:

- autoregressive decoding;
- repeated questions over same documents;
- sessions;
- agent memory.

Do not mix cold/warm comparisons without labels.

---

## 20. Explain memory accounting carefully

Make the distinction impossible to miss:

```text
routing gists/index
≠ full stored source K/V
≠ selected transferred K/V
≠ active attention K/V
```

Avoid ambiguous statements such as “97% KV savings.”

State exactly whether a number means:

- active attention reduction;
- GPU-resident reduction;
- transfer reduction;
- total cache reduction.

---

## 21. Make `#__head` intuitive

Use a concrete example before formalism:

```text
32K logical prompt
4K direct budget

first 28K → #__head
last 4K → direct context
```

Explain:

- it is not a special attention mechanism;
- it reuses PRA reference/chunk/gist/cache mechanics;
- very large heads are encoded in bounded blocks;
- routing chunks may be smaller than encoding blocks;
- streaming moves expired direct tokens into the head/history.

---

## 22. Explain independent-encoding failure strongly

This should be a memorable conceptual result:

**A routing chunk is an addressable retrieval unit, not necessarily an appropriate independent encoding unit.**

Explain why:

```text
large contextual encoding block
→ many small routable native-KV slices
```

can outperform:

```text
many tiny independently encoded fragments
```

under the measured conditions.

---

## 23. Dense attention is a control, not automatically an oracle

Where routed PRA outperforms a dense/full condition, explain possible causes:

- tiny-model training distribution;
- distractor competition;
- context-length extrapolation;
- all tokens remaining active.

Use cautious framing:

> In this controlled regime, dense access is a useful reference condition but not a guaranteed upper bound on loss.

Do not generalize to “sparse is better than dense.”

---

## 24. Consider a short clarification box

If it improves readability, add a compact box/sidebar:

```text
PRA is not RAG preprocessing.
Gists are not the final memory representation.
Top-k chunks are not top-k tokens.
Selected chunks materialize full native K/V.
Logical context may exceed native model context.
The base model is never asked to process the entire logical context at once.
```

Keep it concise.

---

## 25. Related-work restructuring

Organize related work by mechanism instead of a flat list.

Suggested groups:

### Sparse attention
Longformer, BigBird, etc.

### KV selection/compression
H2O, SnapKV, ClusterKV, etc.

### External/retrieved memory
RETRO, kNN-LM, RAG-adjacent work, etc.

### Recurrent/long-term memory
Transformer-XL, Compressive Transformer, Infini-Attention, Titans, FAM, etc.

### Landmark/gist/routing methods
Landmark Attention and related work.

For each group, state the differentiating dimension of PRA in one clear sentence.

Avoid exaggerated novelty language.

---

## 26. Reference-link audit

Audit the entire bibliography.

For each reference where possible, provide clickable:

1. final publisher/conference page if open;
2. DOI;
3. arXiv/preprint;
4. official project/author page when appropriate.

If the final version is paywalled and a preprint exists, include the free preprint link as well.

Use stable URLs.

Avoid search-result URLs.

---

## 27. Citation accuracy

For every important prior-art claim:

- verify metadata;
- verify the cited paper actually supports the characterization;
- prefer primary sources;
- update preprints to final publication metadata where appropriate;
- retain free preprint URLs.

Do not use “first,” “unique,” or similar novelty claims without strong support.

---

## 28. Bibliography hygiene

Standardize:

- author names;
- title capitalization;
- venue;
- year;
- DOI;
- arXiv;
- URL.

Remove duplicates.

---

## 29. Improve transitions

Start major sections with an orientation sentence.

Example:

> The previous experiment showed that PRA can identify useful memory. The next question is whether the stored native K/V remains faithful as the source is divided into many addressable units.

Use section endings/takeaways sparingly:

> **Takeaway.** The high-split failure was primarily an encoding-context problem, motivating separation of encoding and routing granularity.

---

## 30. Terminology audit

Use terms consistently:

```text
reference
encoding block
routing chunk
gist
native K/V
materialization
direct context
logical context
native/model context
active K/V
resident K/V
selected K/V
#__head
streaming history
routing index
```

Avoid using “chunk” alone when encoding vs routing matters.

Define important terms clearly on first use.

---

## 31. Reduce engineering-log tone

Search for prose like:

```text
we patched
we added a flag
the code currently
this test checks
```

Rewrite main scientific prose into:

```text
we introduced
we evaluated
the architecture enforces
the experiment isolates
```

Implementation subsections may still use code/config terminology.

Paper 1 should not read like release notes.

---

## 32. Preserve negative results

Keep and explain:

- multi-gist not universally helping;
- independent-encoding failure;
- cold CPU-offload cost;
- former scalar-routing bottleneck;
- limitations of answer-code probes;
- lack of broad SOTA pretrained validation.

Explain what each negative result taught.

---

## 33. Explain significance, not just numbers

For major results, add the architectural meaning.

### 256-unit scaling
Bounded active memory can remain useful while available addressable context grows.

### Exact tensorized routing
Sparse exact retrieval need not imply Python candidate enumeration or ANN at tested scales.

### CPU-resident K/V
Sparse active attention can become real GPU-capacity reduction by decoupling routing state from full source-K/V residency.

### Bounded `#__head`
Logical context does not have to fit in one native base-model forward.

### Streaming rollover
High-resolution recent context can become selectively retrievable history as generation continues.

---

## 34. Use intuitive examples before equations

Example:

```text
model max = 32
direct tail = 8
logical prompt = 192
old 184 tokens = #__head
```

Then introduce general notation.

This is strongly preferred over starting with abstraction alone.

---

## 35. Avoid unnecessary formalism

Do not introduce theorem/proposition language unless there is an actual theorem.

Use equations to clarify mechanisms and invariants.

The style should be mathematically precise, not ceremonial.

---

## 36. Visual consistency

Audit figures/tables for:

- font size;
- axis labels;
- units;
- legend terminology;
- decimal precision;
- model labels;
- seed reporting;
- split/context terminology.

Use precision appropriate to meaning.

Do not report excessive decimals except for exact-parity statements.

---

## 37. Reproducibility checklist

Add a concise reproducibility subsection or appendix table containing:

```text
model configs
training budgets
datasets
seed list
hardware
PyTorch/CUDA
routing parameters
gist parameters
encoding chunking
routing chunking
materialization budget
native context limit
KV residency
benchmark scripts
result artifact paths
```

A reader should not have to explore the repository blindly.

---

## 38. Map paper results to scripts/artifacts

For each major table/figure, provide in a reproducibility appendix/table:

```text
benchmark script/module
result JSON/CSV
figure-generation path
```

Keep raw paths out of main prose unless pedagogically useful.

---

## 39. Paper 0 stale-claim review

Search Paper 0 for stale statements about:

- split-64 failure;
- routing latency;
- lack of indexed/tensorized routing;
- GPU memory;
- `#__head`;
- model-bounded context;
- streaming;
- current result/test counts.

Update using only demonstrated results.

---

## 40. Paper 0 abstract

Keep it concise and conceptual.

Explain:

- progressive disclosure of native K/V;
- direct context vs addressable memory;
- logical context beyond native model context;
- representative controlled evidence;
- limitations.

Use only one or two strong numbers if needed.

---

## 41. Paper 0 memorable framing

Early in the paper use a formulation close to:

> Recent context remains directly visible at high resolution; older or external context is retained as addressable memory and disclosed only when routing predicts that it is useful.

This should be one of the clearest descriptions of PRA.

---

## 42. Clarify PRA vs RAG

Simple framing:

```text
RAG:
retrieve text/documents around model inference.

PRA:
route and materialize layer-native K/V inside the model's memory/attention path.
```

State that they can be complementary.

Do not claim PRA universally replaces RAG.

---

## 43. Clarify PRA vs sparse attention / KV compression

Explain:

- sparse attention limits token interactions;
- KV compression approximates/reduces stored state;
- PRA can preserve exact native token K/V for selected chunks while sparsifying which memory is exposed;
- alternative materialization strategies are future work.

---

## 44. Tone

Motivate significance without hype.

Good:

> For serving systems, reducing active and GPU-resident K/V can affect memory capacity, batch size, and latency.

Avoid grandiose claims.

Let measurements carry the argument.

---

## 45. “Adventure of discovery” style

Use restrained narrative momentum:

> The first scaling experiment produced an unexpected failure.

> That failure turned out to be informative.

> Once encoding and routing granularity were separated, the picture changed sharply.

> The next bottleneck was no longer model quality but the prototype router itself.

> Removing that implementation artifact exposed the next systems constraint: where inactive native K/V resides.

Encouraged.

Do not become theatrical.

---

## 46. Reduce repetition

Paper 1 is long.

Search for repeated full explanations of:

- gists vs native K/V;
- active vs resident K/V;
- routing vs encoding chunks;
- routing-speed caveats;
- answer-code qualification.

Keep one full explanation and shorter reminders later.

---

## 47. Introduction roadmap and contributions

At the end of Paper 1 introduction, provide a narrative roadmap, not just section numbers.

Update the contribution list to current demonstrated contributions only.

Possible categories:

1. PRA native-KV architecture;
2. contextual/historical slicing result;
3. exact tensorized routing + packed index;
4. residency/transfer separation;
5. bounded logical context with `#__head`, encoding/routing separation, materialization budget, streaming;
6. reproducible implementation/diagnostics.

Do not list future work as a contribution.

---

## 48. Limitations

Make limitations concrete:

- tiny/small custom models;
- controlled answer-code probes;
- no broad pretrained SOTA validation yet;
- RoPE positional study deferred to Paper 1.5;
- serving benchmarks still incomplete where true;
- overlap/historical encoding tradeoffs;
- CPU transfer cost;
- whole-chunk materialization;
- no claim of unlimited semantic memory capacity.

---

## 49. Future-work organization

Group future work:

```text
pretrained/HF integration
RoPE positional semantics
alternative materialization
ANN at very large indexes
tiered/distributed residency
semantic/marker-aware chunking
agent/harness adaptation
```

Remove already-completed items.

---

## 50. Verify all numerical claims

Search every:

- percentage;
- latency;
- loss/RCB;
- context length;
- split count;
- test count;
- speedup.

Verify against current artifacts.

Delete stale values.

---

## 51. Build and visually inspect PDFs

After editing:

1. build Paper 1;
2. build Paper 0;
3. resolve all citations/references;
4. inspect every page visually;
5. check tables/figures;
6. check bold emphasis;
7. check hyperlink behavior;
8. check bibliography links;
9. check print readability;
10. ensure no accidental local filesystem paths leak into prose.

---

## 52. Reader-simulation pass

Do a final pass pretending to be:

> a strong ML researcher familiar with Transformers but new to PRA.

For every section ask:

- Why am I reading this section?
- Was the problem explained before the solution?
- Are symbols defined?
- Do I understand what the experiment tests?
- Do I understand what the result means?
- Can I distinguish observation from interpretation?
- Could I reimplement the mechanism?
- Does the section motivate the next one?

Fix sections where the answer is no.

---

## 53. Final deliverables

Required:

```text
updated Paper 1 .tex + PDF
updated Paper 0 .tex + PDF
updated BibTeX/bibliography
updated figures/tables only where needed
reproducibility appendix/table
updated roadmap if the review uncovers documentation/technical gaps
```

Then:

1. run paper build checks;
2. run relevant tests if source-line references/code docs were touched;
3. commit;
4. push to `main`;
5. report commit SHA;
6. summarize major editorial/scientific changes;
7. list any claims/references that could not be fully verified.

---

## Quality bar

The final papers should be:

**rigorous but readable; mathematical but not pedantic; technically detailed but not dry; cautious in claims but clear about significance.**

The target reader reaction is:

> “I understand the idea, I understand why each design decision emerged, I know what the experiments establish, and I could implement a minimal PRA system myself.”

The papers should feel like a sequence of discoveries rather than a feature changelog.
