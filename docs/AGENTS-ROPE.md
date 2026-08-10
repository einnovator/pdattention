# AGENTS-ROPE.md — Paper 1.5: Positional Semantics for Retrieved Native-KV Memory

## Mission

After the model-bounded PRA work is stable, create a controlled RoPE research program and a standalone Paper 1.5.

Do not make this merely “PRA supports RoPE.” Investigate the broader question:

> How should positional information be represented, preserved, transformed, or compressed when transformer K/V is encoded in bounded blocks, stored externally, retrieved later, and used at logical distances beyond the model's native context?

PRA is the experimental architecture, but frame findings for retrieved KV memory, KV paging/offload, streaming/recurrent memory, and chunked long-context inference generally.

Do not start with Hugging Face. Use controlled tiny/small models first, then use the findings to de-risk Paper 2.

## Core hypotheses

Test:

1. **RoPE block-translation invariance.** Common translation of a block preserves within-block Q/K positional geometry much better than learned absolute positioning.
2. **Fragmentation decomposition.** Under RoPE, forced-block degradation should be dominated more by lost cross-block contextualization than by positional-coordinate relocation.
3. **Overlap repairs fragmentation.** Moderate encoding overlap/historical context should recover much of the independent-block gap.
4. **Portable K representation.** Pre-RoPE K plus position metadata and deferred rotation may be more portable for retrieved memory than post-RoPE K with baked-in phase.
5. **Remote distance compression.** Exact metric distance may become less important for remote memory than local order, global ordering, and coarse past/near/far relations.
6. **Routing is mostly semantic.** Gist routing should not require precise absolute position unless experiments demonstrate value.

## Matched models

Train otherwise-matched models:

- learned absolute positional embeddings;
- RoPE;
- strongly desired: sinusoidal absolute positions;
- optional: ALiBi.

Keep architecture, dimensions, heads, FFN, vocabulary, optimizer, training tokens, datasets, PRA configuration, and seeds matched.

Use at least tiny and small tiers. Target five seeds for main claims where practical.

## RoPE implementation

Implement RoPE explicitly and inspectably. Expose:

```python
q_raw
q_rope
k_raw
k_rope
v
```

Provide testable RoPE functions rather than hiding the transform in an opaque fused kernel.

Unit-test mathematical translation behavior before model experiments.

## Experiment 1 — Pure positional translation

Hold Q/K content fixed. Compare a block at positions `0..L-1` with the same block at `c..c+L-1`.

Compare absolute, sinusoidal, and RoPE.

Measure:

- QK score difference;
- attention-distribution difference;
- KL/JS divergence;
- top-attended-token agreement;
- output-vector difference.

This experiment isolates positional representation from contextualization.

## Experiment 2 — Actual block relocation

Encode identical content under different block/local positions and compare:

- hidden states;
- raw K;
- rotated K;
- V;
- attention logits/distributions;
- outputs.

Keep this separate from Experiment 1 because re-encoding introduces contextual effects.

## Experiment 3 — Fragmentation decomposition

Reproduce Paper-1-style scaling with:

- dense historical encoding;
- independent blocks;
- overlapping blocks;
- historical-window encoding;
- historical native-KV slicing.

Compare absolute vs RoPE across increasing split counts/logical-context ratios.

Measure RCB, loss, routing recall/MRR/coverage, hidden/KV differences, and attention-output differences.

Main question:

> How much RoPE degradation remains after positional relocation is separated from missing contextualization?

## Experiment 4 — Encoding-overlap sweep

For sources larger than native context, sweep approximately:

`0%, 5%, 10%, 25%, 50%`

or a reduced sensible set.

Compare absolute and RoPE. Include marker-aware boundaries if already supported.

Measure quality, RCB, boundary errors, routing quality, encoding cost, and duplicate/context tokens processed.

## Boundary probes

Create deterministic dependencies crossing encoding boundaries. Vary dependency distance from the boundary, e.g.:

`1, 4, 16, 64, 128` tokens.

Use definition/query, key/value, relation-chain, and ordered-event probes.

This should reveal how much overlap is needed to repair contextualization loss.

## Experiment 5 — Pre-RoPE vs post-RoPE K

Implement controlled memory modes:

### Post-RoPE
Store `K_rope, V` using source-encoding positions.

### Pre-RoPE
Store `K_raw, V` plus logical/local span metadata and apply the chosen positional transform when selected memory is materialized/matched.

Compare under:

- same-position reuse;
- local position reset;
- block relocation;
- long logical distance;
- `#__head`;
- streaming rollover if available.

Measure task quality, RCB, attention distributions, materialization overhead, cache bytes, and warm latency.

Use precise terminology in Paper 1.5:

- pre-positional native K;
- post-RoPE native K;
- native V.

Do not ambiguously call both simply “native K.”

## Experiment 6 — Beyond-native logical distance

Train with native context `L`, then evaluate PRA logical contexts:

`1L, 2L, 4L, 8L, 16L, 32L`, and optionally `64L`.

The underlying model must never process more than `L` tokens in one native operation.

Use controlled answer-code/key-value tasks first.

## Experiment 7 — Retrieved-memory RoPE policies

For pre-RoPE K, implement a clean policy interface and compare:

- exact/logical relative distance where meaningful;
- local chunk position;
- clipped distance;
- strongly desired: log-compressed distance;
- bucketed distance;
- remote-past constant.

Examples:

```text
clipped: d_eff = min(d, D)

bucketed:
near / medium / far / very-far
```

Preserve exact local order within chunks.

Do not overfit policies to one benchmark.

## Experiment 8 — Distance precision ablation

Keep semantic content fixed while varying logical distance.

Test whether distinguishing, for example, 20K from 21K matters compared with treating both as remote history.

Plot quality versus positional-distance precision at increasing logical distances.

This is a central independent-paper question:

> How much positional resolution does retrieved transformer memory require as distance increases?

## Experiment 9 — Order vs exact distance

Construct memory chunks `A B C D`.

Compare:

- correct order;
- reverse;
- random permutation;
- correct order + exact distances;
- correct order + compressed distances;
- correct order + constant remote-past treatment.

Separate:

- content;
- local within-chunk order;
- global chunk order;
- exact metric distance.

Test whether coarse ordering is more important than precise long-range distance.

## Experiment 10 — Routing position ablation

Compare gist routing using:

- semantic gist only;
- semantic gist + normalized logical position;
- semantic gist + coarse position bucket.

If positional information does not improve routing, keep routing semantic and place positional semantics at native-Q/K materialization.

## `#__head` study

Use `#__head` as the continuous-history test.

Train at native context `L`; evaluate logical prompt sizes through at least `32L` where practical.

Compare:

- truncation;
- absolute-position PRA head;
- RoPE PRA head;
- RoPE + encoding overlap;
- RoPE + historical-window encoding;
- oracle memory;
- shuffled/wrong memory.

The answer must be outside the direct tail.

Track routing recall, RCB/task metric, active K/V, logical distance, encoding overlap, and positional policy.

## Streaming study

Once PRA streaming rollover is stable, generate beyond native context while expired direct tokens migrate into PRA memory.

Use controlled dependencies, not subjective free-form quality.

Record native context, generated length, logical-context ratio, rollover count, retrieved historical chunks, max native operation length, and task quality.

## Oracle controls

Every important positional experiment should include:

- normal routing;
- oracle correct chunk;
- wrong/shuffled chunk.

If normal fails and oracle succeeds, suspect routing. If oracle fails too, suspect encoding/materialization/position/model capacity.

## Isolate position from contextualization

Where possible compare:

1. same hidden/Q/K content + different positional transform;
2. separately re-encoded blocks with different available context.

Do not conflate these in plots or conclusions.

## Instrumentation

Optional debug captures, disabled by default:

- hidden states;
- q_raw/q_rope;
- k_raw/k_rope;
- V;
- attention logits/probabilities/output;
- logical positions;
- local encoding positions;
- effective retrieved-memory positions.

Useful mechanistic metrics:

- cosine similarity;
- L2/relative norm difference;
- attention-logit RMSE;
- KL/JS;
- top-k attention agreement;
- optional CKA if useful.

## Performance accounting

For post-RoPE versus pre-RoPE storage report:

- cache bytes;
- position-metadata bytes;
- materialization-time RoPE cost;
- attention time;
- warm query latency.

Semantics come first; do not prematurely optimize.

## Causality

All LM experiments must preserve causality.

No future source token may leak into an earlier prediction through block overlap, historical windows, or retrieved memory.

Add explicit leakage tests.

## Interaction with model-bounded PRA

Reuse the model-bounded abstractions already implemented:

- `model_max_context_tokens`;
- encoding chunking;
- routing chunking;
- overlap/marker modes;
- materialization budget;
- `#__head`;
- streaming rollover;
- CPU/GPU K/V residency.

Do not create a parallel RoPE-only chunking/cache system.

## Statistical discipline

Use identical seeds/configs for matched comparisons.

For major claims report mean, standard deviation, and preferably paired differences across seeds.

Do not select only favorable seeds.

Store raw per-example/per-seed results.

## Paper 1.5 deliverable

Create a new paper directory following repository conventions, e.g.:

```text
docs/papers/paper1_5_rope_retrieved_kv/
```

Produce:

- `paper.tex`;
- `paper.pdf`;
- bibliography;
- generated tables/figures;
- shared JSON/CSV results;
- README/experiment reproduction notes if repository conventions support them.

### Suggested paper structure

1. Abstract
2. Introduction
3. Retrieved Native-KV Memory and the Positional Problem
4. Absolute Positioning and RoPE
5. PRA as a Controlled Experimental Framework
6. Separating Positional and Contextual Fragmentation
7. Experimental Setup
8. Translation/Relocation Experiments
9. Encoding Fragmentation and Overlap
10. Pre-RoPE vs Post-RoPE Memory
11. Beyond-Native-Context Distance Experiments
12. Distance Compression and Ordering
13. `#__head` and Streaming Memory
14. Systems Cost
15. Discussion
16. Relation to Long-Context/KV-Memory Work
17. Limitations
18. Conclusion

## Minimum bar for independent publication

Do not call the work complete merely because “RoPE works with PRA.”

Aim to establish at least three strong general findings among:

1. quantitative block-translation invariance versus absolute positioning;
2. decomposition of fragmentation into positional versus contextual components;
3. overlap/historical-context recovery curves;
4. pre-RoPE versus post-RoPE portability;
5. reduced need for exact metric distance in remote memory;
6. importance of ordering versus exact long-range distance.

If only implementation compatibility is demonstrated, keep the work as Paper-1 supplementary material rather than forcing a standalone publication.

## Related-work pass

Perform a serious literature review before final claims.

Cover at minimum:

- original RoPE;
- positional extrapolation/interpolation;
- RoPE scaling approaches relevant to long context;
- ALiBi;
- Transformer-XL relative positions;
- recurrent/segment-level transformer memory;
- retrieved/external KV-cache systems;
- KV compression/offload/paging;
- long-context attention methods relevant to positional treatment.

Search for prior work specifically on relocating/reusing cached K/V under RoPE and deferred/pre-RoPE key storage.

Do not claim novelty until this search is complete.

## Figures to prioritize

1. conceptual diagram: raw K -> RoPE -> stored/retrieved memory;
2. pure translation invariance plot;
3. fragmentation: absolute vs RoPE;
4. quality vs encoding overlap;
5. pre-RoPE vs post-RoPE relocation;
6. quality vs logical/native context ratio;
7. quality vs distance precision;
8. order-vs-distance ablation.

## Reproducibility artifacts

Suggested result files:

```text
rope_translation.*
rope_fragmentation.*
rope_overlap.*
rope_pre_post_k.*
rope_distance_scaling.*
rope_distance_policy.*
rope_order_ablation.*
rope_head_scaling.*
rope_streaming.*
```

Use JSON/CSV plus generated figures. Include git SHA, device, framework versions, model config, training budget, dataset, seeds, and exact positional policy.

## Tests

Add unit/integration tests for:

- RoPE mathematical translation invariance;
- q/k raw vs rotated shapes;
- pre/post-RoPE storage;
- deferred rotation;
- logical/local position metadata;
- clipping/bucketing policies;
- causal legality;
- model context bounds;
- `#__head`;
- streaming rollover;
- CPU/GPU residency parity;
- absolute-model backward compatibility.

Keep all existing PRA tests passing.

## Non-goals

Do not spend this cycle on:

- full HF integration;
- many 7B+ models;
- custom fused CUDA kernels;
- ANN routing;
- alternative materialization research;
- every known RoPE scaling variant;
- distributed memory;
- production serving.

Paper 2 will validate on real pretrained/open models after this work establishes the positional semantics.

## Final workflow

1. pull current `main`;
2. record baseline SHA;
3. run all tests;
4. implement matched positional models and instrumentation;
5. run mathematical sanity checks;
6. run tiny experiments first;
7. inspect results before large sweeps;
8. run small/multi-seed confirmation;
9. perform related-work search;
10. write Paper 1.5 from actual results;
11. rebuild/inspect PDF;
12. update PRA roadmap with findings relevant to Paper 2;
13. run full tests;
14. commit code/results/paper/PDF;
15. push;
16. report commit SHA, main findings, negative findings, and remaining uncertainties.

## Interpretation guardrails

Do not claim RoPE eliminates chunking degradation.

The expected distinction is:

> RoPE can reduce positional-coordinate fragmentation, while bounded independent encoding can still lose cross-block contextualization.

Do not claim exact long-distance position is irrelevant unless controlled ablations demonstrate it.

Do not claim pre-RoPE K is superior unless experiments show a meaningful semantic or systems advantage.

Do not call answer-code probes unrestricted QA.

Do not claim arbitrary context is natively processed by the base model.

## Strategic outcome

The desired progression is:

```text
Paper 0
PRA architecture / position

Paper 1
PRA controlled mechanism + systems evidence

Paper 1.5
positional semantics of retrieved native-KV memory
absolute vs RoPE
fragmentation vs contextualization
portable K representation
distance/order requirements beyond native context

Paper 2
pretrained HF/open-model integration
```

Paper 1.5 should answer the positional questions in a controlled environment so Paper 2 can focus on whether the resulting PRA design transfers to real SOTA/open pretrained models rather than discovering fundamental RoPE semantics inside a large integration effort.
