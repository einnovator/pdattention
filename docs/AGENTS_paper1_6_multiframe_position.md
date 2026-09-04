# AGENTS.md â€” Paper 1.6
## Multi-Frame Positional Geometry for Retrieval-Native Attention

### Purpose
Paper 1.6 follows Paper 1.5. Paper 1.5 establishes correct source-relative transport, deferred RoPE, source/query offsets, and rebinding. Paper 1.6 asks whether independently retrieved resources must share one global positional sequence once external K/V can be rebound correctly.

The result may inform Paper 4.5 positional-policy/profile configuration, but do not change runtime defaults before evidence.

## Core decomposition
Treat every native PRA object as three independent properties:

```text
logical identity
physical residency
positional frame
```

Invariant: `physical location != logical identity != positional coordinate`.

## Main mechanism
For resource r preserve internal geometry while choosing a resource-specific offset:

`p'_(r,i) = b_r + p_(r,i)`.

Different resources may overlap in effective RoPE coordinates while remaining distinct K/V objects.

## Research questions
- Does multi-frame positioning improve mean quality?
- Does it reduce prediction variance under irrelevant resource permutations?
- Does it reduce source-distance/lost-in-the-middle sensitivity?
- Does it reduce distractor sensitivity?
- Does fully overlapping query-adjacent placement create an OOD penalty?
- Are effects model-family/size/workload dependent?
- Should positional policy become a calibrated Paper-4.5 profile dimension?

## Required positional policies
Hold retrieval, selected resources/tokens, materialization, consumer layers, query, and generation fixed. Change only positional policy.

1. `GLOBAL_SOURCE` â€” correct globally coherent/source frame; Paper-1.5 correctness reference.
2. `GLOBAL_PACKED` â€” selected resources packed sequentially into one virtual sequence.
3. `RESOURCE_ADJACENT` â€” each resource independently ends next to q; positional ranges may overlap.
4. `RANK_DISTANCE` â€” offset derived from retrieval rank.
5. `SCORE_DISTANCE` â€” offset derived from a predeclared normalized retrieval-score mapping.
6. `NON_OVERLAPPING_NEAR_BANDS` â€” nearby bounded bands preserving monotonic ordering.
7. `RANDOM_DISTANCE` â€” randomized offsets under a matched envelope as negative control.

Do not add graph/task/type-specific distance policies until these answer the main question.

## Correctness invariants
Every policy uses identical model weights, selected resource contents/order internally, selected-token count, detail/consumer layers, query/local context, attention normalization, retrieval scores, and generation settings.

## RoPE implementation
Reuse Paper-1.5 deferred/pre-RoPE machinery. Prefer position-independent/pre-RoPE K followed by request-specific rebinding. If stored K is post-RoPE, undo/reapply only through validated Paper-1.5 algebra.

Required tests:
- ordinary prefix == `GLOBAL_SOURCE`;
- pre/post rebinding equivalence;
- resource internal distance preservation;
- overlapping frames remain distinct logical/tensor objects;
- cached decode positions remain correct.

## Local query geometry
Keep live sequential trajectory conventional:

`p_q0 = p_source + T_local`.

Attention may expose:

`T_attn = T_selected + T_local + T_q`.

External resource positions are not scheduler sequence lengths.

## Multi-frame attention
At a consumer layer concatenate independently rebound resource K/V with local/query K/V and use one correct attention softmax. The novelty is the positional transform per resource, not separate normalizations.

## Primary permutation experiment
For a fixed evidence set generate multiple irrelevant resource permutations. Measure:
- task accuracy;
- NLL;
- target probability;
- answer flip rate;
- target-probability variance;
- pairwise JS divergence, with KL secondary.

Primary consistency metric: mean pairwise JS divergence across permutations. Strongest positive result: same/better quality plus lower permutation variance.

## Source-distance experiment
Hold evidence fixed while varying its original/global source distance from q. Measure accuracy/NLL, target probability, evidence attention, and answer flips.

## Distractor experiment
Sweep irrelevant selected resources: `0, 2, 4, 8, 16, 32`, with matched token budgets where needed. Test whether query-adjacent frames make distractors too salient.

## Multi-resource experiment
Sweep independently relevant resources: `1, 2, 4, 8`. Include redundant, complementary multi-hop, and conflicting evidence.

## Workloads
- W1 controlled synthetic facts â€” causal isolation.
- W2 HotpotQA / 2Wiki / MuSiQue â€” multi-hop composition.
- W3 QASPER â€” long-document/source-distance effects.
- W4 Paper-7 typed records â€” independent tool/API/DB/log records.
- Optional W5 Paper-8 task records only if W1-W4 show meaningful effects.

## Models
Start with Qwen3-0.6B and Llama 3.2-1B. Add Gemma only under validated topology and retain `PARTIAL_TOPOLOGY`. If compute permits add one larger model per family. Do not infer family-wide behavior from one size.

## Layer-profile interaction
Primary experiments use `REFERENCE_CORRECTNESS`. Then test a limited factorial subset with calibrated `QUALITY_MAX_CANDIDATE`, `BALANCED`, and `ECONOMY`.

Question: can improved positional geometry permit fewer consumer/detail layers at equal quality?

## Metrics
Primary: accuracy/EM/task metric, NLL, target-token probability, permutation JS/KL, answer flip rate, calibration/Brier where applicable, distractor sensitivity, source-distance slope.

Secondary: evidence attention mass, attention entropy, resource-level attention, logit margin.

Systems: rebinding latency, extra metadata/memory, pre-RoPE vs post-RoPE storage cost.

## Statistical design
Use paired comparisons because every policy sees identical evidence. Report bootstrap CIs and paired effects. Keep per-model/per-workload results visible; do not average away sign reversals.

## Falsification
The multi-frame quality hypothesis is unsupported if correct global geometry matches or beats resource-relative policies on quality and robustness.

Report explicitly if overlapping frames hurt quality, distractors become over-salient, consistency improves but accuracy falls, gains are synthetic-only, policies fail cross-family transfer, or global topology remains best.

A negative result still establishes Paper-1.5 rebinding as a correctness/systems abstraction rather than a semantic-quality mechanism.

## Incorrect-geometry control
Include a restart/collision positional bug control similar to the Paper-8 failure. Use it only to establish the correctness floor.

## Engine independence
Run primary experiments on the simplest validated HF/native path. Reproduce a small subset on one engine-native path, preferably MLX initially. Do not require all engines.

## Paper 4.5 decision rule
If no robust benefit:
- keep reference/correct geometry;
- expose multi-frame only as experimental.

If robust model/workload-specific benefit:
- add `position_policy` to semantic PRA profiles;
- calibrate it with layer/materialization policy;
- add evidence/provenance to the profile registry.

Possible runtime YAML:

```yaml
position:
  policy: resource_adjacent
  parameters: {}
  evidence_tier: BENCHMARK
```

Change defaults only after held-out cross-model validation.

## Runtime requirements if promoted
Every request/native PRA block must explicitly preserve source positional frame, resource-relative mapping, query-relative rebinding, and policy version. Physical residency may change without changing logical identity/source frame; request-specific rebinding may change without duplicating immutable resource identity.

## Feedback to engine papers
If multi-frame value is established, patch vLLM/SGLang/MLX integrations to preserve independently:

```text
logical identity
physical placement
request-specific positional mapping
```

Engines must not infer PRA position from ordinary prefix length.

## Artifacts
Create:

```text
docs/papers/paper1_6_multiframe_position/
  AGENTS.md
  paper.tex
  README.md
experiments/paper1_6/
docs/papers/shared/results/paper1_6/
```

Required result files:
`policy_manifest.json`, `permutation_results.csv`, `distance_results.csv`, `distractor_results.csv`, `multi_resource_results.csv`, `layer_interaction_results.csv`, `cross_model_results.csv`, `rebinding_cost.json`.

## Primary figures
1. global vs multi-frame geometry;
2. quality by positional policy;
3. permutation divergence/answer flips;
4. quality vs source distance;
5. quality vs distractor count;
6. quality vs number of resources;
7. model-family comparison;
8. optional layer-profile Ã— position-policy Pareto frontier.

## Boundary with Paper 1.5
Paper 1.5 = correct transport/rebinding machinery. Paper 1.6 = empirical choice of positional topology. Patch 1.5 only with a short forward reference.

## Related work
Cover RoPE/relative positions, position interpolation/scaling, long-context positional effects/lost-in-the-middle, retrieval/external-memory attention, permutation sensitivity, and relational/structured positional encodings where directly relevant. Do not claim prior work studies independent PRA resource-relative native-K/V frames unless it actually does.

## Claim hierarchy
Strong positive: resource-relative frames preserve internal geometry while reducing arbitrary serialization-order/source-distance sensitivity.

Conservative: correct external-memory geometry is essential, and positional topology materially affects robustness even with retrieval and selected K/V fixed.

Negative: once geometry is correct, pretrained models prefer conventional global topology; multi-frame rebinding is primarily a systems abstraction.

## Stop gate
Ready for editorial consolidation when:
- all seven policies are correct;
- prefix/`GLOBAL_SOURCE` parity passes;
- permutation, distance, and distractor curves exist;
- Qwen + Llama tested;
- synthetic + at least one natural workload tested;
- layer-profile interaction measured on one model;
- one engine-native reproduction exists if practical;
- correctness/quality/consistency claims are separated;
- Paper-4.5 recommendation is explicit;
- tests/artifacts/plots/PDF/visual QA pass.

