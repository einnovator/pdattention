# AGENTS-pra-experiments.md

## Mission

Extend the existing PRAttention repository with a reproducible, phased experiment suite for evaluating standard SelfAttention, full PRAttention, and hybrid SelfAttention→PRAttention architectures on WikiText-2.

The current six configurations are useful as smoke tests but are not yet sufficient to support architectural conclusions.

The experiment suite must answer:

1. Can all three architectures learn ordinary WikiText-2 language modelling?
2. Does PRAttention train stably when no explicit references are present?
3. Does pretraining on ordinary text improve later learning on reference-structured text?
4. Does the hybrid architecture benefit from learning local syntax in early SelfAttention layers and reference-oriented structure in later PRAttention layers?
5. Does PRAttention use actual referenced content, or only exploit formatting, position, token frequency, or extra capacity?
6. Does PRAttention help more as reference distance increases or local context becomes insufficient?
7. Are gains reproducible across seeds, training budgets, and model sizes?

Do not treat one-epoch runs as evidence of performance. Preserve them as smoke tests only.

---

# 1. Existing architecture configurations

The current tiny model family is:

```yaml
td_sa_tiny:
  model:
    d_model: 256
    n_heads: 4
    n_layers: 4
    n_vanilla_layers: 4
    n_mixed_layers: 0
    max_seq_len: 256
    dropout: 0.1
    model_variant: td_sa
  pra:
    pra_layer_ids: []

td_pra_tiny:
  model:
    d_model: 256
    n_heads: 4
    n_layers: 4
    n_vanilla_layers: 0
    n_mixed_layers: 0
    max_seq_len: 256
    dropout: 0.1
    model_variant: td_pra
  pra:
    pra_layer_ids: [0, 1, 2, 3]

tdx_pra_tiny:
  model:
    d_model: 256
    n_heads: 4
    n_layers: 4
    n_vanilla_layers: 2
    n_mixed_layers: 0
    max_seq_len: 256
    dropout: 0.1
    model_variant: tdx_pra
  pra:
    pra_layer_ids: [2, 3]
```

Interpretation:

- `td_sa_tiny`: four standard causal SelfAttention layers.
- `td_pra_tiny`: four PRAttention layers.
- `tdx_pra_tiny`: two lower SelfAttention layers followed by two upper PRAttention layers.

The six current smoke-test combinations are:

| ID | Dataset | Architecture | Training |
|---|---|---|---|
| `smoke_plain_sa` | WikiText-2 | 4 SelfAttention layers | 1 epoch |
| `smoke_plain_pra` | WikiText-2 | 4 PRAttention layers | 1 epoch |
| `smoke_plain_hybrid` | WikiText-2 | 2 SelfAttention + 2 PRAttention | 1 epoch |
| `smoke_refs_sa` | WikiText-2 with references | 4 SelfAttention layers | 1 epoch |
| `smoke_refs_pra` | WikiText-2 with references | 4 PRAttention layers | 1 epoch |
| `smoke_refs_hybrid` | WikiText-2 with references | 2 SelfAttention + 2 PRAttention | 1 epoch |

Preserve these exact configurations as a fast integration-test matrix.

---

# 2. Reference dataset definition

The current reference dataset is derived from WikiText-2.

For each WikiText entry:

1. Split the entry into approximately 1–5 document parts.
2. Present earlier parts as addressable document or section fragments.
3. Use a tail fragment as the prediction target.
4. The tail is not necessarily a literal question-answer example.
5. The completion may depend indirectly on one or more earlier parts.
6. The relevant information may be semantic, lexical, entity-based, structural, or discourse-based.
7. The target remains causal language modelling over natural text.

This must be described as:

> reference-conditioned continuation or indirect completion

Do not describe it as a conventional QA dataset unless an explicit question and answer are actually generated.

The reference generator must preserve natural WikiText language and produce metadata identifying:

- source WikiText entry;
- part boundaries;
- part identifiers;
- candidate reference parts;
- intended relevant parts, if known;
- tail or continuation span;
- token distance from relevant parts to target;
- number of parts;
- whether the target is answerable from local context alone;
- whether referenced content was copied, paraphrased, or only indirectly relevant.

---

# 3. Experimental rules

## 3.1 Compare equal training budgets

Use processed training tokens as the primary budget.

Do not compare models only because they completed the same number of epochs if preprocessing changes the number of tokens per sample.

Record:

```text
processed_tokens
optimizer_steps
sequences_seen
wall_clock_time
```

## 3.2 Keep non-architectural variables fixed

For controlled comparisons, keep fixed:

- tokenizer;
- vocabulary;
- training split;
- validation split;
- test split;
- random seed;
- context length;
- batch size;
- effective tokens per optimizer step;
- optimizer;
- learning-rate schedule;
- warmup;
- weight decay;
- dropout;
- initialization;
- gradient clipping;
- training-token budget;
- evaluation frequency.

## 3.3 Match parameter counts where possible

Report:

- total parameters;
- trainable parameters;
- embedding parameters;
- attention parameters;
- MLP parameters;
- reference-specific parameters.

If PRAttention adds parameters, create at least one matched-capacity SelfAttention control by adjusting width or FFN size.

Report both:

1. same-width comparison;
2. approximately parameter-matched comparison.

## 3.4 Use multiple seeds for any claimed result

Smoke tests may use one seed.

Pilot experiments should use at least two seeds.

Final tiny-model comparisons should use at least three seeds, preferably five when inexpensive.

Recommended default:

```yaml
seeds: [1, 7, 21]
```

## 3.5 Save complete provenance

Every run must save:

- run ID;
- resolved configuration;
- git commit;
- dirty-working-tree flag;
- device and backend;
- PyTorch version;
- Python version;
- tokenizer hash;
- dataset generator version;
- dataset hash;
- model parameter counts;
- seed;
- trainable-layer list;
- optimizer configuration;
- training tokens;
- validation metrics;
- test metrics;
- elapsed time;
- peak memory when available;
- best-checkpoint path.

---

# 4. Required metrics

## 4.1 Language-modelling metrics

Always report:

- training cross-entropy;
- validation cross-entropy;
- test cross-entropy;
- perplexity;
- token accuracy;
- tokens per second;
- optimizer steps per second;
- peak memory;
- wall-clock time.

## 4.2 Reference-specific metrics

Where reference metadata is available, report:

- correct-reference top-1 accuracy;
- correct-reference top-k accuracy;
- mean reciprocal rank;
- normalized discounted cumulative gain when multiple references are relevant;
- reference attention mass on relevant parts;
- reference attention entropy;
- percentage of samples where the reference branch is used;
- average gate value per layer;
- LM loss with references enabled;
- LM loss with references disabled;
- LM loss with references shuffled;
- LM loss with references replaced by irrelevant parts;
- LM loss grouped by reference distance;
- LM loss grouped by number of candidate parts;
- LM loss grouped by local-context sufficiency.

## 4.3 Representation diagnostics

Implement optional probes for:

- token embedding nearest neighbours;
- hidden-state cosine similarity;
- layer-wise representation norms;
- layer-wise activation variance;
- gradient norms;
- attention entropy;
- gate distributions;
- reference-score distributions.

These are diagnostics, not primary success metrics.

---

# 5. Phase 0 — Correctness and smoke tests

## Objective

Verify that the six existing combinations execute correctly.

## Runs

```text
smoke_plain_sa
smoke_plain_pra
smoke_plain_hybrid
smoke_refs_sa
smoke_refs_pra
smoke_refs_hybrid
```

## Required checks

Each run must verify:

- dataset construction completes;
- tokenization completes;
- batch shapes are correct;
- causal masks are correct;
- reference masks are correct;
- no target-token leakage exists;
- model forward pass completes;
- loss is finite;
- backward pass completes;
- optimizer step changes trainable parameters;
- frozen parameters do not change;
- checkpoint save/load reproduces logits within tolerance;
- validation pass completes;
- no NaN or Inf values occur;
- reference tensors are consumed only by PRAttention-enabled layers.

## Tiny-overfit test

Create a deterministic 8–32 sample subset.

Each architecture must be able to overfit it to very low training loss.

Failure to overfit should block longer runs.

## Leakage tests

Add tests proving that:

- target tokens are not present in accessible reference values unless intentionally allowed;
- reference labels are derived only from source parts;
- validation/test entries do not share transformed instances with training;
- splitting a WikiText entry does not place one part in training and another in validation/test;
- shuffled references alter reference metadata but not the target sequence.

---

# 6. Phase 1 — Ordinary language-model pretraining

## Objective

Learn meaningful token embeddings, local syntax, common phrases, and basic semantic structure before evaluating reference resolution.

Use unmodified WikiText-2.

## Architectures

Train:

```text
plain_sa_pretrain
plain_pra_pretrain
plain_hybrid_pretrain
```

Based on:

- `td_sa_tiny`;
- `td_pra_tiny`;
- `tdx_pra_tiny`.

## Main questions

1. Does each architecture converge?
2. How does validation perplexity compare?
3. Is full PRAttention intrinsically harder to optimize?
4. Does the hybrid model retain the language-modelling quality of the SelfAttention baseline?
5. Does PRAttention collapse or behave like local attention when references are absent?

## Training schedule

Do not use one epoch as the main budget.

Implement token-budget based schedules.

Suggested initial pilot:

```yaml
training:
  max_tokens: 50_000_000
  eval_every_tokens: 500_000
  early_stopping_patience_evals: 10
  save_best: true
  monitor: validation_loss
```

Because WikiText-2 is small, the exact budget may be reduced after observing convergence.

Support equivalent epoch-based configuration, but internally record token counts.

## Checkpoints

Save:

- initialization checkpoint;
- periodic checkpoints;
- best-validation checkpoint;
- final checkpoint.

The best checkpoint from each architecture becomes an input to later phases.

## Phase 1 success criteria

At minimum:

- stable decreasing training loss;
- finite validation loss;
- ability to overfit the tiny subset;
- meaningful improvement over unigram/random baseline;
- no severe train/validation divergence during the selected training budget;
- repeatable results across at least two seeds.

Do not require all architectures to have equal perplexity before proceeding.

---

# 7. Phase 2 — Train the new reference path while preserving language ability

## Objective

Introduce reference-conditioned continuation after ordinary language structure has been learned.

Phase 2 is deliberately conservative: train the new reference-specific machinery while freezing most pretrained language-model parameters.

This phase should reveal whether reference selection can be learned from meaningful hidden representations without destabilizing the pretrained model.

## Initialization

For each architecture:

### SelfAttention control

Initialize from:

```text
plain_sa_pretrain
```

Continue on reference-formatted WikiText-2.

There is no new PRAttention path to train. This run controls for:

- changed data formatting;
- special tokens;
- document boundaries;
- continuation structure;
- extra context.

Run ID:

```text
refs_sa_control_finetune
```

### Full PRAttention

Initialize from:

```text
plain_pra_pretrain
```

Train reference-specific projections, routing, gates, reference embeddings, and optional reference-selection heads.

Run ID:

```text
refs_pra_refpath
```

### Hybrid

Initialize from:

```text
plain_hybrid_pretrain
```

Train reference-specific parameters in layers 2 and 3 while preserving lower SelfAttention layers.

Run ID:

```text
refs_hybrid_refpath
```

## Frozen parameters

During the initial Phase 2 substage, freeze:

- token embeddings;
- positional embeddings;
- lower SelfAttention layers;
- standard local attention projections;
- MLP/FFN blocks;
- final language-model head;
- any pretrained parameters not specific to reference routing.

For the full PRAttention architecture, distinguish between:

- pretrained local or ordinary language-modelling components;
- newly activated reference-specific components.

Do not freeze the whole layer if local and reference paths share a module. Freeze individual parameter groups.

## Trainable parameters

Train:

- reference query projections;
- reference key projections;
- reference value projections;
- reference output projections;
- reference router or scorer;
- per-layer reference gates;
- reference-type embeddings;
- document-part embeddings;
- reference-position embeddings;
- optional reference LayerNorms;
- auxiliary reference-selection head;
- any parameters that did not exist or were inactive during Phase 1.

Optionally allow immediately adjacent normalization parameters to train if strict freezing causes instability.

## Recommended gate initialization

If using sigmoid gates:

```python
gate = sigmoid(gate_logit)
```

Initialize:

```python
gate_logit = -4.0
```

This gives a small initial reference contribution.

Do not initialize a random reference branch with full residual strength.

## Objectives

Use:

```text
L_total =
    L_language_model
  + lambda_ref * L_reference_selection
  + lambda_gate * L_gate_regularization
```

Where available:

- `L_language_model` is normal causal next-token loss;
- `L_reference_selection` uses known relevant-part metadata;
- `L_gate_regularization` should prevent immediate saturation, not force usage indefinitely.

Support turning auxiliary losses off.

## Differential learning rates

Recommended initial ranges:

```yaml
optimizer_groups:
  reference_parameters:
    lr: 3e-4
  optional_reference_norms:
    lr: 1e-4
  frozen_base:
    lr: 0.0
```

Make these configurable.

## Phase 2 evaluation

Evaluate both language preservation and reference learning:

- plain WikiText-2 validation loss before and after Phase 2;
- reference-formatted validation loss;
- correct-reference top-k accuracy;
- loss with valid references;
- loss with shuffled references;
- loss with reference path disabled;
- gate activation by layer;
- relevant-reference attention mass.

## Phase 2 transition criteria

Proceed to Phase 3 when at least some of the following hold:

- reference-selection accuracy exceeds chance;
- valid references reduce LM loss relative to shuffled references;
- gate values are nonzero but not saturated everywhere;
- plain-text validation loss has not catastrophically degraded;
- gradients reach reference-specific parameters;
- reference use is stronger on samples labelled as non-local.

Do not require perfect reference selection.

---

# 8. Phase 3 — Joint end-to-end fine-tuning

## Objective

Allow the language model and reference mechanism to adapt to each other.

Unlike Phase 2, all relevant model parameters may now update.

## Initialization

Initialize from the best Phase 2 checkpoints:

```text
refs_sa_control_finetune
refs_pra_refpath
refs_hybrid_refpath
```

## Trainable parameters

Unfreeze:

- token embeddings;
- positional embeddings, if learned;
- all attention layers;
- all PRAttention layers;
- all MLP blocks;
- LayerNorms;
- language-model head;
- reference projections;
- gates;
- router/scorer;
- auxiliary heads.

## Differential learning rates

Use a smaller learning rate for pretrained base parameters and a larger one for reference-specific parameters.

Example:

```yaml
optimizer_groups:
  base_transformer:
    lr: 1e-5
  reference_parameters:
    lr: 1e-4
  embeddings:
    lr: 5e-6
  lm_head:
    lr: 1e-5
```

Support tied embeddings correctly.

## Data mixture

Do not train exclusively on reference-formatted examples.

Use a configurable mixture, for example:

```yaml
data_mixture:
  plain_wikitext2: 0.6
  reference_wikitext2: 0.4
```

Also support a curriculum:

```text
start: 80% plain / 20% reference
middle: 60% plain / 40% reference
late: 40% plain / 60% reference
```

The schedule should be based on processed tokens, not epochs.

## Main comparison

Compare:

```text
SelfAttention control
Full PRAttention
2 SelfAttention + 2 PRAttention hybrid
```

Under the same:

- tokenizer;
- data mixture;
- token budget;
- context length;
- optimizer policy;
- seeds.

## Primary hypothesis

The hybrid architecture may be especially effective because:

- lower SelfAttention layers learn local lexical and syntactic structure;
- upper PRAttention layers operate on more meaningful and abstract hidden representations;
- reference resolution is delayed until the model has constructed useful internal features.

Do not assume this is true. Treat it as a testable hypothesis.

---

# 9. Phase 4 — Reference-use ablations

## Objective

Determine whether any gain is caused by genuine use of reference content.

Every promising PRAttention or hybrid checkpoint must be evaluated under the following conditions.

## Ablation A — Valid references

Normal reference inputs.

## Ablation B — Reference path disabled

Set reference contribution to zero.

## Ablation C — Shuffled references

Keep the same number and length of references but shuffle them across samples.

## Ablation D — Irrelevant references

Replace relevant parts with parts from unrelated WikiText entries.

## Ablation E — Empty references

Preserve reference markers and metadata structure but remove reference content.

## Ablation F — Random reference IDs

Randomize reference identifiers while keeping content fixed.

## Ablation G — Position-only control

Preserve reference positions but mask reference token content.

## Ablation H — Content-only control

Preserve content but remove explicit reference IDs or structural markers.

## Ablation I — Oracle reference

Provide only the known relevant reference part.

This measures the upper bound when retrieval is perfect.

## Ablation J — All parts visible locally

Concatenate all parts directly into the local context when they fit.

This compares external/reference access with ordinary long-context SelfAttention.

## Required interpretation

Evidence of genuine reference use requires:

- valid references outperform shuffled or irrelevant references;
- disabling the reference path worsens relevant examples;
- gains increase on examples that cannot be solved from the local tail alone;
- attention or routing assigns higher scores to relevant parts than controls.

Formatting-only improvements are not sufficient evidence.

---

# 10. Phase 5 — Difficulty and distance curriculum

## Objective

Measure when PRAttention becomes useful.

Partition reference examples by:

- number of document parts;
- number of candidate references;
- token distance;
- whether relevant content fits in local context;
- lexical overlap;
- entity overlap;
- direct versus indirect dependency;
- single-reference versus multi-reference dependency;
- continuation length.

Suggested buckets:

```text
distance:
  near: 0–64 tokens
  medium: 65–256 tokens
  far: 257–1024 tokens
  very_far: >1024 tokens

parts:
  1
  2
  3
  4
  5

dependency:
  local_sufficient
  reference_helpful
  reference_required
```

The exact bucket boundaries must be configurable and based on tokenized distance.

## Expected analysis

Plot or tabulate:

- loss by distance;
- perplexity by distance;
- reference-selection accuracy by distance;
- gain over SelfAttention by distance;
- gain over disabled-reference ablation;
- gate usage by distance;
- relevant-reference attention mass by distance.

A central expected pattern is:

> PRAttention should provide little advantage when the target is locally predictable, but increasing advantage when relevant information is outside the effective local context.

Treat this as a hypothesis, not an assumption.

---

# 11. Phase 6 — Training-order ablations

## Objective

Test the claim that language pretraining should precede reference learning.

Run at least:

### A. From scratch on reference data

```text
scratch_refs_sa
scratch_refs_pra
scratch_refs_hybrid
```

### B. Pretrain plain, then frozen-base reference training

```text
pretrain_then_refpath_sa
pretrain_then_refpath_pra
pretrain_then_refpath_hybrid
```

### C. Pretrain plain, then direct joint fine-tuning

Skip the frozen Phase 2 substage.

```text
pretrain_then_joint_sa
pretrain_then_joint_pra
pretrain_then_joint_hybrid
```

### D. Mixed curriculum from initialization

Start with mostly plain text and gradually increase reference examples.

```text
curriculum_mixed_sa
curriculum_mixed_pra
curriculum_mixed_hybrid
```

## Questions

- Does pretraining improve reference-selection speed?
- Does pretraining produce better final loss?
- Does the frozen Phase 2 substage prevent catastrophic forgetting?
- Is direct joint fine-tuning sufficient?
- Does the hybrid architecture benefit more from staged training than full PRAttention?
- Does full PRAttention need a longer ordinary-language pretraining period?

---

# 12. Phase 7 — Architecture-order ablations

## Objective

Test whether PRAttention works best in upper layers.

Add configurable layer placements while preserving four total layers:

```text
SA-SA-SA-SA
PRA-PRA-PRA-PRA
SA-SA-PRA-PRA
SA-PRA-SA-PRA
PRA-PRA-SA-SA
SA-PRA-PRA-PRA
PRA-SA-SA-SA
```

Required primary comparison:

```text
4 SA
4 PRA
2 SA + 2 PRA
```

Secondary placements should run only after the primary experiment harness is stable.

## Hypotheses

- Early SelfAttention may be better for local lexical and syntactic processing.
- Later PRAttention may be better for semantically meaningful reference selection.
- Full PRAttention may learn equivalent behavior but require more data or training.
- PRAttention in early layers may attend using weak, immature representations.

Log layer-wise reference selection and gate values to test these hypotheses.

---

# 13. Phase 8 — Context-length controls

## Objective

Separate reference benefits from simple context-window effects.

Run selected architectures with:

```text
max_seq_len = 64
max_seq_len = 128
max_seq_len = 256
max_seq_len = 512
```

Subject to hardware limits.

Compare:

- SelfAttention with longer context;
- PRAttention with shorter local context plus references;
- hybrid with shorter local context plus references;
- equal-memory configurations;
- equal-compute configurations where possible.

Report:

- validation loss;
- tokens per second;
- peak memory;
- gain per unit compute;
- gain per unit memory.

---

# 14. Phase 9 — Scale-up configurations

Do not scale before the tiny experiments are stable and informative.

## Tiny

Existing:

```yaml
d_model: 256
n_heads: 4
n_layers: 4
d_ff: 1024
max_seq_len: 256
```

## Small

Add:

```yaml
d_model: 384
n_heads: 6
n_layers: 6
d_ff: 1536
max_seq_len: 256
```

Primary variants:

```text
6 SA
6 PRA
3 SA + 3 PRA
4 SA + 2 PRA
```

## Medium

Optional later:

```yaml
d_model: 512
n_heads: 8
n_layers: 8
d_ff: 2048
max_seq_len: 256
```

Primary variants:

```text
8 SA
8 PRA
4 SA + 4 PRA
6 SA + 2 PRA
```

Use the tiny suite to eliminate weak configurations before scale-up.

---

# 15. Dataset improvements

The current generator should be retained but upgraded.

## Required metadata

Each generated sample should include:

```json
{
  "source_entry_id": "...",
  "part_ids": ["p0", "p1", "p2"],
  "relevant_part_ids": ["p0"],
  "target_part_id": "p2",
  "num_parts": 3,
  "reference_distance_tokens": 418,
  "local_context_sufficient": false,
  "dependency_type": "indirect_continuation",
  "generation_version": "..."
}
```

## Required dataset variants

Generate:

1. `refs_natural`
   - current natural split/continuation logic.

2. `refs_oracle`
   - only known relevant parts provided.

3. `refs_distractors`
   - relevant parts mixed with unrelated distractors.

4. `refs_shuffled`
   - references reassigned across samples.

5. `refs_empty`
   - reference structure without content.

6. `refs_local_sufficient`
   - target can be predicted from nearby context.

7. `refs_reference_required`
   - local context is intentionally truncated so relevant earlier content is required.

8. `refs_multi_part`
   - completion depends on more than one earlier part where feasible.

## Avoid trivial shortcuts

The generator must prevent:

- correct reference always being first;
- relevant reference always being longest;
- relevant reference always being closest;
- IDs encoding the correct answer;
- target being copied verbatim into metadata;
- fixed number of references per class;
- train/test duplication;
- target-tail markers uniquely identifying the answer.

Randomize candidate order independently for each sample.

---

# 16. Configuration structure

Add experiment-level YAML files rather than hard-coding runs.

Suggested structure:

```text
configs/
  models/
    td_sa_tiny.yaml
    td_pra_tiny.yaml
    tdx_pra_tiny.yaml

  datasets/
    wikitext2_plain.yaml
    wikitext2_refs.yaml
    wikitext2_refs_shuffled.yaml
    wikitext2_refs_oracle.yaml

  phases/
    phase0_smoke.yaml
    phase1_pretrain.yaml
    phase2_refpath.yaml
    phase3_joint.yaml
    phase4_ablations.yaml
    phase5_distance.yaml
    phase6_training_order.yaml

  experiments/
    tiny_primary_matrix.yaml
```

Every run should resolve to one complete immutable configuration.

---

# 17. Recommended run naming

Use:

```text
{phase}_{dataset}_{architecture}_{seed}_{budget}
```

Examples:

```text
p1_plain_sa_s1_50m
p1_plain_pra_s1_50m
p1_plain_hybrid_s1_50m

p2_refs_sa_s1_10m
p2_refs_pra_s1_10m
p2_refs_hybrid_s1_10m

p3_mix_sa_s1_30m
p3_mix_pra_s1_30m
p3_mix_hybrid_s1_30m
```

Do not encode ambiguous terms such as `tiny_test2_final`.

---

# 18. Checkpoint transfer

Implement explicit checkpoint transfer utilities.

Required capabilities:

- load exact same architecture;
- load only matching parameter names;
- report missing keys;
- report unexpected keys;
- initialize new reference parameters separately;
- freeze parameter groups from configuration;
- verify frozen parameters are unchanged;
- save parent checkpoint ID in child-run metadata.

For each transferred checkpoint, print:

```text
loaded_parameters
new_parameters
frozen_parameters
trainable_parameters
```

Fail loudly on accidental partial loads unless explicitly allowed.

---

# 19. Optimizer and freezing implementation

Provide named parameter groups:

```text
embeddings
local_attention
reference_attention
mlp
normalization
lm_head
reference_router
reference_gates
reference_auxiliary_head
```

Configuration should support:

```yaml
parameter_groups:
  embeddings:
    trainable: false
    lr: 0.0

  local_attention:
    trainable: false
    lr: 0.0

  reference_attention:
    trainable: true
    lr: 3e-4

  reference_router:
    trainable: true
    lr: 3e-4

  reference_gates:
    trainable: true
    lr: 1e-4
```

Add a startup assertion that optimizer groups contain only parameters with `requires_grad=True`.

---

# 20. Reports and plots

Generate one machine-readable result file per run:

```text
results/{run_id}/metrics.json
results/{run_id}/history.jsonl
results/{run_id}/resolved_config.yaml
results/{run_id}/provenance.json
```

Generate comparison reports for each phase.

Required plots:

- train loss versus processed tokens;
- validation loss versus processed tokens;
- perplexity versus processed tokens;
- reference accuracy versus processed tokens;
- valid versus shuffled-reference loss;
- gate value by layer;
- relevant-reference attention mass by layer;
- loss versus reference distance;
- throughput versus architecture;
- memory versus architecture.

Report mean and standard deviation across seeds.

Do not smooth curves without preserving raw data.

---

# 21. Statistical reporting

For primary comparisons, report:

- mean;
- standard deviation;
- median;
- individual seed values;
- paired difference by seed where seeds and data order match.

Bootstrap confidence intervals are optional for the tiny phase.

Avoid claiming significance from a single run.

---

# 22. Minimum primary experiment matrix

After smoke tests, implement this minimum suite.

## Phase 1

| Dataset | SA | PRA | Hybrid |
|---|---:|---:|---:|
| Plain WikiText-2 | yes | yes | yes |

## Phase 2

| Dataset | SA | PRA | Hybrid |
|---|---:|---:|---:|
| References | control fine-tune | frozen-base ref-path training | frozen-base ref-path training |

## Phase 3

| Dataset mixture | SA | PRA | Hybrid |
|---|---:|---:|---:|
| Plain + references | yes | yes | yes |

## Phase 4

For PRA and hybrid:

- valid references;
- disabled reference path;
- shuffled references;
- irrelevant references;
- oracle references.

## Seeds

At least:

```text
1, 7, 21
```

This minimum matrix is the first milestone.

---

# 23. Initial recommended execution order

Run in this order:

1. Six existing one-epoch smoke tests.
2. Tiny-overfit tests for SA, PRA, and hybrid.
3. Phase 1 plain WikiText-2 pretraining for SA.
4. Phase 1 pretraining for hybrid.
5. Phase 1 pretraining for full PRA.
6. Phase 2 reference-path training for hybrid.
7. Phase 2 reference-path training for full PRA.
8. SelfAttention reference-format control.
9. Phase 3 joint training for all three.
10. Valid/shuffled/disabled/oracle ablations.
11. Repeat promising settings across three seeds.
12. Add distance and difficulty analysis.
13. Scale only the strongest configurations.

The ordering intentionally prioritizes the hybrid model before full PRA after establishing the SelfAttention baseline, because upper-layer PRAttention is the most direct test of the hypothesis that meaningful representations should precede reference resolution.

---

# 24. Decision gates

## Gate A — Infrastructure ready

Proceed only if:

- all smoke tests pass;
- tiny-overfit tests pass;
- no leakage is detected;
- checkpoints round-trip correctly.

## Gate B — Language modelling works

Proceed only if:

- SelfAttention learns WikiText-2;
- hybrid and PRA runs are numerically stable;
- baseline validation curves are reproducible.

## Gate C — Reference mechanism learns

Proceed only if:

- valid references outperform at least one corrupted-reference control;
- reference-selection metrics exceed chance or LM loss demonstrates reference use;
- no catastrophic forgetting occurs.

## Gate D — Architectural claim is plausible

Scale only if:

- PRA or hybrid improves reference-required samples;
- gains survive shuffled and disabled-reference ablations;
- results repeat across seeds;
- gains are not solely due to more parameters.

---

# 25. Coding standards

Codex must:

- preserve existing public APIs where practical;
- add type hints;
- add docstrings for new public functions;
- use deterministic seeds;
- isolate dataset generation from training;
- avoid hidden global configuration;
- validate configurations before training;
- make all freezing decisions explicit;
- add unit tests for masks, references, checkpoint transfer, and dataset splitting;
- avoid introducing Hugging Face Transformers unless already required;
- keep tokenizer and dataset dependencies modular;
- support CPU, CUDA, and MPS when possible;
- fail clearly when an unsupported backend operation is encountered.

---

# 26. Tests to add

At minimum:

```text
test_plain_batch_shapes
test_reference_batch_shapes
test_causal_mask_no_future_access
test_reference_mask_respects_candidates
test_no_target_leakage
test_dataset_split_grouped_by_source_entry
test_checkpoint_roundtrip
test_checkpoint_transfer_reports_new_parameters
test_frozen_parameters_unchanged
test_reference_parameters_receive_gradients
test_disabled_reference_matches_expected_path
test_shuffled_reference_changes_reference_assignment
test_tiny_subset_overfit_sa
test_tiny_subset_overfit_pra
test_tiny_subset_overfit_hybrid
```

---

# 27. Deliverables

Codex should produce:

1. The six preserved smoke-test configurations.
2. Dataset-generation metadata and versioning.
3. Phase-specific experiment configurations.
4. Checkpoint-transfer utilities.
5. Parameter-freezing and optimizer-group utilities.
6. Reference corruption/ablation datasets.
7. Metric logging.
8. Comparison reports.
9. Plots.
10. Unit and integration tests.
11. A concise `EXPERIMENTS.md` describing how to launch each phase.
12. A generated experiment manifest listing all run IDs and dependencies.

---

# 28. First implementation milestone

The first implementation milestone is complete when the following command family can be expressed through repository-native CLI commands or equivalent:

```bash
# Smoke tests
run phase0 --model td_sa_tiny --dataset wikitext2_plain
run phase0 --model td_pra_tiny --dataset wikitext2_plain
run phase0 --model tdx_pra_tiny --dataset wikitext2_plain
run phase0 --model td_sa_tiny --dataset wikitext2_refs
run phase0 --model td_pra_tiny --dataset wikitext2_refs
run phase0 --model tdx_pra_tiny --dataset wikitext2_refs

# Plain pretraining
run phase1 --model td_sa_tiny --dataset wikitext2_plain
run phase1 --model td_pra_tiny --dataset wikitext2_plain
run phase1 --model tdx_pra_tiny --dataset wikitext2_plain

# Reference-path training
run phase2 --model td_pra_tiny --parent <plain-pra-checkpoint>
run phase2 --model tdx_pra_tiny --parent <plain-hybrid-checkpoint>

# SelfAttention formatting control
run phase2 --model td_sa_tiny --parent <plain-sa-checkpoint>

# Joint fine-tuning
run phase3 --model td_sa_tiny --parent <phase2-sa-checkpoint>
run phase3 --model td_pra_tiny --parent <phase2-pra-checkpoint>
run phase3 --model tdx_pra_tiny --parent <phase2-hybrid-checkpoint>

# Ablations
evaluate references-valid
evaluate references-disabled
evaluate references-shuffled
evaluate references-irrelevant
evaluate references-oracle
```

Adapt the exact commands to the repository's existing CLI rather than creating a conflicting second command system.

---

# 29. Core interpretation rule

The central scientific comparison is not merely:

```text
PRAttention loss < SelfAttention loss
```

The stronger required evidence is:

```text
PRAttention with valid references
    <
PRAttention with shuffled, disabled, or irrelevant references
```

especially on samples where:

```text
relevant information is outside the effective local context
```

Only then is there evidence that the architecture is resolving and using references rather than benefiting from incidental formatting or capacity.

---

# 30. Immediate Codex tasks

Start by:

1. Inspecting the existing model, dataset, trainer, configuration, and CLI structure.
2. Mapping the three existing model variants to named parameter groups.
3. Preserving the six one-epoch smoke tests.
4. Adding deterministic tiny-overfit tests.
5. Adding dataset metadata and leakage checks.
6. Implementing checkpoint transfer and freezing.
7. Creating Phase 1 plain WikiText-2 training configs.
8. Creating Phase 2 reference-path configs.
9. Creating Phase 3 mixed joint-training configs.
10. Adding valid, disabled, shuffled, irrelevant, and oracle evaluation modes.

Do not begin medium-scale experiments before the Phase 0 and tiny-overfit checks pass.
