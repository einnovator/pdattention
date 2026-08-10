# AGENTS.md — Native-KV PRA Reorientation

## Mission

Recenter PRA on the original hypothesis:

> **PRA is primarily an inference-time approximation to very-long / approximately unbounded native self-attention.** It externalizes the model's own layer-specific K/V state, indexes it cheaply using references/chunks/gists, retrieves a small relevant subset, and injects those original/native K/V back into ordinary self-attention.

Core PRA should **not require reference-conditioned training**. Train SelfAttention only to obtain a working language model; then freeze it and evaluate PRA as an inference mechanism.

The existing separate cross-attention + `mem_o_proj` implementation remains supported, but becomes an **optional adapted transport** and a later experiment rather than the definition or primary evidence for PRA.

Goals:
1. Add an explicit transport parameter: native K/V vs existing cross-attention.
2. Make `native_kv` the canonical/default PRA transport.
3. Evaluate PRA from a trained SA checkpoint with zero PRA training and ideally zero new trainable parameters.
4. Reorganize Paper 0 and Paper 1 around this hypothesis.
5. Move existing cross-attention training experiments later in Paper 1.
6. Prepare Paper 2: PRA wrapping/adaptation of pretrained Hugging Face LLMs.

---

## 1. PRA is not RAG

Core PRA:

```text
model-native layer K/V
 -> externalized storage
 -> references/chunks
 -> layer-specific gists/index
 -> progressive selection
 -> selected original/native K/V
 -> ordinary self-attention
```

RAG/search is orthogonal preprocessing/context acquisition, used when information was neither supplied nor referenced:

```text
missing information
 -> search/RAG/tool
 -> text/data
 -> model processing
 -> native layer K/V
 -> PRA may subsequently manage those K/V
```

Use this distinction:

> **RAG/search acquires context. PRA manages and selectively activates context already available to the inference process.**

A reference is an addressable region of inference context/state, not necessarily an externally retrieved document.

---

## 2. PRA as approximation to full self-attention

For layer `l`:

```text
Y_full = W_O^l Attn(Q_l, K_full^l, V_full^l)
```

PRA seeks a small subset such that:

```text
Y_PRA = W_O^l Attn(
    Q_l,
    [K_selected^l ; K_local^l],
    [V_selected^l ; V_local^l]
)

Y_PRA ~= Y_full
```

with selected K/V much smaller than full historical K/V.

Primary metrics:

```text
quality_gap = loss_PRA - loss_full_SA
active_fraction = active_token_KV / total_accessible_token_KV
```

The central scaling question is whether quality stays close to full SA while `active_fraction` becomes very small as accessible context grows.

Do not claim literal constant-cost infinite context. Archival storage and routing/index costs remain. Prefer **approximately unbounded context with bounded/small active token-level attention**.

---

## 3. Transport modes

Add an explicit config parameter, adapting naming to repository conventions if needed:

```python
memory_transport: Literal["native_kv", "cross_attention"] = "native_kv"
```

### `native_kv` — canonical PRA

At each PRA layer:

1. Compute local Q/K/V normally.
2. Route over references/chunks/gists.
3. Materialize selected **native layer-specific K/V**.
4. Concatenate selected memory K/V with local K/V.
5. Apply **one shared attention softmax** over combined keys.
6. Use the existing `o_proj`.
7. Do not use `mem_o_proj`.
8. Do not separately normalize local and memory attention.
9. Do not require `memory_alpha`.
10. Add no PRA-specific trainable transport parameters.

Conceptually:

```python
k = concat(k_memory, k_local, dim=sequence_dim)
v = concat(v_memory, v_local, dim=sequence_dim)
weights = softmax(q @ k.transpose(-2, -1) * scale + combined_mask)
out = o_proj(weights @ v)
```

Handle masks and memory visibility correctly.

### `cross_attention` — optional adapted transport

Preserve existing behavior:

```text
local_out  = o_proj(Attn(Q, K_local, V_local))
memory_out = mem_o_proj(Attn(Q, K_memory, V_memory))
out = local_out + memory_alpha * memory_out
```

Do not delete old code/results.

Potential advantages: preserves the local SA path, offers a trainable adapter/no-op initialization, and may compensate for representation/calibration differences.

Costs: separate softmax normalization, new trainable parameters, alpha/gating calibration, possible reference-conditioned training, and departure from pure native-KV reuse.

Treat it as an optional extension, not core PRA.

---

## 4. Gists are indexes, not transported memory

Maintain:

```text
routing representation != transported representation
```

Mean, last, multi-gist, K-means, SOM, Hebbian/prototype, hybrid, etc. should normally summarize the model's own layer-specific key geometry for routing:

```text
native chunk K/V
 -> gist(s)
 -> query/gist scoring
 -> select chunk
 -> retrieve original chunk K/V
 -> native attention
```

Do not replace token K/V with gists unless a separately named compressed-transport experiment explicitly tests that.

GRU gists are learned routing representations. They need not exactly follow native K distribution if used only for routing. A GRU gist does not imply `mem_o_proj` is required.

---

## 5. Historical K/V vs independently re-encoded chunks

For a causal Transformer, if prefix `A` is processed normally before continuation `B`, cached K/V for `A` are the same historical K/V full causal attention would use while processing `B`: future `B` cannot alter `A`.

Cleanest PRA case:

```text
process A normally
 -> externalize/store A native K/V
 -> later process B
 -> retrieve selected A K/V
 -> inject into native attention
```

No representation adapter is theoretically required.

If an arbitrary chunk is independently re-encoded without its original left context, deeper states can differ because preceding context is absent. Position can also differ. Measure these approximation errors instead of immediately hiding them with training.

Progression:

```text
exact historical K/V
 -> independently re-encoded chunks
 -> positional policies/remapping
 -> optional learned/adapted transport
```

---

## 6. Position handling

Do not assume position is the only source of chunk differences; missing left context can alter deeper representations.

Preserve intra-chunk order. Prepare explicit later ablations such as:

```text
original_global_position
contiguous_before_local
fixed_virtual_distance
distance_clipped
```

For RoPE/HF models, investigate pre-RoPE key caching or undo/reapply rotations when assigning retrieved memory a new/virtual position.

Do not add learned positional correction merely to make the first experiment work.

---

## 7. Training philosophy

First PRA experiments require **no PRA/reference training**:

```text
train ordinary SelfAttention LM
 -> freeze checkpoint
 -> full-context SA
 -> tail/local-only SA
 -> native-KV PRA inference:
      all-memory
      sparse oracle
      routed
      shuffled
      disabled
```

Add tests confirming `native_kv` introduces no required trainable parameters.

Existing scratch/joint/frozen cross-attention adaptation experiments remain valid but answer a different question:

> Can an adapted memory pathway be added while preserving base language behavior?

They must not imply that PRA intrinsically requires reference-conditioned training.

---

# 8. Paper 1 experimental progression

## A — Train SelfAttention baseline

Train the existing small SA model sufficiently to obtain a usable LM. Report architecture, budget, seeds, loss/perplexity, accuracy if useful, and hardware.

This is LM pretraining, not PRA training. Freeze the checkpoint thereafter.

## B — Native-KV equivalence sanity check

For causal sequence `A | B`, compare ordinary full SA with PRA where A's already-computed native K/V are externalized and injected back while B is evaluated.

Expected in exact-cache conditions:

```text
loss_PRA_native_all ~= loss_SA_full
```

If this fails materially, debug transport/masks/positions before routing.

## C — Fixed-target split scaling

Test:

```text
splits = [2, 3, 5, 8, 16, 32, 64]
```

**Keep the source document, final local tail, prediction target, and evaluated target tokens identical across split counts.** Only partition the displaced prefix differently.

Do not retain the old confound where split count changes target span or processed-token budget.

Prefer automatically representing the displaced prefix as implicit `#__head` with internal chunks, using the same mechanism intended for prompts larger than direct-access capacity.

## D — Oracle native-KV retrieval

For each split count, where feasible evaluate:

```text
SA-full
SA-tail
PRA-native-all
PRA-native-oracle
PRA-native-disabled
PRA-native-shuffled
```

Definitions:
- `SA-full`: dense full context.
- `SA-tail`: only direct/local tail.
- `PRA-native-all`: all displaced K/V reintroduced; isolates transport/partition error.
- `PRA-native-oracle`: only known relevant chunks reintroduced; tests sparse approximation.
- `PRA-native-disabled`: memory omitted; should reproduce local-only behavior.
- `PRA-native-shuffled`: wrong/misaligned content; tests causality.

Report:

```text
transport_gap = loss(PRA-native-all) - loss(SA-full)
sparse_gap = loss(PRA-native-oracle) - loss(PRA-native-all)
memory_benefit = loss(SA-tail) - loss(PRA-native-oracle)
content_causality = loss(PRA-native-shuffled) - loss(PRA-native-valid/oracle)
```

## E — Active-KV budget frontier

For each split count vary selected/materialized chunks, e.g.:

```text
top_k_chunks = 1, 2, 4, 8, ...
```

Report:
- active token K/V;
- total accessible token K/V;
- active fraction;
- loss/perplexity;
- gap to full SA.

Plot:

```text
quality gap vs active-KV fraction
quality vs total accessible context at fixed active-KV budget
```

The desired result is growing accessible context without proportional growth of active token-level attention.

## F — Routed native-KV retrieval

Only after oracle/native transport works, test routing.

Compare oracle, routed and shuffled. Report reference/chunk accuracy, recall@k, oracle gap, routed loss, selected K/V count, routing/index cost and token-attention cost.

This isolates transport error from selection error.

## G — Dependency-sensitive evaluation

WikiText-2 is useful for natural-language continuity, but many targets may not require distant context.

Where feasible compute:

```text
dependency_gain = loss_tail - loss_full
```

Stratify examples and separately report cases where full history materially improves prediction.

Later add tasks where displaced evidence is explicitly necessary.

## H — Existing cross-attention experiments

Move current reference-conditioned experiments here/later.

Preserve:
- scratch reference training;
- joint fine-tuning;
- frozen adaptation;
- catastrophic forgetting;
- routing accuracy;
- shuffled/oracle controls;
- paired-seed statistics.

Reframe them as evaluation of an **adapted cross-attention transport**, not evidence that native PRA requires training.

---

# 9. Paper 0 changes

Paper 0 is the architectural/position paper. Modify conservatively but make the core hypothesis unambiguous.

### Abstract/introduction

State that canonical PRA:
- operates on model-native layer K/V;
- can be enabled at inference time;
- does not intrinsically require reference-conditioned training;
- progressively selects sparse historical K/V;
- approximates full long-context self-attention;
- aims to decouple accessible context size from active token-attention size.

Avoid framing PRA as RAG inside attention.

### Add/strengthen “PRA as approximation to full self-attention”

Compare full attention over `N` historical states with local + `R` selected states, `R << N`.

Discuss the desired regime:

```text
N -> very large
R -> small/bounded
R/N -> 0
quality_gap -> small
```

while acknowledging archival storage and routing cost.

### Clarify references

References include:
- explicit user references;
- historical chunks;
- automatically generated implicit references;
- displaced prompt prefix `#__head`.

A user should be able to provide a long prompt without manually writing reference syntax.

### PRA versus RAG

Add a concise section:

```text
RAG/search = context acquisition
PRA        = context management/selective activation
```

If RAG is needed:

```text
search -> content -> model-native K/V -> PRA
```

### Selection versus transport

Make `native_kv` canonical. Describe cross-attention as optional adapted transport. Clarify that gists normally index/select native K/V rather than replace them.

### Complexity

Emphasize bounded/small active token-level attention, not constant total system cost. Discuss flat gist scans, hierarchical routing, archival storage, compression/prototypes and possible sublinear lookup.

---

# 10. Paper 1 changes

Reorganize Paper 1 approximately as:

```text
1. Experimental setup and SA pretraining
2. Native-KV equivalence
3. Split/context scaling: 2,3,5,8,16,32,64
4. Oracle sparse retrieval
5. Active-KV budget / quality frontier
6. Routed retrieval
7. Content-causality and dependency-sensitive controls
8. Independent-chunk / positional ablations
9. Cross-attention adaptation experiments (existing results)
10. Limitations and transition to pretrained LLMs
```

Do not lead with the current 25 plain-language / 40 reference-condition run story. Preserve it later.

Primary claim, **only if supported by new results**:

> A normally trained causal Transformer can be converted to PRA at inference time without reference-conditioned training, and sparse reactivation of model-native historical K/V can approximate larger-context self-attention under a much smaller active attention budget.

Do not claim this before the data support it.

---

# 11. Existing results

Preserve current tables/statistics unless genuinely superseded.

They can still support:
- cross-attention PRA can train stably;
- frozen adaptation avoids catastrophic forgetting;
- joint tuning can degrade ordinary LM performance;
- current routing/content causality is weak;
- current paired-seed statistics.

These become secondary adapted-transport findings and must not imply PRA itself requires a reference adaptation stage.

---

# 12. Paper 2 transition — Hugging Face PRA

End Paper 1 and update the Paper 0 roadmap so Paper 2 follows naturally:

> If PRA is fundamentally an inference-time transformation over native attention K/V, the next question is whether existing pretrained Hugging Face causal LLMs can be wrapped with PRA without retraining their base weights.

Paper 2 should investigate:
- wrapping existing HF attention modules;
- native per-layer K/V extraction/externalization/reinjection;
- MHA/GQA/MQA;
- RoPE remapping and virtual positions;
- modern causal masks;
- SDPA/FlashAttention compatibility;
- HF cache formats;
- quantized K/V caches;
- fixed active attention budgets with growing logical context;
- exact historical cache first;
- independently encoded references second;
- routing/gist scalability;
- latency, memory and quality;
- learned adapters only after the zero-training baseline.

Paper 1 should avoid toy-only choices that cannot map cleanly onto standard HF attention.

---

# 13. Tests and safeguards

Add focused tests:

1. **Native equivalence:** restoring all historical native K/V with equivalent masks/positions gives output close to ordinary SA.
2. **Disabled-memory equivalence:** native PRA with no memory matches local-only SA.
3. **No training requirement:** native transport does not require `mem_o_proj` or other trainable transport parameters.
4. **Cross-attention backward compatibility:** old mode/checkpoints remain runnable.
5. **Routing/transport separation:** tests can bypass routing and directly supply oracle chunks.
6. **Fixed-target split generation:** all split variants share identical local tail, target tokens and evaluation mask.
7. **Causality controls:** shuffled/wrong memory is reproducible and genuinely differs from oracle memory.
8. **Metrics:** explicitly record active K/V and total accessible K/V.

---

# 14. Implementation discipline

Keep changes minimal and modular. Reuse existing reference/chunk/cache/gist mechanics. Do not rewrite the project around the new transport mode.

Localize the transport distinction around attention materialization/forward computation. Keep selection/materialization transport-independent where possible.

Document tensor shapes, masks, and whether keys are pre/post positional transformation.

Preserve existing CLI/config behavior where feasible, but make `native_kv` the intended default for new core experiments.

Do not silently change old results. Regenerate only experiments whose definitions change, and label legacy/current cross-attention results clearly.

---

# 15. Completion criteria

Complete when:

- `native_kv` and `cross_attention` are explicit supported modes;
- native K/V uses shared normalization + existing `o_proj`;
- native PRA runs from an SA checkpoint without PRA training;
- native/oracle equivalence tests pass;
- fixed-target 2/3/5/8/16/32/64 evaluation exists;
- active-KV scaling metrics are emitted;
- oracle/routed/shuffled/disabled controls exist;
- existing cross-attention experiments remain runnable;
- Paper 0 reflects the inference-time sparse-full-attention hypothesis and PRA/RAG distinction;
- Paper 1 is reordered around training-free native-KV experiments;
- both papers motivate Paper 2 on Hugging Face pretrained-model adaptation.

## Guiding principle

When implementation choices are ambiguous, preserve this invariant:

> **PRA should first attempt to make the right native K/V available to the pretrained attention operation, rather than train a new operation to consume memory.**

## Strategic north star: PRA as an inference-economics result

Do not reduce PRA to “reference-aware attention”, “memory augmentation”, or “RAG inside the Transformer”.

The high-impact hypothesis is stronger:

> **PRA may allow accessible/effective context to grow by orders of magnitude while keeping expensive active token-level attention approximately bounded or slowly growing.**

If this holds on existing pretrained LLMs, PRA could change the economics of long-context inference because context length would become increasingly dominated by:

- storage of historical K/V,
- indexing/routing,
- movement of selected K/V between memory tiers,

rather than by dense neural attention over the entire accessible history.

The desired asymptotic regime is:

```text
accessible context N -> very large
active token K/V R -> small or approximately bounded
R / N -> 0
quality gap to full attention -> small



Then add this to the **Paper 1 experimental progression** section:

```md
### Add system-scaling metrics to every long-context experiment

Do not report only loss/perplexity.

For every split/context-size condition, record where technically available:

```text
accessible tokens
active local tokens
retrieved token K/V
active_fraction
number of references/chunks
routing comparisons
routing latency
attention latency
total inference latency
peak GPU memory
KV storage size
KV transfer volume
gap to full-SA quality

quality gap vs accessible context
active K/V vs accessible context
active fraction vs accessible context
latency vs accessible context
GPU memory vs accessible context
quality gap vs active-KV budget

accessible context increases by a large factor
active K/V remains approximately constant
latency/memory grow much more slowly than dense attention
quality remains close to full-context SA