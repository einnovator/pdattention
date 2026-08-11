# Paper 2 Hugging Face Experiments

Paper 2 uses one shared PRA routing/materialization core beneath thin family adapters. The
first milestone is intentionally limited to Qwen, eager attention, one upper PRA layer, and
the frozen `Qwen/Qwen3-0.6B` checkpoint at revision
`c1899de289a04d12100db370d81485cdf75e47ca`.

Run the offline adapter tests:

```powershell
python -m pytest -q tests/test_hf_integration.py
```

Run the pretrained CUDA gate and write JSON/CSV artifacts:

```powershell
python experiments/paper2_hf/qwen/run_first_night.py --device cuda
```

The script evaluates disabled-adapter logits, hidden states, greedy generation and cache
shapes before enabling memory. It then captures native GQA K/V for one explicit reference,
runs cached generation, and moves an oversized logical prompt head into `#__head` while
checking the configured native-operation bound. Results are written beneath
`docs/papers/shared/results/paper2_hf/qwen/`.

Run the deliberately small unrestricted-QA smoke matrix:

```powershell
python experiments/paper2_hf/qa/run_smoke.py --device cuda
```

This compares question-only truncation, dense text truncated to the same native limit,
oracle evidence text-RAG, and zero-shot PRA on one HotpotQA and one QASPER example. It is a
pipeline diagnostic, not an accuracy estimate.

Run the staged routing-representation comparison:

```powershell
python experiments/paper2_hf/routing/run_representation.py --device cuda `
  --examples-per-dataset 8 --representations post_rope_key,pre_rope_key,hidden_state `
  --chunk-sizes 32 --top-k 3,8,16 --stem qwen_routing_representation
```

The runner captures matched pre-RoPE Q/K, post-RoPE Q/K, and attention-input hidden states,
but stores only the configured compact routing gist alongside post-RoPE native detail K/V. It
measures evidence ranks without generating answers and checkpoints after every example.

Expected before running:

- H1: pre-RoPE routing should improve evidence recall and weaken late-position score bias.
- H2: post-RoPE mean routing should degrade more as routing chunks grow.
- H3: larger `k` should improve recall while increasing selected fraction and K/V cost.
- H4: hidden-state means may win if native key features are not semantic retrieval features.

Observed results must be retained even when they reject these hypotheses. Llama remains gated
on sparse, position-independent evidence retrieval from a representative Qwen subset.

Before the independent confirmation seed, the Qwen-to-Llama promotion gate is fixed as:

- combined any-evidence recall@3 at least `0.70`;
- each dataset's recall@3 at least `0.60`;
- combined MRR at least `0.50`;
- mean selected fraction at most `0.10`; and
- absolute mean score-position correlation at most `0.15`.

The thresholds require useful sparse retrieval rather than merely beating random ranking or
recovering evidence by selecting a broad fraction of the source.

Before training a router, run the parameter-free contiguous multi-gist diagnostic:

```powershell
python experiments/paper2_hf/routing/run_representation.py --device cuda `
  --examples-per-dataset 8 --representations attention_input_hidden_state --chunk-sizes 32 `
  --gist-mode segment_mean --gist-counts 1,2,4,8 --top-k 3,8,16 `
  --stem qwen_routing_segment_mean
```

This keeps 32-token parent chunks and their post-RoPE native K/V fixed. Each parent is divided
into balanced contiguous sub-chunks, with one attention-input hidden-state mean per sub-chunk.
The chunk score is the maximum cosine score over its gists, and selecting several gists from one
parent can materialize that parent's K/V only once.

Predeclared expectations:

- H5: multiple segment means improve recall@3 and MRR over one mean at the same selected-chunk
  and materialization budgets;
- H6: gains saturate before eight gists if four- to eight-token summaries preserve enough local
  evidence identity; and
- H7: routing-index bytes and scoring time grow with gist count, while selected native-K/V bytes
  remain determined by parent chunks and the unchanged materialization budget.

Use the existing promotion thresholds. Select a candidate for independent confirmation by
recall@3, then MRR, then lower routing-index cost. Retain all results if the diagnostic fails;
that failure strengthens the case for an evidence-supervised router with frozen Qwen.

Observed over two 32-example confirmation subsets (64 evaluations per gist count):

- `G=1/2/4/8` recall@3 is `0.391/0.313/0.266/0.281`, and MRR is
  `0.326/0.292/0.234/0.249`; H5 is rejected;
- `G=8` reaches recall@16 `0.828` versus `0.797` for `G=1`, but does not improve sparse
  recall or MRR, so H6's proposed beneficial saturation is not observed;
- routing-cache overhead is `1.57%/3.14%/6.28%/12.56%`, while selected and materialized parent
  fractions are unchanged at fixed `k`; H7 is supported for bytes and K/V materialization, but
  warm routing time is effectively flat at this scale; and
- no setting passes the Qwen-to-Llama gate. The next intervention is the predeclared small
  evidence-supervised router with Qwen frozen.

Before training that adaptor, run one higher-resolution, parameter-free diagnostic:

```powershell
python experiments/paper2_hf/routing/run_representation.py --device cuda `
  --examples-per-dataset 8 --representations attention_input_hidden_state `
  --chunk-sizes 32 --gist-mode segment_mean --gist-counts 1,32 `
  --top-k 3,8,16 --stem qwen_routing_hidden_token_max
```

With a 32-token parent, `G=32` makes every segment one token. The existing maximum-over-gists
score is therefore exactly the diagnostic `max_t cos(h_query, h_t)` while selected parent IDs,
post-RoPE K/V payloads, and materialization limits stay unchanged.

Predeclared expectation H8: if token-max substantially improves recall@3 and MRR over `G=1`,
mean pooling is losing sparse evidence and richer pooling deserves further work. If token-max
also performs poorly, the more immediate problem is query/evidence alignment in the frozen
hidden-state space. Report the approximately 32-fold routing-index increase, measured routing
time, selected fraction, and materialized K/V separately; token-max is a diagnostic rather than
a proposed production index.

Observed on the matched 16-example diagnostic:

- combined recall@3 changes from `0.4375` at `G=1` to `0.3750` at `G=32`;
- combined MRR changes from `0.2755` to `0.3143`, but the direction differs by dataset:
  HotpotQA improves from `0.3384` to `0.4690`, while QASPER falls from `0.2126` to `0.1595`;
- the routing index grows from about `1.57%` to `50%` of detail-K/V bytes; and
- parent selected fractions and materialization budgets remain matched. H8 is not supported as
  a general sparse-routing improvement, though the HotpotQA MRR change suggests that pooling
  loss is dataset-dependent.

Run the canonical routing/payload path end to end:

```powershell
python experiments/paper2_hf/qa/run_hidden_postrope_confirmation.py --device cuda
```

This confirms HotpotQA, QASPER, and displaced `#__head` history with attention-input
hidden-state routing and native post-RoPE K/V. It reports routing recall, selected fraction,
materialized tokens, answer EM/F1, and native-limit violations separately.

Before learned alignment, run the final zero-parameter K-space control. A centered-RoPE gist
first pools pre-RoPE keys and then applies one native Qwen rotation at the exact known center of
the summarized span:

```text
centered gist = R(span_center) mean(K_pre)
query = Q_post
```

Primary matched representation comparison:

```powershell
python experiments/paper2_hf/routing/run_representation.py --device cuda `
  --examples-per-dataset 8 `
  --representations post_rope_key,pre_rope_key,attention_input_hidden_state,centered_rope_key `
  --chunk-sizes 32 --gist-mode mean --gist-counts 1 --top-k 3,8,16 `
  --stem centered_rope_gist_comparison `
  --output-dir docs/papers/shared/results/paper2_hf/routing/centered_rope
```

Centered segment sweep:

```powershell
python experiments/paper2_hf/routing/run_representation.py --device cuda `
  --examples-per-dataset 8 --representations centered_rope_key --chunk-sizes 32 `
  --gist-mode segment_mean --gist-counts 1,2,4,8 --top-k 3,8,16 `
  --stem centered_rope_segment_mean `
  --output-dir docs/papers/shared/results/paper2_hf/routing/centered_rope
```

Predeclared expectations:

- E1: centered-RoPE should beat post-RoPE mean and reduce its late-position correlation;
- E2: it may modestly beat pre-RoPE mean if approximate relative distance helps relevance;
- E3: attention-input hidden-state mean will probably remain the strongest sparse semantic
  router; and
- E4: `G=2/4/8` may improve centered routing because each gist reduces semantic pooling width
  and center-position approximation error simultaneously.

The runner also reports correlation with per-chunk native token-QK maximum and mean scores.
Exact fractional centers are primary; matched `floor` and `ceil` controls use
`--center-policy` only to verify that half-position handling is not driving the result.

Observed on the matched 16-example primary run:

- post/centered/pre/hidden Recall@3 is `0.125/0.188/0.313/0.438`, with MRR
  `0.131/0.168/0.301/0.275`;
- centered routing reduces score-position correlation from post-RoPE's `0.651` to `0.567`,
  but does not approach pre-RoPE (`0.008`) or hidden-state (`-0.021`) position neutrality;
- centered `G=1/2/4/8` Recall@3 is `0.188/0.125/0.250/0.250`, while Spearman correlation
  with native token-QK maximum ordering rises `0.521/0.602/0.710/0.874`; and
- exact/floor/ceil centers all give Recall@3 `0.125` on the matched eight-example control.

E1 is directionally supported but based on one paired gain and no losses; E2 is rejected, E3
is supported, and E4 has weak partial support. Centered-RoPE validates the positional-aggregation
mechanism but not a sparse semantic-routing solution. Freeze the current zero-parameter search
and proceed to the tiny learned hidden-state router with Qwen frozen.

Llama and Gemma remain intentionally unimplemented until Qwen exposes no shared-core issue.

## Query representation study

The next frozen-Qwen study holds memory routing fixed at one mean attention-input hidden-state
gist per 32-token chunk and varies only the representation of the current information need.
The predeclared broad validation comparison is:

```powershell
python experiments/paper2_hf/routing/run_query_strategies.py `
  --device cuda --split validation --example-offset 0 --examples-per-dataset 8 `
  --stem query_strategy_sweep
```

It compares the last-token baseline with recent uniform means (`W=4,8,16`), exponential
means (`W=16`, half-life `2,4,8`), a linear-decay control, the exact question-span mean,
and decayed question-span means (half-life `4,8`). Recall@3 is primary; Recall@8/16, MRR,
coverage, source-position correlation, query norm, cosine to the last-token query, and
pairwise query similarity are recorded. Exact question spans are benchmark metadata and are
not assumed to exist in ordinary generation.

Predictions recorded before execution:

- H1: moderate exponential recency pooling modestly improves Recall@3, especially on QASPER;
- H2: long uniform windows eventually blur the fully formed information need;
- H3: half-life 4--8 is the most plausible useful range; and
- H4: decayed exact-question pooling is the strongest hand-designed candidate.

After selecting at most three candidates on validation, confirmation uses new QA identities
starting at offset 8. The last-token baseline is always repeated. Query aggregation is promoted
only if its combined Recall@3 rises by at least `0.10`, neither dataset loses Recall@3, and
position-correlation magnitude does not increase by more than `0.10`. Otherwise the simplest
last-token query remains canonical and the result is treated as evidence for learned alignment.

The learned-alignment gate uses frozen features from 24 train, 8 validation, and 16 test
identities per dataset. It first compares shared and asymmetric linear projections at routing
widths 64 and 128, with five adapter seeds, for both `last` and the strongest aggregate ablation
(`question_exp_h2.0`). Every in-document chunk participates in the multi-positive contrastive
denominator, so the objective includes random, lexical, position-matched, and current-router
false-positive negatives without a lossy sampling stage. The Qwen backbone remains frozen.

The predeclared Qwen-to-Llama promotion gate requires combined Recall@3 at least `0.70`,
HotpotQA and QASPER Recall@3 each at least `0.50`, combined Recall@8 at least `0.80`, absolute
score-position correlation no greater than `0.20`, and a routing vector no wider than 128
float32 values. Five-seed variation, shuffled-label training, and Hotpot-to-QASPER plus
QASPER-to-Hotpot transfer are required before promotion.

Observed outcomes:

- On 32 identity-disjoint confirmation examples, last/question-decay-H2/uniform-W32 Recall@3
  is `0.469/0.250/0.313`; the last state remains canonical.
- Validation selected the asymmetric linear 1024-to-128 adapter (`262,144` parameters,
  `0.044%` of Qwen). Its five-seed held-out Recall@3 is `0.563 +/- 0.031`, with HotpotQA
  `0.400` and QASPER `0.725`; it does not pass the `0.70` combined gate.
- A shared 64-dimensional adapter reaches `0.650 +/- 0.041` held-out Recall@3, but this is an
  exploratory test-best result, not the validation-selected confirmatory model.
- Canonical shuffled-label Recall@3 is `0.231`. Hotpot-to-QASPER and QASPER-to-Hotpot transfer
  are `0.150` and `0.225`, so the learned geometry is not domain-general.
- The validation-selected adapter retrieves evidence in `4/8` end-to-end probes, while answer
  F1 remains exactly `0.090` with PRA disabled and enabled. Retrieval and causal use remain
  separate gates.

The Qwen-to-Llama gate is not passed. Increase routing-data diversity before increasing adapter
capacity; investigate memory-use alignment only after retrieval generalizes.
