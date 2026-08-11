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

Llama and Gemma remain intentionally unimplemented until Qwen exposes no shared-core issue.
