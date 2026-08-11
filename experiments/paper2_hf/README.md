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

Llama and Gemma remain intentionally unimplemented until Qwen exposes no shared-core issue.
