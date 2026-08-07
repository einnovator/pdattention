# AGENTS.md — Instructions for Codex / Coding Agents

## Mission

Build a research prototype of **URI-addressed Progressive Retrieval Attention (PRA)**.

The main deliverable is a working standalone PyTorch package that can train and evaluate a tiny decoder-only language model using PRA over explicit lightweight reference tokens backed by a runtime reference table.

## Core architecture

A prompt may contain explicit reference tokens:

```text
The answer is in <REF_1>.
```

The runtime should:

1. Parse reference tokens from text.
2. Resolve each token through a runtime `ReferenceTable` to a URI, document fragment, summary, and optional children.
3. Encode the resolved fragment in a separate inference context.
4. Capture/cache layer-specific K/V tensors for PRA layers.
5. During main inference, let selected layers retrieve the matching reference memory.
6. Attend/cross-attend to retrieved K/V in addition to recent tokens.
7. Optionally recurse into child anchors if the selected reference is still only a summary.

## Implementation priorities

### Priority 1: standalone PyTorch

Work in:

```text
pra_torch/
experiments/
tests/
notebooks/
```

Implement clean, readable code with shape comments.

Required components:

- `config.py`
- `refs.py`
- `resolver.py`
- `memory.py`
- `attention.py`
- `model.py`
- `data.py`
- `train.py`
- `eval.py`

### Priority 2: tests

Add tests for:

- Reference parsing.
- URI resolving.
- Cache construction.
- Attention shape correctness.
- Training loop sanity.

### Priority 3: Hugging Face wrappers

Only after standalone works.

Start with compatibility wrapper:

```text
original self-attention output + alpha * PRA cross-attention output
```

Do not directly replace Llama/Qwen/Mistral attention first.

## Do not do initially

- Do not start with vLLM.
- Do not start with CUDA kernels.
- Do not start with FlashAttention.
- Do not directly patch all layers of Llama 3.x.
- Do not assume the same memory K/V works across layers.
- Do not use raw token embeddings as final reference memory for all layers except in toy/debug mode.

## Research questions

1. Can summary handles guide memory expansion better than one-shot RAG?
2. Can a small model with PRA solve synthetic long-context QA at lower token cost?
3. Which trigger works best?
   - attention to ref token
   - cosine similarity between hidden state and summary vector
   - learned gate
   - hybrid score
4. How much fine-tuning is needed for pretrained models?
5. Can recursive anchor expansion reduce context cost without losing recall?

## Acceptance criteria for first prototype

A successful first prototype should run:

```bash
python -m pra_torch.cli train --steps 200
python -m pra_torch.cli eval
pytest
```

and produce a report comparing:

- no refs
- full context
- simple RAG
- PRA
