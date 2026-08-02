# AGENTS.md — Paper 2

## Goal
Write the paper on integrating PRA with existing pretrained Hugging Face causal language models.

## Main Contribution
Show how PRA can be introduced as an adapter/cross-attention branch while preserving original self-attention, RoPE, GQA/MQA, masks, and cache behavior.

## Must Cover
- Why full attention replacement is fragile.
- Why wrapper/adapters are safer.
- Qwen/TinyLlama first.
- Llama 3.x, Mistral, Gemma, DeepSeek as progressively harder targets.
- Frozen base model plus trainable PRA modules.
- Optional LoRA/PEFT.
- Why native KV injection is deferred.

## Experiments
Compare:
- base model
- prompt RAG
- summary-first RAG
- PRA adapter
- fine-tuned PRA adapter

## Avoid
- Do not claim compatibility with every model unless tested.
