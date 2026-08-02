# Hugging Face PRA Wrappers

This folder is for Phase 2, after the standalone PyTorch prototype works.

Recommended first approach: do not replace model attention. Instead wrap selected upper layers:

```text
original_self_attention(hidden_states) + alpha * pra_cross_attention(hidden_states, resolved_ref_memory)
```

This keeps pretrained model behavior mostly intact.

Use small models first:

- Qwen/Qwen2.5-0.5B
- TinyLlama/TinyLlama-1.1B-Chat-v1.0

Use:

```python
attn_implementation="eager"
```

Avoid FlashAttention/SDPA until the wrapper works.
