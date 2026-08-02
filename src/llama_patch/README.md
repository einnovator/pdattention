# Llama Patch Notes

Target: Llama 3.x style decoder-only models.

Do later, after standalone + generic HF adapter.

Key constraints:

- Preserve RoPE handling.
- Preserve GQA layout: num_attention_heads may differ from num_key_value_heads.
- Preserve past_key_value cache format.
- Use `attn_implementation="eager"` first.
- Patch only upper layers first.

Recommended first Llama-family target: TinyLlama or another small Llama-like model, not Llama 3.x 8B/70B.

True PRA K/V injection requires storing per-layer pre/post-RoPE K/V for resolved refs. Safer first version: cross-attention adapter.
