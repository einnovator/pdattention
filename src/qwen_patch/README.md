# Qwen Patch Notes

Qwen2/Qwen2.5 are good first HF targets because small checkpoints are available.

Recommended target:

- Qwen/Qwen2.5-0.5B

Constraints:

- Preserve RoPE.
- Preserve GQA.
- Use eager attention initially.
- Patch only last 2-4 layers.

Start with generic `hf_wrappers/pra_wrapper.py`, then specialize only if needed.
