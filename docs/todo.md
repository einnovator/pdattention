# TODO

## Standalone PyTorch

- [ ] Improve tokenizer from char-level to small BPE or byte-level.
- [x] Add lightweight reference tokens with runtime table instead of char placeholder.
- [ ] Track actual reference token positions.
- [ ] Trigger retrieval from attention-to-ref-token score.
- [ ] Add learned gate.
- [ ] Support batch-specific PRA caches.
- [ ] Add recursive resolver expansion.
- [ ] Add full baseline runner.
- [ ] Add accuracy metrics.
- [ ] Save/load tokenizer robustly.
- [ ] Add experiment config YAML.

## HF wrapper

- [ ] Add proper package import path.
- [ ] Add LoRA/PEFT training script.
- [ ] Use final hidden states for memory summaries.
- [ ] Add support for model-specific layer path detection.
- [ ] Add tests with tiny random HF model.

## Later research

- [ ] True per-layer K/V injection.
- [ ] RoPE-aware memory positions.
- [ ] vLLM paged-KV integration.
- [ ] CUDA/FlashAttention-compatible implementation.
