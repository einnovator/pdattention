# PRA conditional memory-use adapter

**Status: research opt-in. This is not the SDK default.**

- Base model: `Qwen/Qwen3-0.6B`
- Type: conditional output-projection LoRA
- Rank / alpha: 32 / 32
- PRA depth: last 14 layers (14--27)
- Parameters: 1,376,256 (0.2309% of base)
- Selected seed: 11

The adapter improves oracle memory integration but degrades learned-routing HotpotQA. Keep
frozen PRA plus the compatible router as the product default. This artifact is provided for
controlled oracle and adaptation studies. PRA-off execution structurally bypasses these weights.
See the sweep result directory for five-seed metrics and the full validation grid.
