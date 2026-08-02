# Mistral Patch Notes

Mistral has sliding-window attention in many variants. This complicates direct K/V injection.

Recommended path:

1. Generic HF cross-attention adapter first.
2. Patch only upper layers.
3. Do not interfere with sliding-window local attention.
4. Treat PRA memory as separate cross-attention memory, not part of sliding window.

Potential issue: generation cache and attention masks differ across transformer versions.
