# Instruction-epoch N=3 fixed-trajectory oracle

`evidence.json` replays the clean 3/3 boundary-free persistent-FULL sequence
with the locked Qwen3-Coder tokenizer. It keeps the system prompt, every
genuine user instruction, and the complete latest instruction epoch, while
retiring assistant/tool causal groups from older epochs.

This is logical opportunity evidence, not an autonomous quality result. The
three source trajectory paths and SHA-256 digests are embedded in the evidence
file. The corresponding autonomous E0 treatment is registered by
`autonomous_persistent_instruction_epoch_n3_e0_v1.json`.
