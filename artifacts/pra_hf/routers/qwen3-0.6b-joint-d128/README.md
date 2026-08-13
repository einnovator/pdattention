# PRA routing adapter

This is the learned Qwen3-0.6B router used by the Paper 2 last-14 LoRA sweep. It is compatible
with the public `pra_hf` router artifact API. The recommended default is frozen PRA plus this
router. The linked conditional memory-use adapter is a research-only opt-in because it improves
oracle integration but degrades routed HotpotQA.
