# Gemma checkpoint matrix

Pinned checkpoint: `google/gemma-3-1b-it` at `dcc83ea841ab6100d6b47a070329e1ba4cf78752`.

| Stage | Architecture | Trainable scope | Status |
|---|---|---|---|
| G0 | native local/global Gemma | none (evaluation) | pending measured baseline |
| G1 | global slots (5, 11, 17, 23) replaced by PRA | none | pending measured frozen baseline |
| G2 | exact-slot Gemma-PRA | Q/O + following MLP LoRA | implementation ready; training not launched |
| G3 | exact-slot Gemma-PRA | Q/K/V/O + following MLP LoRA | implementation ready; training not launched |
| G4 | exact-slot Gemma-PRA | broad decoder LoRA | implementation ready; training not launched |
| G5 | exact-slot Gemma-PRA | full weight | blocked on benchmark/distributed budget |
| G6 | Gemma-like PRA native | full scratch | gated on G5; not launched |
