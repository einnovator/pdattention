# Paper 4 controlled model ladder

## Fixed architecture

- Controlled randomized associative chains from Paper 2.5
- 64 hidden dimensions, 8 decoder layers, 4 attention heads
- RoPE positions and 256-token native-operation limit
- LocalSA window: 16 tokens
- PRA consumer layers: zero-indexed layers 3 and 7
- Native K/V transport under one memory-plus-local softmax
- Oracle-fixed memory identities during Gate 0

The oracle intervention removes routing and materialization errors from the
training signal. It asks only whether transformer plasticity makes correct
sparse memory more useful. No evidence labels are supplied to the model; they
are retained solely for blinded attention diagnostics.

## Matched ladder

| Rung | Initialization | Trainable components | Training steps |
|---|---|---|---:|
| GlobalSA | matched random initialization | all weights | 800 |
| LocalSA | same seed and initialization as GlobalSA | all weights | 800 |
| Frozen PRA | converted LocalSA checkpoint | none | 0 |
| Consumer LoRA | converted LocalSA checkpoint | PRA Q/O and immediate FFNs | 500 |
| Interface LoRA | converted LocalSA checkpoint | PRA Q/K/V/O and immediate FFNs | 500 |
| Broad LoRA | converted LocalSA checkpoint | all decoder linear modules | 500 |
| Full-weight PRA | converted LocalSA checkpoint | all weights | 500 |
| Native PRA | matched random initialization | all weights from scratch | 1,300 |

Native PRA and full-weight continued PRA receive the same total number of
training steps. Every converted rung begins from the same seed-local LocalSA
checkpoint and uses the same PRA topology and controlled corpus.

## Causal memory conditions

- `none`: query-only local computation
- `matched_distractor`: the same number of non-evidence facts as gold facts
- `evidence_only`: only exact path facts
- `whole_parent`: path facts plus all generated distractors

The primary causal effect is the gold-answer margin under `evidence_only`
minus the margin under `matched_distractor`. Memory modularity compares
`evidence_only` with `whole_parent`.
