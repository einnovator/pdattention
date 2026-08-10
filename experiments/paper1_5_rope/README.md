# Paper 1.5 RoPE Experiments

This directory isolates the matched position-semantics experiments from canonical PRA core.
The default run trains matched absolute/RoPE self-attention checkpoints, converts each to
parameter-preserving native-KV PRA, and evaluates fixed-target synthetic examples across
2, 5, 16, 32, and 64 source splits.

```powershell
python experiments/paper1_5_rope/run_core_experiments.py --device cuda
python experiments/paper1_5_rope/eval_distance_policy.py --device cuda --iterations 500
```

Use `--smoke` for one two-step seed. Canonical JSON, CSV, and plots are written under
`docs/papers/shared/results/paper1_5_rope/`; resumable checkpoints remain under `out/`.
Distance-policy alternatives are experiment-only and do not change canonical post-position K.

## Expectations Recorded Before the Logical-Offset Run

| Comparison | Expected | Why |
|---|---|---|
| post-local vs post-offset | offset improves; exact layer-0 parity | restores source-relative positions |
| pre-local vs post-local | near parity | same raw K and effective positions |
| pre-offset vs post-offset | near parity | same raw K and effective rotation |
| pre-offset runtime vs post-offset | pre is slower | deferred rotation adds work |
| offset to overlap | deeper layers improve | restores part of the missing left context |
| routed vs oracle after offset and overlap | oracle is no worse | isolates selection |
| K-only relocation vs common Q/K translation | K-only changes; common translation is invariant | relative-position algebra |

These are preregistered directional expectations, not acceptance criteria. Result JSON records
whether each observation agrees, partly agrees, or disagrees.
