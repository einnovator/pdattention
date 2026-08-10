# Paper 1.5 RoPE Experiments

This directory isolates the matched position-semantics experiments from canonical PRA core.
The default run trains matched absolute/RoPE self-attention checkpoints, converts each to
parameter-preserving native-KV PRA, and evaluates fixed-target synthetic examples across
2, 5, 16, 32, and 64 source splits.

```powershell
python experiments/paper1_5_rope/run_core_experiments.py --device cuda
python experiments/paper1_5_rope/eval_distance_policy.py --device cuda --iterations 500
python experiments/paper1_5_rope/eval_logical_offsets.py --device cuda
python experiments/paper1_5_rope/eval_head_offset_progression.py --device cuda --max-examples 4
python experiments/paper1_5_rope/summarize_next_iteration.py
```

## Final Validation Matrix

The night validation is resumable and writes only under
`docs/papers/shared/results/paper1_5_rope/validation/`:

```powershell
python experiments/paper1_5_rope/train_validation_checkpoints.py --device cuda
python experiments/paper1_5_rope/eval_logical_offsets.py --device cuda `
  --position-modes absolute sinusoidal rope `
  --output-dir docs/papers/shared/results/paper1_5_rope/validation `
  --result-name positional_mechanism_offset_validation
python experiments/paper1_5_rope/run_wikitext_validation.py --device cuda
python experiments/paper1_5_rope/run_qa_validation.py --dataset hotpotqa --device cuda
python experiments/paper1_5_rope/run_qa_validation.py --dataset qasper --device cuda
```

Use `--smoke --tiers tiny --position-modes sinusoidal --seeds 1` to reproduce one
two-step path before a full run. Full defaults use five matched seeds and both tiers.

### Expectations Recorded Before Validation

| Family | Expected | Why |
|---|---|---|
| Sinusoidal | offsets reduce reset error | supports general source continuity rather than a RoPE-only effect |
| Small models | offset-effect direction survives increased capacity | the coordinate contract should not depend on the tiny tier |
| WikiText | offsets reduce representation error and often, but not always, improve loss | missing contextualization can dominate position repair |
| HotpotQA/QASPER | positional fidelity improves while routing/composition remain limiting | correct K positions do not choose a useful memory set |
| Overlap | improvement may follow position repair but need not be monotonic | historical context has a cost and can alter distractor composition |

These expectations are directional hypotheses, not acceptance criteria. Dense access is a
control rather than an assumed oracle, and the evidence-only condition may be worse than a
router-selected combination.

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
