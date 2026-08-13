# Paper 2.5 iterative PRA experiments

Run commands from the repository root.

## Projection and local-associative gates

Gate 1 corrects asymmetric frontier projection without retraining:

```powershell
python experiments/paper2_5_iterative_pra/run_local_associative_closure.py --device cuda
```

Gate 2 captures 256-token contextual parents with eight 32-token local means,
then compares one-shot parent routing, projection-correct parent closure, and
local-gist closure under the same final parent/KV budget:

```powershell
python experiments/paper2_5_iterative_pra/precompute_local_features.py --device cuda
python experiments/paper2_5_iterative_pra/run_gate2_local_closure.py --device cuda
```

Artifacts are written under
`docs/papers/shared/results/paper2_5_iterative_pra/local_associative_closure/`.
The public SDK keeps one-shot routing as its default. The opt-in
`routing_mode="local_iterative"` path requires a routing adapter.
