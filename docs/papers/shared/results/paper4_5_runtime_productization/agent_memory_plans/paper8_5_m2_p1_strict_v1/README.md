# Paper 8.5 M2/P1 frozen plans for Paper 4.5

These fixtures are the engine-realization handoff for the three strict
exact-prefix M2/P1 successes in Paper 8.5. They contain no new routing decision.
Each row reconstructs and hash-validates the full pre-selection request, records
the exact mandatory inline indices for the current episode, and materializes the
remaining selected logical records as ordered Paper 4.5 resources.

| Task | Requests | Fixture SHA-256 | Paper 8.5 logical outcome |
|---|---:|---|---|
| Task 3 | 7 | `04c9fbad18b9a98648da86edecf8a88fe40dc915a1e58e75154ab472dcd2f1e5` | solve, 11.52% own-trajectory saving |
| Task 4 | 6 | `b1f0b05f3266d2d3cf2dd18c3d308da90898aedd9d4b7909b3a6f4e80d8bf777` | solve, 52.56% own-trajectory saving |
| Task 5 | 9 | `e292c5e255a17d4b77f354f09a73cedbae2e9172d965872811f537d0cf9dfc82` | solve, 49.64% own-trajectory saving |

Paper 4.5 must use `--selection-policy paper8.5-frontier-dag-m2-p1-v1`
with the corresponding `--selection-replay` file. The runner fails closed if a
request hash diverges, if the plan omits system/current state, or if the fixture
content digest changes. The same file is to be replayed unchanged for HF, MLX,
SGLang, vLLM, and llama.cpp.

The logical policy result remains a Paper 8.5 claim. Paper 4.5 reports only
engine realization: resident selected K/V, history re-encoding, K/V-copy bytes,
consumer temporaries, lifecycle correctness, and end-to-end behavior.
