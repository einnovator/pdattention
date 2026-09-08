# Paper 3.3 factorial and budget audit

This artifact separates evidence admission from retrieval-linked neural
interaction and tests whether sparse interaction becomes more useful as the
selected source context grows from 512 to 1,024 tokens.

## Canonical runs

Both canonical runs use `mlx-community/Qwen3-1.7B-4bit` at revision
`3b1b1768`, the same 150 frozen MultiHop-RAG identities, macOS 26.6.2, Python
3.12.14, NumPy 2.5.2, Torch 2.13.0, and Transformers 5.16.1. They were produced
from commit `99f9fd45`.

| Directory | Source budget | Conditions | Rows | Selection SHA-256 | Elapsed |
| --- | ---: | ---: | ---: | --- | ---: |
| `raw_512` | 512 | 8 | 1,200 | `a4b30cffe7f0...dd294ee` | 2,973.8 s |
| `raw_1024` | 1,024 | 4 | 600 | `cc4c0d7fea7f...c73bb43` | 3,568.4 s |

The selection digests differ because the source-token budget changes. The
cohort, model revision, query, candidate set, router configuration, prompt, and
evaluator remain fixed.

## Main results

At 512 tokens, pair SA alone is statistically compatible with independent PRA.
Expansion lowers F1 from 0.0896 to 0.0761; expansion plus pair SA reaches
0.0874 and recovers 84.1% of that loss. The paired
difference-in-differences is `+0.0092 [-0.0090, 0.0280]` F1 and
`+0.0400 [-0.0600, 0.1333]` official score.

At 1,024 tokens, packed RAG exceeds independent PRA by 0.0301 F1. Full-pair SA
recovers 0.0112 F1, or 37.1% of that deficit, while selecting 3.69% of physical
edges. Boundary-only SA recovers 0.0126 F1, or 41.8%, with 0.0083% of physical
edges. Both paired F1 intervals exclude zero, but neither reaches packed F1 and
official-score recovery is weak. The implementation uses dense kernels, so
edge fractions are mechanism measurements rather than realized speedups.

`raw_512_mac8` records the aggregate and provenance summary from a complete
second-host replay with the same selection digest and model revision. It
preserves the positive factorial direction, but differs in Python, NumPy,
Torch, sentence-transformers, and macOS versions. It is kept as a non-pooled
robustness artifact and is not used in the canonical cross-budget figure.

## Reproduce the publication summary

```bash
PYTHONPATH=src python -m experiments.paper3_3_crossdoc_expansion.summarize_factorial \
  --run-512 docs/papers/shared/results/paper3_3_crossdoc_expansion/factorial_budget_audit/raw_512 \
  --run-1024 docs/papers/shared/results/paper3_3_crossdoc_expansion/factorial_budget_audit/raw_1024 \
  --replication-512 docs/papers/shared/results/paper3_3_crossdoc_expansion/factorial_budget_audit/raw_512_mac8 \
  --output docs/papers/shared/results/paper3_3_crossdoc_expansion/factorial_budget_audit
```

The five bootstrap seeds are deterministic paired resampling calculations over
one model execution per condition. They are not five independently trained or
decoded model runs.
