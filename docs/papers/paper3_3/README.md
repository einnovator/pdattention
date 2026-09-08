# Paper 3.3: Cross-Document Dependencies for Persistent Native Memory

Paper 3.3 separates two requirements of cross-document composition: selecting
useful inter-record dependencies and executing them in a form the frozen
decoder can consume. Persistent independent records preserve reusable native
K/V, but omit the causal history created by an ordinary packed RAG prefix.

The matched Paper 3.2 audit places that boundary. On the same 150 Paper 3.3
identities, packed RAG and independent PRA score `0.427` and `0.433` near 512
selected source tokens. Near 1,024 tokens they score `0.527` and `0.333`; the
paired independent-minus-packed interval excludes zero. Paper 3.3 studies the
dependency structure behind this context-scale transition.

## Evidence Boundary

- `ESTABLISHED_INHERITED`: Paper 3.2 packed, independent PRA, and rank-8
  residual endpoints, cited with their original provenance.
- `MEASURED_SMOKE`: Paper 3.3 host-observer parity, 0%/100% mask endpoints,
  oracle edge/mass sweeps, interaction localization on the mechanism cohort,
  and interventional target selection on ten frozen validation questions.
- `MEASURED_POWERED`: the 150-question frozen-test comparison of raw attention,
  pair-NLL, pair-JS, and pair-NLL-times-attention rankings; the corrected
  BM25-v2 cross-document expansion validation/test frontiers; and the
  150-question Qwen3-1.7B frozen generation and gold-support-oracle replays;
  plus the inherited 780-row matched 512/1,024-token protocol audit and the
  new same-host expansion-by-interaction factorial and budget replay.
- `MEASURED_REDUCED`: matched 30-question Qwen3-4B, Qwen3-8B, and
  Llama-3.1-8B generation replays using the same frozen selection-cache digest.
- `DESIGN_ONLY`: the query-conditioned learned pair selector. Training remains
  locked unless the oracle gate passes.

The canonical split is in
`../shared/results/paper3_3_sparse_crossdoc/splits.json`. It excludes all 30
final Paper 3.2 residual-evaluation identities from Paper 3.3 train,
validation, and test data.

## Mechanism Decision

The ten-question Qwen3-1.7B-4bit mechanism cohort found:

| Condition | F1 | Official | Physical edges |
| --- | ---: | ---: | ---: |
| Packed RAG | 0.1490 | 0.600 | 100% |
| Independent PRA | 0.1431 | 0.600 | 0% |
| Oracle top-attention | 0.1488 | 0.600 | 0.1% |

The sparse point recovered 97.3% of the small F1 gap, but the absolute
prespecified `0.19/0.67` gate was not met and one ten-question cohort cannot
establish equivalence. Larger budgets were non-monotonic. Learned-selector
training therefore remained locked pending the powered oracle reported below.

## Interventional Validation

The frozen validation diagnostic compares raw attention with document-pair and
layer leave-one-group-out answer-NLL and first-step-JS rankings. Independent PRA
reversed the small-cohort endpoint and exceeded packed F1 (`0.1673` versus
`0.1094`), while packed retained higher official score (`0.600` versus `0.500`).
No ranking was monotonic through 1%. Pair-JS at 0.5% had the highest validation
F1 (`0.1727`), but its paired F1 interval versus packed included zero. This is a
target-selection result, not a test claim. Raw attention and all three pair
targets proceeded to the frozen 150-question test; layer-only targets did not.

## Powered Test Decision

The complete frozen test passed 150/150 ordinary/instrumented host checks and
150/150 full-replay checks. Best sparse points at or below 1% physical edges
were:

| Condition | Edges | F1 | Official |
| --- | ---: | ---: | ---: |
| Packed RAG | 100% | 0.1119 | 0.3333 |
| Independent PRA | 0% | 0.1129 | 0.3667 |
| Raw attention | 0.05% | 0.1148 | 0.3867 |
| Pair JS | 1% | 0.1205 | 0.3800 |
| Pair NLL | 1% | **0.1243** | 0.3933 |
| Pair NLL x attention | 0.1% | 0.1215 | **0.4000** |

Pair NLL improved mean F1 by 0.0124 over packed, but its paired 95% interval
was `[-0.0055, 0.0315]`. Every sparse frontier remained non-monotonic and the
absolute `0.19/0.67` gate was missed. Learned-selector training remains locked.
The canonical compact artifact is
`../shared/results/paper3_3_sparse_crossdoc/interventional_test_n150/publication_summary.json`;
it is hash-linked to the full 60 MB per-question manifest retained on the
experiment host.

## Retrieval-Guided Expansion Decision

The corrected BM25-v2 validation sweep evaluates 221 configurations over 150
questions. Dense selected-only search at `k=1` and a 64-source-token ceiling
was frozen for test. It gained 2.94 validation support-span points, but only
0.89 held-out points, while 95.3% of held-out admitted tokens were outside
annotated support. The 64-token support oracle gained 9.83 held-out points.

On 150 Qwen3-1.7B questions, direct dense expansion reduced F1 from 0.0896 to
0.0761. Retrieval-linked pair SA restored F1 to 0.0874. Gold-support expansion
also reduced F1, to 0.0829, while boundary SA restored it to 0.0895. Reduced
Qwen3-4B, Qwen3-8B, and Llama-3.1-8B cohorts had inconsistent signs. Expansion,
linked attention, recursive depth, and learned selection therefore remain
research-only. The tracked artifacts are under
`../shared/results/paper3_3_crossdoc_expansion/`; all model runs carry the
same frozen selection-cache SHA-256.

## Factorial and Budget Decision

The matched 512-token factorial separates evidence admission from linked
self-attention:

| Condition | F1 | Official |
| --- | ---: | ---: |
| Independent PRA | 0.0896 | 0.4333 |
| Pair SA only | 0.0917 | 0.4600 |
| Expansion only | 0.0761 | 0.3733 |
| Expansion + pair SA | 0.0874 | 0.4400 |

Pair SA recovers 84.1% of the expansion-induced F1 loss, but the factorial
interaction is `+0.0092` with interval `[-0.0090, 0.0280]`. At 1,024 source
tokens, full-pair and boundary-only SA improve F1 over independent PRA by
`0.0112 [0.0016, 0.0220]` and `0.0126 [0.0023, 0.0240]`, recovering 37.1% and
41.8% of the packed deficit. Boundary-only replay selects just `0.0083%` of
physical edges, but official-score recovery remains weak and no condition
reaches packed quality. These are logical-edge measurements on dense kernels.

The canonical 512/1,024 runs share one M5 host environment and model revision.
A complete 512-token replay from a second Apple host is retained as a
non-pooled robustness artifact. See
`../shared/results/paper3_3_crossdoc_expansion/factorial_budget_audit/`.

## Reproduce

See `experiments/paper3_3_sparse_crossdoc/README.md` for split generation,
fixture smoke, and natural oracle commands. Focused tests are:

```bash
PYTHONPATH=src python -m pytest \
  tests/test_sparse_crossdoc.py \
  tests/test_paper3_3_oracle.py \
  tests/test_rag_mlx_native.py \
  tests/test_rag_causal_decomposition.py -q
```

Build the paper from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error paper_3_3.tex
```

Compressed physical teacher graphs and the full per-question manifest are
reproducible experiment-host artifacts and are excluded from Git. The tracked
publication summary preserves provenance, aggregate conditions, endpoint
parity, localization summaries, paired intervals, runtime metadata, and the
full-manifest digest.
