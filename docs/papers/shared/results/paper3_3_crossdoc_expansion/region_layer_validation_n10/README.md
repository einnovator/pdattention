# Task-aware region/layer validation audit

This artifact is a ten-question validation mechanism audit at a fixed 1,024
selected-source-token budget. It tests 36 equal-cost cells per question:

```text
4 contiguous layer bands x 3 source regions x 3 target regions
```

Each region is an eight-token prefix, middle, or suffix window on retrieval-
linked record pairs. Singleton cells are ranked by per-example gold-answer NLL
gain from `NO_CROSS_DOC_PACKED`. This is an oracle headroom signal, not a
deployable selector. The best singleton and the union of the four best positive
singletons are decoded separately so selection quality is not conflated with
multi-cell consumption quality.

## Provenance

- code commit: `6d96fcc21d6ca6b372e7c14a8e6a7ff208f5f4a3`
- model: `mlx-community/Qwen3-1.7B-4bit`
- model revision: `3b1b1768f8f8cf8351c712464f906e86c2b8269e`
- selection-cache SHA-256:
  `8b666e2ec1503f822a6ddd492c90c0ea49f221ba73240b010b3cdfdcb7045bf5`
- run-configuration digest:
  `af9bcd0c17ba488cbf55df55be733aeb86fdfeeb499fa93787640dccf0bbdf27`
- host: medium Apple-silicon experiment host (M5, 16 GB)
- elapsed time: 2,593.0 seconds

Gold-answer probability reductions use float32 before `logsumexp`; this fixes
the coarse 0.125-NLL quantization observed in the preliminary smoke.

## Result

The singleton oracle realizes a mean NLL reduction of `0.1548` with a
conservative paired bootstrap interval `[0.0748, 0.2416]`, but its F1 change
from no-cross packed is `-0.0010 [-0.0029, 0]` and its official-score change is
exactly zero. The ranked four-cell union predicts `0.4382` additive NLL gain but
realizes only `0.0941 [0.0058, 0.1918]`; its F1 change is
`-0.0023 [-0.0048, 0]` and its official-score change is zero.
The singleton and union select `0.0082%` and `0.0328%` of physical edges on
average.

The suffix-to-prefix boundary hypothesis is not selected as the best cell on
any question. Its boundary-minus-equal-cost-control contrast is negative in
the first three layer bands and near zero in the last. In layers 0--6 the
contrast is `-0.3344 [-0.6836, -0.0731]`, and the boundary cell itself has mean
NLL gain `-0.4308 [-0.9197, -0.0891]`.

This validation gate does not justify a powered test or selector training.
It instead identifies a consumption/calibration problem: NLL-targeted local
effects do not translate into answer F1, and independently useful cells are
strongly non-additive.

All timing was measured through a dense host kernel under logical sparse masks.
It is diagnostic only and supports no runtime-benefit claim.

## Files

- `selection_cache.jsonl`: frozen ten-question selected contexts.
- `region_layer_diagnostics.jsonl`: 360 singleton selection measurements.
- `summary.json`: run metadata and aggregate condition means.
- `publication/region_layer_publication_summary.json`: strict paired reduction.

The 70-row `rows.jsonl` generation file is retained on the experiment host and
excluded by the repository's raw-row policy. Its digest is preserved below;
the tracked publication summary contains its paired reductions.

SHA-256 digests copied from the experiment host:

```text
39f3896e725cb5dbc3e28c6bfefd1f74f85a813cd0937146b70bf5fdd5ab3c18  summary.json
6d6da0196bc44a3c03d78c816dcad1056da820a1c4b367d5d28bc6f76c6f05ef  rows.jsonl
0d2860c842270c880f9c956e771e7aac2693f66e41e2c50c9d6dc415c8a6913d  region_layer_diagnostics.jsonl
19f079c36ade86d89bef0686beb09cb7eb32bddc616e11fa7ebdd348c58a4b6f  publication/region_layer_publication_summary.json
8b666e2ec1503f822a6ddd492c90c0ea49f221ba73240b010b3cdfdcb7045bf5  selection_cache.jsonl
```
