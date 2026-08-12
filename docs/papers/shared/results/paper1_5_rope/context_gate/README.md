# RoPE Context Gate: Preregistered Expectations

The gate separates two variables that are often both called chunk size:

1. Encoding-context size: jointly encode 1, 2, or 4 adjacent references while
   evaluating the same five-way partition.
2. Materialized-unit size: evaluate 16-, 5-, and 2-way fixed-target partitions
   while encoding the complete historical source once.

The expected signatures are the same as the distance-study table. All headline
comparisons use matched seeds, oracle references first, and zero native-context
limit violations as an acceptance criterion.

## Frozen Result

Larger encoding groups materially help the small tier when all neighboring K/V
is retained: exact HotpotQA loss changes from 0.0235 at one reference per block
to 0.0026 at four; QASPER changes from 0.0936 to 0.0026. The evidence-only row
is invariant because evidence is causally first, so later references cannot
change its hidden state.

Larger materialized units generally help. Exact oracle loss rises as HotpotQA
is divided into 2, 5, and 16 units (tiny: 0.0607, 0.0655, 0.0775; small:
0.0026, 0.0030, 0.0040). QASPER follows the same broad direction with one
tiny-tier plateau.

Across 100 matched dataset-tier-seed-composition cells, fixed `D=64` and exact
split 50/50. Clipped `D=64` wins 68/100 but changes mean loss by only
-0.000093. Contextualization and composition dominate useful near-range
placement differences in this controlled setting. All native-limit violations
remain zero.

Run `python experiments/paper1_5_rope/summarize_retrieval_geometry.py` to
regenerate compact publication tables and both plots from the frozen rows.
