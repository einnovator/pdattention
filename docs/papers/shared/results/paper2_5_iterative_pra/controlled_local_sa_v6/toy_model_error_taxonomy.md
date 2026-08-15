# Toy-model error taxonomy

| Failure class | Representative paired identity | Machine-readable evidence |
|---|---|---|
| Missed path/root | `chain-100004-d1` / w16 / seed 17 | selected condition with zero evidence recall |
| Complete oracle evidence but wrong label | `chain-100004-d1` / w16 / seed 17 | complete path = 1 and final correct = 0 |
| Positive immediate margin effect erased later | `chain-833107-d1` / w16 / seed 17 | `erased_by_final_layer = 1` |

The complete taxonomy also includes selected distractor memory, weak evidence
attention, non-positive answer-direction alignment, and intervention divergence.
Rows are joined by `example_id`, `seed`, `window`, and `condition` across the
mechanistic CSV files; the examples above are illustrative audit pointers, not
independent statistical units.
