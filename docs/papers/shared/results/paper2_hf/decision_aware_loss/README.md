# Decision-Aware QASPER Loss

This final Paper 2 experiment tests whether explicit binary-decision weighting
can preserve routed-QASPER answer realization while improving yes/no polarity.
It changes only the loss applied to the existing residual-16 adapter:

```text
mean sequence CE + lambda * grouped yes/no CE
```

The grouped term pools all four tokenizer-native single-token forms of each
polarity. Model, router, last-14 placement, residual width, routed references,
memory budget, optimizer, learning rate, 32 updates, generation length, and
five seeds remain fixed.

## Protocol

- Train: 12 QASPER identities, 7 yes / 5 no; majority baseline 58.3%.
- Validation: 4 disjoint identities, 1 yes / 3 no; majority baseline 75.0%.
- Test: 8 untouched identities, 5 yes / 3 no; majority baseline 62.5%.
- Lambda sweep: 0, 0.25, 0.5, 1, 2 across seeds 11, 23, 37, 53, and 71.
- Validation selects lambda by decoded polarity, then margin accuracy, F1, EOS,
  containment, mean margin, and lower lambda.
- Test evaluates only lambda 0 and the validation-selected lambda 2.

## Result

Lambda 2 raised validation polarity from 25% to 40%, but did not generalize.
On test, sequence-only training reached 55.0% +/- 11.2% polarity, F1 .511,
containment .550, EOS 1.000, sequence logP -.961, and mean polarity margin
+.143. Lambda 2 reached 40.0% +/- 16.3% polarity, F1 .361, containment .400,
EOS 1.000, sequence logP -1.286, and margin -.390. Both remain below the 62.5%
test majority baseline; the existing oracle-trained residual-16 remains the
stronger balanced condition at 72.5% +/- 5.6% polarity.

For lambda 2, margin accuracy is 37.5% with no memory, 40.0% with routed memory,
and 45.0% with oracle memory. Evidence quality still moves the decision in the
expected aggregate direction, but the auxiliary loss does not solve the small-
cohort generalization problem.

All 40 lambda-zero seed/item rows exactly reproduce the prior routed-QASPER
baseline across generation and probability metrics. All 40 selected-adapter
no-memory comparisons are exact, and the selected adapter changes WikiText-2
loss by 0.0000 for every seed.

The stop rule applies. No larger loss search, adapter, router, or test-guided
retuning follows this result.

`decision_aware_loss.json` is canonical. The CSV files expose validation,
per-seed test, five-seed aggregates, and WikiText controls. The PDF and PNG plot
the validation sweep and the untouched test comparison.
