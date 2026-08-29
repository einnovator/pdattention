# Paper 6.2: PRA-MLX

Measured artifacts are under
`docs/papers/shared/results/paper6_2_mlx/`. The serving smoke uses the MLX-LM
HTTP server; the rotating-cache study uses the in-process Python API.

```bash
python -m experiments.engine_serving.summarize
cd docs/papers/paper6_2_mlx
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

The paper now includes an in-process native selected-K/V executor, exact
split-cache parity on Qwen, Llama, and Gemma, and 5/5 rotating-local recovery on
Qwen and Llama. Gemma's ordinary and native answer-format controls both score
0/5 despite exact logits, so that row is mechanism parity rather than quality.
Five-seed layer-profile, persistence, and concurrency sweeps are also included,
along with 40-example-per-dataset QASPER and HotpotQA natural-text transport
controls. Those controls test source-dependent native transport, not end-task QA.
The original-answer extension now also routes four candidate documents or
QASPER paragraphs with the SDK hybrid index. Across QASPER, HotpotQA, and
2Wiki, routed ordinary and native execution agree on all 60 paired examples;
the remaining oracle gap is evidence discovery rather than MLX consumption.
A matched four-regime E0/E2 benchmark preserves all 840 outputs and removes
90.6--93.0% of visible prompt tokens. Cold, warm, multi-query, and concurrent
E2/E0 cost ratios are 1.130, 1.155, 1.148, and 1.132, exposing the remaining
unfused native-decode cost despite cheaper native ingestion.

The bounded-residency extension covers 1,020 natural-QA requests over five
seeds, three datasets, and compact-K/V budgets 1, 2, 4, and 8. QA F1 is stable
across budgets. Budgets below the eight-resource working set reload every
repeated access; budget 8 eliminates reloads and reduces mean resolution from
113--119 ms to 53--57 ms while raising peak compact residency to 151--168 MiB.

The routed natural-QA artifacts can be regenerated on Apple Silicon with:

```bash
for dataset in qasper hotpotqa 2wikimultihopqa; do
  PYTHONPATH=src:. python -m experiments.paper6_2_mlx.run_routed_answer_quality \
    --dataset "$dataset" --examples-per-seed 4 --route-top-k 4 \
    --output "docs/papers/shared/results/paper6_2_mlx/routed_answer_quality_${dataset}.json"
done
```
