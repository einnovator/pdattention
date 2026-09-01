# Paper 6.2: PRA-MLX

Measured artifacts are under
`docs/papers/shared/results/paper6_2_mlx/`. The serving smoke uses the MLX-LM
HTTP server; the rotating-cache study uses the in-process Python API.

```bash
python -m experiments.engine_serving.summarize
PYTHONPATH=src:. python -m experiments.paper4_5_runtime.run_storage_lifecycle
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
The expanded 149-unique-question confirmation is 2,086/2,086 exact and yields
ratios 1.043/1.139/1.133/1.072; the native wrapper remains semantically exact
but economically unoptimized.

The shared lifecycle manager now uses independently checksummed, mmap-readable
K/V segments per layer. Across 80 examples spanning three datasets at 0.6B and
QASPER at 1.7B, lossless WARM remains 80/80 exact and recovers after manager
restart. Explicit int8 COLD is exact in only 13/80 generated sequences and
remains outside the default profile.

Five-example Llama-3.2-1B and Gemma-3-1B lifecycle replications are each 5/5
lossless-WARM exact. Event-loop-owned promotion preserves all 35 outputs and,
with 500 ms lead, makes all five objects HOT before demand. The native HTTP
gateway now covers concurrency one through eight, streamed cancellation, and
session cleanup. These are serialized-runner queueing measurements, not a
claim of parallel model execution.

A separate physical-tier curve covers concurrency 1, 2, 4, 8, and 16 for
shared and independent resources. All 124 HOT/WARM requests remain
baseline-exact. At concurrency 16, shared WARM avoids 630 MiB of duplicate K/V
and sustains 2.91 requests/s; independent WARM performs 16 promotions and
sustains 1.22 requests/s.

The bounded-residency extension covers 1,020 natural-QA requests over five
seeds, three datasets, and compact-K/V budgets 1, 2, 4, and 8. QA F1 is stable
across budgets. Budgets below the eight-resource working set reload every
repeated access; budget 8 eliminates reloads and reduces mean resolution from
113--119 ms to 53--57 ms while raising peak compact residency to 151--168 MiB.
A separate 1,125-request long-session run uses three full rounds plus a final
revisit: capacities two and four average 17 reloads per seed, while capacity
eight eliminates reloads and evictions without changing F1 or answer
log-probability. Selective int8 on 20 QASPER examples improves exact generation
from 4/20 for all K/V to 14/20 for the late quarter, still below the lossless
product gate.

An independent Apple M4 Pro replication adds Qwen3-4B. Across 20 examples per
dataset and five seeds, ordinary split-cache and full-precision native K/V have
identical aggregate F1 on QASPER, HotpotQA, and 2Wiki at both 1.7B and 4B.
Shuffled memory degrades F1 and answer likelihood. In this in-process runner,
native completion takes 44.5--54.5% of ordinary split-cache time. A separate
1,500-request 4B session sweep visits eight resources three times. Budget 8
eliminates repeat reloads and lowers mean resolution by about 68% while leaving
the cold-load p95 and QA quality unchanged.

The M4 Pro cohorts can be regenerated with:

```bash
for dataset in qasper hotpotqa 2wikimultihopqa; do
  PYTHONPATH=src python -m experiments.paper6_2_mlx.run_answer_quality_pressure \
    --dataset "$dataset" --model mlx-community/Qwen3-4B-4bit \
    --output "docs/papers/shared/results/paper6_2_mlx/answer_quality_${dataset}_qwen3_4b_m4pro.json"
  PYTHONPATH=src python -m experiments.paper6_2_mlx.run_bounded_residency \
    --dataset "$dataset" --model mlx-community/Qwen3-4B-4bit \
    --resources-per-seed 8 --resident-resource-budgets 1 2 4 8 \
    --session-rounds 3 --max-new-tokens 12 \
    --output "docs/papers/shared/results/paper6_2_mlx/bounded_residency_${dataset}_qwen3_4b_m4pro.json"
done
```

Use `summarize_m4_scaling` and `summarize_m4_pressure` under
`experiments.paper6_2_mlx` to regenerate the manuscript tables, plots, and
machine-readable summaries.

The larger-model campaign uses pinned Qwen3 8B/14B/32B Q4 checkpoints and a
30B-A3B MoE control on the 48 GiB M4 Pro. It compares selected-text E0,
concatenated native E2, live segmented E2, and model-normalized contiguous
consumer suffixes. Raw rows are checkpointed beside each JSON output.

```bash
PYTHONPATH=src python experiments/mac_scaling/run_mlx_profile_scaling.py \
  --model mlx-community/Qwen3-8B-4bit \
  --revision 545dc4251c05440727734bcd94334791f6ab0192 \
  --dataset qasper --dataset hotpotqa --dataset 2wikimultihopqa \
  --output docs/papers/shared/results/mac_scaling/qwen3_8b_mlx_profiles.json
```

Model IDs, immutable revisions, seeds, profiles, and evidence tiers are in
`experiments/mac_scaling/campaign.json`. Run `summarize_mlx_scaling.py` after
all model points to regenerate the shared table and figure.

A final 1,000-request QASPER matrix crosses HOT/WARM resource budgets 2 and 8
with local rotating windows 64 and 256. Tight WARM capacity produces SOURCE
reconstruction as intended; F1 is identical in all cells, and retaining every
small object in disk-backed WARM does not improve resolve p95 on this host.

The routed natural-QA artifacts can be regenerated on Apple Silicon with:

```bash
for dataset in qasper hotpotqa 2wikimultihopqa; do
  PYTHONPATH=src:. python -m experiments.paper6_2_mlx.run_routed_answer_quality \
    --dataset "$dataset" --examples-per-seed 4 --route-top-k 4 \
    --output "docs/papers/shared/results/paper6_2_mlx/routed_answer_quality_${dataset}.json"
done
```

Run the live lifecycle probe with:

```bash
PYTHONPATH=src:. python -m experiments.paper6_2_mlx.run_live_storage_lifecycle
PYTHONPATH=src:. python -m experiments.engine_serving.summarize_live_storage_lifecycle
PYTHONPATH=src:. python -m experiments.engine_serving.summarize_mac_engine_extension
```
