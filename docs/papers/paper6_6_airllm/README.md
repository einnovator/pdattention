# Paper 6.6: PRA on AirLLM

This companion paper studies independent model-weight and semantic-context
streaming. AirLLM owns layer/expert weight movement; the shared PRA runtime owns
records, routing, selection, authorization, and selected native K/V.

## Build

```bash
cd docs/papers/paper6_6_airllm
latexmk -pdf -interaction=nonstopmode -halt-on-error paper6_6_airllm.tex
```

## Reproduce

```bash
python experiments/paper6_6_airllm/audit_airllm_environment.py \
  --airllm-source /path/to/airllm --device cuda --hardware "GPU" \
  --output docs/papers/shared/results/paper6_6_airllm/airllm_capability_audit.json

PYTHONPATH=src python -m experiments.paper6_6_airllm.run_layer_streaming_benchmark \
  --output docs/papers/shared/results/paper6_6_airllm/layer_streaming_controlled.json

python -m experiments.paper6_6_airllm.plot_results \
  --input docs/papers/shared/results/paper6_6_airllm/layer_streaming_controlled.json \
  --output-dir docs/papers/paper6_6_airllm/figures \
  --summary docs/papers/shared/results/paper6_6_airllm/layer_streaming_summary.json
```

`run_mlx_e0.py` is the live Apple/MLX selected-text baseline. It is E0 by
design. `run_cuda_native.py` exercises the HF-backed AirLLM path and writes a
checkpointed JSON report even when a stop gate fails.

The selector-frozen natural-QA runner records directly measured TTFT, ITL,
completion time, token counts, and CUDA allocation for E0 selected text and E2
native K/V. Keep short-output timing diagnostics separate from larger quality
cohorts when interpreting their generated tables:

```bash
PYTHONPATH=src python -m experiments.paper6_6_airllm.run_cuda_natural \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --shard-dir /path/to/airllm-tinyllama-shards \
  --max-examples-per-dataset 5 --warm-repeats 1 --max-new-tokens 4 \
  --output docs/papers/shared/results/paper6_6_airllm/tinyllama_timed_15.json

PYTHONPATH=src python -m experiments.paper6_6_airllm.summarize_cuda_natural \
  docs/papers/shared/results/paper6_6_airllm/tinyllama_timed_15.json \
  --output docs/papers/shared/results/paper6_6_airllm/tinyllama_timed_15_summary.json \
  --table docs/papers/shared/results/paper6_6_airllm/generated_timed_15_quality_table.tex \
  --timing-table docs/papers/shared/results/paper6_6_airllm/generated_timed_15_timing_table.tex
```
