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
