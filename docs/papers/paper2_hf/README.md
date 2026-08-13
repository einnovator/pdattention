# Paper 2: Hugging Face PRA Integration

This manuscript records the pretrained-model integration and productization phase. The current
evidence covers frozen Qwen3-0.6B and SmolLM2-135M Llama-family checkpoints, configurable PRA
consumption bands, eager attention, native GQA K/V, explicit reference memory, route-once
identity reuse, a bounded `#__head` prompt, stable router artifacts, and the public `pra_hf`
API. A thin Gemma 3 adapter and pinned official-checkpoint run preserve Gemma's periodic
local/global attention schedule. The official 1B checkpoint passes exact disabled parity,
five-seed routing, public API, bounded-memory, and causal-use smoke gates. Its global-capable
layers are 5, 11, 17, and 23; the measured routed path consumes memory at 17 and 23.
Serving-optimized kernels remain future work.

General source-position, RoPE rebinding, pooled-geometry, and attention-versus-semantic-routing
results belong to companion Paper 1.5 on `research/paper1-5-rope`. Paper 2 treats those results
as controlled motivation and reports their pretrained transfer, exact HF integration, sparse
routing, causal memory use, adaptation, and productization. Secondary Qwen geometry and routing
controls remain in the appendix and their original artifacts are unchanged.

Build from this directory:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

Reproduce the current artifacts from the repository root:

```powershell
python experiments/paper2_hf/qwen/run_first_night.py --device cuda
python experiments/paper2_hf/qa/run_smoke.py --device cuda
python -m experiments.paper2_hf.qa.run_multilayer_pra --device cuda
python -m experiments.paper2_hf.qa.run_multilayer_pra --device cuda --phase routed --router learned --schedules last_8,last_half
python -m experiments.paper2_hf.qa.run_multilayer_head --device cuda --router learned --schedule last_8
python -m experiments.paper2_hf.productize_router --help
python -m experiments.paper2_hf.run_product_demo --help
python -m experiments.paper2_hf.summarize_productization
python -m experiments.paper2_hf.gemma.run_gemma3_1b
python -m experiments.paper2_hf.gemma.summarize_gemma3_1b
python experiments/paper2_hf/qa/run_overnight_lora_sweep.py --device cuda
python experiments/paper2_hf/summarize_overnight_lora_sweep.py
python experiments/paper2_hf/build_behavioral_judge_package.py `
  --input docs/papers/shared/results/paper2_hf/last14_combo/last14_combo.json `
  --output-dir docs/papers/shared/results/paper2_hf/behavioral_judge `
  --seed 1234 --include-order-reversal --include-controls --batch-size 64
python experiments/paper2_hf/score_behavioral_judge_results.py `
  --truth docs/papers/shared/results/paper2_hf/behavioral_judge/behavioral_judge_truth.json `
  --responses `
    docs/papers/shared/results/paper2_hf/behavioral_judge/behavioral_judge_llm_results_chatgpt.json `
    docs/papers/shared/results/paper2_hf/behavioral_judge/judge_output_claude_sonnet5.json `
  --output docs/papers/shared/results/paper2_hf/behavioral_judge/behavioral_judge_results.json `
  --derived-output-dir docs/papers/shared/results/paper2_hf/behavioral_judge/derived
python -m experiments.paper2_hf.qa.run_qasper_diagnostic `
  --output-dir docs/papers/shared/results/paper2_hf/error_analysis
```

Structured results are under `docs/papers/shared/results/paper2_hf/`; release router artifacts
are under `artifacts/pra_hf/routers/`. The user-facing family workflow is
`pra-hf-demo/pra_hf_model_families.ipynb`. The behavioral judge directory contains the blind
package, private truth mapping, response schema, calibration controls, optional batches, two
complete judge responses, and their pair-collapsed aggregate report. Never send the truth mapping
to a judge.

The 32-token QASPER follow-up is under `shared/results/paper2_hf/error_analysis/`. It preserves
the frozen eight-token judge package and adds finish diagnostics, an auditable error taxonomy,
yes/no polarity margins, five-seed adapter controls, and a bounded routed-memory residual probe.
To rebuild only its tables and plots from the completed JSON without rerunning inference, use
`--refresh-existing docs/papers/shared/results/paper2_hf/error_analysis/generation_error_analysis.json`.
