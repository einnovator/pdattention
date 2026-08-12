# Paper 2: Hugging Face PRA Integration

This manuscript records the pretrained-model integration and productization phase. The current
evidence covers frozen Qwen3-0.6B and SmolLM2-135M Llama-family checkpoints, configurable PRA
consumption bands, eager attention, native GQA K/V, explicit reference memory, route-once
identity reuse, a bounded `#__head` prompt, stable router artifacts, and the public `pra_hf`
API. A thin Gemma 3 adapter and pinned official-checkpoint run preserve Gemma's periodic
local/global attention schedule. The official 1B checkpoint passes exact disabled parity,
five-seed routing, public API, bounded-memory, and causal-use smoke gates. Serving-optimized
kernels remain future work.

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
```

Structured results are under `docs/papers/shared/results/paper2_hf/`; release router artifacts
are under `artifacts/pra_hf/routers/`.
