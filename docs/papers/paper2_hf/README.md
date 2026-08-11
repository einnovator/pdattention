# Paper 2: Hugging Face PRA Integration

This manuscript records the pretrained-model integration phase. The current evidence covers
one frozen `Qwen/Qwen3-0.6B` checkpoint, configurable PRA consumption bands, eager attention,
native GQA K/V, explicit reference memory, route-once identity reuse, and a bounded `#__head`
prompt. Llama and Gemma are future phases.

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
```

Structured results are under `docs/papers/shared/results/paper2_hf/`.
