# Paper 6.8 reproduction

The source audit pins Ollama commit `e37a00a8fa94fca07cefd544bffc9d8997ebcd44`.
The measured local daemon is Ollama `0.6.8`; source and measured runtime versions
are reported separately.

Run the natural E0 cohort through Ollama's OpenAI-compatible endpoint with
`experiments/engine_serving/run_openai_natural_e0.py`. Run load, keep-alive,
model-switch, and unload qualification with:

```bash
set PYTHONPATH=src
python experiments/paper6_8_ollama/run_lifecycle.py
python experiments/paper6_8_ollama/plot_results.py
```

AUTO reports E0 unless an explicit backend executor negotiates E2/E3. The
adapter never infers native PRA from Ollama's version or backend history.
