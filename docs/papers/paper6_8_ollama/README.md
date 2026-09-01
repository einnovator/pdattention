# Paper 6.8 reproduction

The source audit pins Ollama commit `e37a00a8fa94fca07cefd544bffc9d8997ebcd44`.
The measured local daemon is Ollama `0.6.8`; source and measured runtime versions
are reported separately.

Run the natural E0 cohort through Ollama's OpenAI-compatible endpoint with
`experiments/engine_serving/run_openai_natural_e0.py`. For Qwen reasoning
models, pass `--protocol ollama-native --disable-native-thinking`; this uses
`/api/chat` and rejects the empty visible-content behavior observed through
the OpenAI stream. The M4 Pro Qwen3-14B artifact is under
`docs/papers/shared/results/paper6_8_ollama/mac_scaling/`. Run load, keep-alive,
model-switch, and unload qualification with:

```bash
set PYTHONPATH=src
python experiments/paper6_8_ollama/run_lifecycle.py
python experiments/paper6_8_ollama/plot_results.py
```

AUTO reports E0 unless an explicit backend executor returns a validated,
model-fingerprint-bound `pra-engine/1` receipt. The receipt must name the pinned
llama.cpp revision and prove native K/V, unified-cache sequence attachment,
metadata-only attachment, request cleanup, resource identity, and isolation.
The adapter never infers native PRA from Ollama's version or backend history.

Reproduce the handshake, downgrade, model-switch, and unload controls with:

```bash
set PYTHONPATH=src
python experiments/paper6_8_ollama/run_backend_handshake.py
```

The inherited llama.cpp mechanism result is model-backed; the Ollama handshake
cohort is a controlled protocol test. Stock Ollama still exposes no public E2
endpoint, so ordinary installations remain E0.
