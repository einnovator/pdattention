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
model- and artifact-bound `pra-engine/1` receipt. The receipt must name the
pinned llama.cpp revision and prove native K/V, unified-cache sequence
attachment, metadata-only attachment, request cleanup, resource identity, and
isolation. The adapter never infers native PRA from Ollama's version or backend
history.

Reproduce the handshake, downgrade, model-switch, and unload controls with:

```bash
set PYTHONPATH=src
python experiments/paper6_8_ollama/run_backend_handshake.py
```

The live delegation cohort uses the model layer referenced by Ollama's manifest
as the patched llama-server's model path. After applying
`engine-patches/llamacpp/llamacpp-pra-server.patch`, launch llama-server with
`--parallel 4 --kv-unified --slots`, then run:

```bash
PYTHONPATH=src python experiments/paper6_8_ollama/run_native_delegation.py \
  --model qwen2.5:0.5b \
  --model-blob ~/.ollama/models/blobs/sha256-c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515 \
  --model-blob-sha256 c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515 \
  --ollama-manifest ~/.ollama/models/manifests/registry.ollama.ai/library/qwen2.5/0.5b \
  --output docs/papers/shared/results/paper6_8_ollama/native_delegation.json
python experiments/paper6_8_ollama/plot_native_delegation.py \
  docs/papers/shared/results/paper6_8_ollama/native_delegation.json \
  docs/papers/paper6_8_ollama/figures/native_delegation.png
```

The runner refuses a blob whose SHA-256 does not match both the command-line
digest and Ollama's model-layer manifest. The measured cohort delegates 25/25
requests to E2, preserves 25/25 semantic answers, and reproduces 25/25 warm
token sequences. Stock Ollama still exposes no public E2 endpoint, so ordinary
installations remain E0.
