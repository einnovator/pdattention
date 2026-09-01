# PRA-HF model-family demo

The demo directory contains two executed, user-facing notebooks:

- `pra_hf_model_families.ipynb` covers Qwen 3, Llama, and Gemma 3 model integration.
- `pra_runtime_productization.ipynb` is the comprehensive Paper 4.5 systems walkthrough. It
  covers runtime configuration, direct and external cold/warm/hot memory, exact native-K/V
  planning, four physical layouts, cache and profiler accounting, typed discovery and graph
  disclosure, lazy callable and skill records, compact typed tool results, selective replay,
  cursors, size-gated native result routing, lazy selected-region native encoding,
  lifecycle inspection, safe execution, and thin serving handoff.
  It also demonstrates durable user/session resolution, task DAGs, task-scoped records,
  reusable toolsets, cache-correct first-layer phase/shared selection, measured profile
  evidence boundaries, and the coding-agent CLI introduced by Paper 8.

Both import the library directly from this checkout's `src/` directory and run tiny offline
models by default.

The product notebook treats `BALANCED` as the correctness-preserving all-eligible-layer
profile. Fixed suffixes and the held-out noncontiguous calibration did not justify dropping
consumer layers, so reduced profiles remain `CALIBRATION_PENDING`. The corrected 30-example
M5 ladder reaches a `0.980x` concat point estimate and `1.031x` segmented ratio at 14B, but
the concat confidence interval includes parity. The M5 32B row is a capacity rejection, not
a measured latency point. Llama 3.1 8B independently preserves all
60 selected-text outputs through native concat, but remains an all-layer transport result rather
than evidence for a reduced consumer profile.

The model-family notebook answers whether PRA attaches correctly to Qwen, Llama, and Gemma. The
runtime notebook instead follows one Llama request through the mechanisms that turn logical PRA
selection into physical state and a deployable SDK boundary. Its opening table maps the differences
in detail.

Rebuild and execute it from this directory with:

```powershell
python build_demo_notebook.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 pra_hf_model_families.ipynb
python build_runtime_notebook.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 pra_runtime_productization.ipynb
```

Remote pretrained-model cells are opt-in and disabled by default.

The notebook constructs an agent SDK without downloading a chat model. For an interactive
pretrained session, use `pra agent chat MODEL -w . -t "..."`. The canonical CLI also
supports named agent profiles, model onboarding, runtime providers, bundles, and the
optional `pra agent start` web surface; `pra-hf` is retained only as a deprecated alias.
