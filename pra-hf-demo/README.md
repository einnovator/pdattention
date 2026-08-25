# PRA-HF model-family demo

The demo directory contains two executed, user-facing notebooks:

- `pra_hf_model_families.ipynb` covers Qwen 3, Llama, and Gemma 3 model integration.
- `pra_runtime_productization.ipynb` covers the unified model, memory, typed-resource,
  authorization, materialization, inspection, and thin serving-backend workflow.

Both import the library directly from this checkout's `src/` directory and run tiny offline
models by default.

Rebuild and execute it from this directory with:

```powershell
python build_demo_notebook.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 pra_hf_model_families.ipynb
python build_runtime_notebook.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 pra_runtime_productization.ipynb
```

Remote pretrained-model cells are opt-in and disabled by default.
