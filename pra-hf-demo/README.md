# PRA-HF model-family demo

`pra_hf_model_families.ipynb` is an executed, user-facing walkthrough of PRA-HF with Qwen 3,
Llama, and Gemma 3 Hugging Face model classes. It imports the library directly from this
checkout's `src/` directory and runs tiny offline models by default.

Rebuild and execute it from this directory with:

```powershell
python build_demo_notebook.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 pra_hf_model_families.ipynb
```

Remote pretrained-model cells are opt-in and disabled by default.
