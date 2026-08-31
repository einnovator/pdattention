# Paper 6.7 reproduction

Pinned upstream: llama.cpp `458681e1d5d4a29a1463c4732e03226cf384b997`.

The measured E0 qualification uses the shared frozen-selection manifest:

```bash
python experiments/engine_serving/run_openai_natural_e0.py \
  --base-url http://127.0.0.1:18080 \
  --engine llama_cpp_metal --model qwen2.5-0.5b \
  --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
  --output docs/papers/shared/results/paper6_7_llamacpp/metal_natural_e0.json \
  --max-examples-per-dataset 5 --repeats 2 \
  --max-new-tokens 16 --max-full-tokens 1024
```

CPU is run with `-ngl 0`; Metal is run with `-ngl 99`. Generate the paper
figure with `python experiments/paper6_7_llamacpp/plot_results.py`.

Upstream slot save/restore requires `--slot-save-path`. Slot checkpoints are
ordinary sequence-positioned state and are deliberately not classified as E2.
