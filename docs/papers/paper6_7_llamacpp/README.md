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

## Native sequence-attachment probe

The pinned in-process C API supports a narrower E2 mechanism in unified-cache
mode. Apply `engine-patches/llamacpp/llamacpp-pra-native.patch`, copy the
`engine-patches/llamacpp/examples/pra-native` directory into the upstream
`examples` directory, and build `llama-pra-native`. The multi-case runner is:

```bash
python experiments/paper6_7_llamacpp/run_native_sequence_attach.py \
  --binary /path/to/llama.cpp/build/bin/llama-pra-native \
  --model /path/to/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  --output docs/papers/shared/results/paper6_7_llamacpp/native_sequence_attach.json
```

On the measured M4 Pro, schedule-matched split E0 and E2 were exactly
logit-equal through four decode steps in 10/10 CPU/Metal runs. HTTP lifecycle
wiring and arbitrary non-prefix positional geometry remain open.
