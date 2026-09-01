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
logit-equal through four decode steps in 10/10 CPU/Metal runs.

## Native llama-server sequence attachment

Apply `engine-patches/llamacpp/llamacpp-pra-server.patch` to the same pinned
upstream revision, then build and launch:

```bash
cmake --build build --target llama-server -j
build/bin/llama-server -m model.gguf --parallel 4 --kv-unified -c 4096 \
  -ngl 99 --slots --metrics
```

The patched server advertises `GET /pra/capabilities`. Encode a selected
resource into an explicit slot with `pra_pin_resource=true`; query another slot
with `pra_resource_slot=<source>`. Release it with
`DELETE /pra/resources/<source>`. Reproduce the matched server cohort with:

```bash
python experiments/paper6_7_llamacpp/run_native_server_attach.py \
  --base-url http://127.0.0.1:8080 \
  --output docs/papers/shared/results/paper6_7_llamacpp/native_server_attach.json
```

The measured run produced exact E0/E2 output in 25/25 cases, exact warm reuse
in 25/25, correct absent-memory separation in 25/25, and exact shared-resource
concurrency in 15/15 requests. Arbitrary non-prefix positional geometry and
scheduler-owned slot allocation remain open.
