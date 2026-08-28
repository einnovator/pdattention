# PRA runtime productization

Paper 4.5 asks whether PRA's logical K/V sparsity becomes a measurable,
portable inference primitive before a serving engine is redesigned around it.

The implementation adds:

- one `PRARuntime` facade over the existing HF model API;
- four independent execution-policy axes with global/model/request precedence;
- request/per-layer and token/shared HF execution modes with logical plans;
- Paper 4 authenticated cold/warm/hot memory sessions;
- Paper 6.5 callable tool records, OpenAI/Anthropic skill folders, lazy
  selection/full-view encoding, typed discovery, and safe execution;
- Paper 7 type-aware result compaction with scoped exact backing, address
  search, selected replay, cursors, finite token/byte native-index gates,
  per-record-type overrides, auditable lifecycle states, and lazy selected-region
  native encoding above the full-index budget;
- versioned runtime configuration and inspection;
- deduplicated interval planning and native `[B, Hkv, T, D]` K/V packing;
- byte-bounded LRU accounting and request-stage profiling;
- eager and `torch.compile` gather gates;
- a scheduler-unaware vLLM handoff contract;
- a standalone G00/G10/G01/G11 gateway and E0/E1 capability adapters;
- DeepSeek Harness and Pi event/RPC bridges with tested ordinary-engine fallback;
- validated model-managed task operations, adaptive metadata widening, and frozen
  record-bounded native-consumption plans inherited from Paper 8;
- source-relative native positions with direct queries placed after the longest
  active record, named Paper 3/7/8 materialization profiles, and raw-versus-unique
  interval diagnostics;
- permanent visible-prefix/native-logit and prefill/decode lifetime regressions;
- portable HF-backed OpenAI-compatible SSE streaming with cooperative cancellation;
- tenant/user/session-scoped native-cache keys and per-tenant eviction limits;
- source-revision, position, materialization, and scope-safe physical payload reuse;
- pinned Qwen3-0.6B, Llama-3.2-1B mirror, and Gemma-3-1B cross-model gates;
- CLI and executed notebook workflows.

Reproduce the measured portable profile:

```powershell
$env:PYTHONPATH = "src;."
python -m experiments.paper4_5_runtime.run_runtime_profile
python -m experiments.paper4_5_runtime.run_execution_policy_profile
python -m experiments.paper4_5_runtime.run_agent_plugin_contracts
python -m experiments.paper4_5_runtime.run_cross_model_validation --model all --device cuda
python experiments/paper4_5_runtime/run_layer_profile_calibration.py --model qwen --device cuda
python experiments/paper4_5_runtime/run_layer_profile_calibration.py --model llama --device cuda
python experiments/paper4_5_runtime/run_layer_profile_calibration.py --model gemma --device cuda
python experiments/paper4_5_runtime/run_layer_profile_calibration.py --finalize-only
python -m experiments.paper4_5_runtime.summarize_runtime
```

The cross-model command is restartable with `--model qwen`, `--model llama`,
`--model gemma`, and `--model finalize`. The official Meta Llama checkpoint was
access-blocked during this run, so the checked-in Llama row names the exact
public weight mirror. Gemma is reported as partial topology: native global-layer
mechanics pass, while unchanged local sliding layers prevent full-prefix
equivalence.

Build the paper:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

Try the SDK notebook under `pra-hf-demo/pra_runtime_productization.ipynb`.

The current runtime also integrates Paper 8's durable task/session layer. The
`pra-hf agent chat` command demonstrates the same SDK with task-scoped typed
records, reusable toolsets, local persistence, and per-call write authorization.

Launch the reference gateway with `pra gateway serve`. HF-backed adapters expose
OpenAI-compatible streaming; request-owned references remain active until decode
or cancellation cleanup completes. G10 is an
explicit text-materialization fallback; it is not native-K/V PRA. SGLang and
FreeToken are E0 protocol targets only in this artifact.
