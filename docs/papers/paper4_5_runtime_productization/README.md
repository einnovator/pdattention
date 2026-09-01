# PRA runtime productization

Paper 4.5 asks whether PRA's logical K/V sparsity becomes a measurable,
portable inference primitive before a serving engine is redesigned around it.

The implementation adds:

- one `PRARuntime` facade over the existing HF model API;
- four independent execution-policy axes with global/model/request precedence;
- request/per-layer, token/shared, and cache-correct phase/shared HF execution
  modes with logical plans; phase/shared uses a cache-free routing probe and
  requires routing at the first active PRA layer;
- Paper 4 authenticated cold/warm/hot memory sessions;
- Paper 6.5 callable tool records, OpenAI/Anthropic skill folders, lazy
  selection/full-view encoding, typed discovery, and safe execution;
- Paper 7 type-aware result compaction with scoped exact backing, address
  search, selected replay, cursors, finite token/byte native-index gates,
  per-record-type overrides, auditable lifecycle states, and lazy selected-region
  native encoding above the full-index budget;
- versioned runtime configuration and inspection;
- a product-profile registry that exposes three-case quality optima only as
  `QUALITY_MAX_CANDIDATE` with `SMOKE / CALIBRATION_PENDING`, reserving
  `QUALITY_MAX` for workload-scale validation;
- deduplicated interval planning and native `[B, Hkv, T, D]` K/V packing;
- byte-bounded LRU accounting and request-stage profiling;
- eager and `torch.compile` gather gates;
- a scheduler-unaware vLLM handoff contract;
- a standalone G00/G10/G01/G11 gateway and E0/E1/E2 capability adapters;
- engine-type-aware cache/session profiles independent of PRA integration depth;
- prepared gateway sessions, explicit FULL/DELTA/AUTO history, resource
  ADD/UPDATE/REMOVE/UNCHANGED operations, stable cache-affinity hints, and
  non-sensitive session inspection/close endpoints;
- prefix-preserving G10 placement that keeps the prior serialized conversation
  byte-identical instead of changing a message-zero evidence block;
- DeepSeek Harness and Pi event/RPC bridges with tested ordinary-engine fallback;
- validated model-managed task operations, adaptive metadata widening, and frozen
  record-bounded native-consumption plans inherited from Paper 8;
- source-relative native positions with direct queries placed after the longest
  active record, named Paper 3/7/8 materialization profiles, and raw-versus-unique
  interval diagnostics;
- permanent visible-prefix/native-logit and prefill/decode lifetime regressions;
- portable HF-backed OpenAI-compatible SSE streaming with cooperative cancellation;
- tenant/user/session-scoped native-cache keys and per-tenant eviction limits;
- atomic tenant/scope revalidation during both storage promotion and request pinning;
- source-revision, position, materialization, and scope-safe physical payload reuse;
- pinned Qwen3-0.6B, Llama-3.2-1B mirror, and Gemma-3-1B cross-model gates;
- the canonical `pra` model-onboarding, profile, bundle, runtime-provider,
  agent, gateway, and Hub command tree, with `pra-hf` retained as a deprecated
  alias;
- versioned named agent profiles and an optional FastAPI/WebSocket multi-session UI;
- CLI and executed notebook workflows;
- a selector-frozen E0 selected-text versus E2 native-K/V benchmark spanning
  cold, warm, multi-query, and concurrency-eight schedules on MLX-LM,
  SGLang-MLX, and vLLM-Metal, with disjoint quality, input, PRA, ingestion,
  serving, and reuse metrics.
- an expanded 149-unique-question confirmation per engine: 6,258/6,258 exact
  E0/E2 pairs overall, with engine-specific cost rather than a blanket speedup;
- an engine-neutral `HOT/WARM/COLD/SOURCE` storage manager with named profiles,
  strict fingerprints, lossless WARM/COLD stores, independent compression and
  int8 policy, deterministic weighted eviction, typed-record priors, task and
  dependency retention, delayed closure compaction, and session cleanup.
- live manager bridges for vLLM pages, SGLang HiCache-backed arrays, MLX arrays,
  segmented mmap WARM storage, durable restart recovery, and persistent
  lifecycle metrics. The expanded lossless WARM study is exact on 265/265
  engine/model/dataset pairs across Qwen, Llama, and Gemma; int8 COLD is exact
  on only 61/265 and remains opt-in. The pinned SGLang-MLX build cannot load
  Gemma's per-layer sliding-window topology.
- event-loop-owned, deduplicated WARM promotion with hot-set admission;
- online SGLang/MLX streaming, cancellation, cleanup, and queueing curves;
- shared- versus independent-resource concurrency through 16 sessions;
- selective K/V int8 calibration and sustained multi-query pressure over
  1,125 natural-QA requests.
- a pinned M4 Pro Qwen3 8B/14B/32B and 30B-A3B profile campaign with exact
  concatenated E0/E2 sequence parity, a live segmented-attention candidate,
  and model-normalized consumer-layer calibration over 60 natural-QA
  model--example pairs.
- corrected M4/M5 MLX model-scaling evidence imported into the product matrix:
  `BALANCED` consumes native memory at all eligible layers, while segmented and
  reduced-layer candidates remain `CALIBRATION_PENDING`.

Reproduce the measured portable profile:

```powershell
$env:PYTHONPATH = "src;."
python -m experiments.paper4_5_runtime.run_runtime_profile
python -m experiments.paper4_5_runtime.run_execution_policy_profile
python -m experiments.paper4_5_runtime.run_agent_plugin_contracts
python -m experiments.paper4_5_runtime.run_gateway_session_profile
python -m experiments.paper4_5_runtime.run_storage_lifecycle
python -m experiments.paper6_vllm.run_live_storage_lifecycle
python -m experiments.paper6_1_sglang.run_live_storage_lifecycle
python -m experiments.paper6_2_mlx.run_live_storage_lifecycle
python -m experiments.engine_serving.summarize_live_storage_lifecycle
python -m experiments.engine_serving.summarize_mac_engine_extension
python -m experiments.paper4_5_runtime.run_cross_model_validation --model all --device cuda
python experiments/paper4_5_runtime/run_layer_profile_calibration.py --model qwen --device cuda
python experiments/paper4_5_runtime/run_layer_profile_calibration.py --model llama --device cuda
python experiments/paper4_5_runtime/run_layer_profile_calibration.py --model gemma --device cuda
python experiments/paper4_5_runtime/run_layer_profile_calibration.py --finalize-only
python -m experiments.paper4_5_runtime.build_product_matrix_v2
python -m experiments.paper4_5_runtime.build_engine_qualification
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
`pra agent chat` command demonstrates the same SDK with task-scoped typed
records, reusable toolsets, local persistence, and per-call write authorization.

Launch the reference gateway with `pra gateway serve`. HF-backed adapters expose
OpenAI-compatible streaming; request-owned references remain active until decode
or cancellation cleanup completes. G10 is an
explicit text-materialization fallback; it is not native-K/V PRA. FreeToken is
an E0 protocol target in this artifact. Companion Papers 6.1 and 6.2 measure
SGLang-MLX and MLX-LM E2 mechanisms separately; those runs do not change Paper
4.5's HF-centered evidence tier. The gateway experiment measures
exact logical-prefix stability and transport bytes with a simulated
adapter. It does not report a physical engine cache hit, scheduler affinity,
or remote-engine speedup. `pra runtime serve MODEL -e ENGINE` is now the common
launch path. vLLM remains conservatively E0; companion SGLang and MLX providers
advertise measured E2 mechanism support without importing those papers' metrics
into Paper 4.5.
