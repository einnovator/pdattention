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
- CLI and executed notebook workflows.

Reproduce the measured portable profile:

```powershell
$env:PYTHONPATH = "src;."
python -m experiments.paper4_5_runtime.run_runtime_profile
python -m experiments.paper4_5_runtime.run_execution_policy_profile
python -m experiments.paper4_5_runtime.summarize_runtime
```

Build the paper:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

Try the SDK notebook under `pra-hf-demo/pra_runtime_productization.ipynb`.

The current runtime also integrates Paper 8's durable task/session layer. The
`pra-hf agent chat` command demonstrates the same SDK with task-scoped typed
records, reusable toolsets, local persistence, and per-call write authorization.

Launch the non-streaming reference gateway with `pra gateway serve`. G10 is an
explicit text-materialization fallback; it is not native-K/V PRA. SGLang and
FreeToken are E0 protocol targets only in this artifact.
