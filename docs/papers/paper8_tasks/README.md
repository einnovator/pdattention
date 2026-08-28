# Paper 8: Task-Aware PRA

**Status:** EXPERIMENTALLY FROZEN / READY FOR EXTERNAL REVIEW

**Branch:** `research/paper8-tasks`

**Frozen manuscript source:** recorded after the editorial-freeze commit below.

This paper studies one physical agent session containing multiple interleaved
logical tasks. Task structure scopes candidates before ordinary PRA discovery;
Paper 7 continues to own compact views, exact backing, selective replay, and
native-K/V materialization.

The causal hierarchy is intentional:

1. Oracle task identity and relations isolate the task-scope mechanism.
2. A frozen production native-Q/K router confirms evidence allocation and
   context economy over 15 typed-record cases.
3. Validated acquisition experiments test explicit-structure transcription.
4. The native-consumption ablation is a secondary integration diagnostic.

Primary production results are 33.3% to 93.3% complete evidence availability,
75.0% to 0.0% cross-task contamination, 234.3 to 93.5 requested native tokens,
and 13.3% to 66.7% routed-visible answer accuracy for session versus structural
scope. These bounded results do not establish latent task planning, sparse
native-memory integration, or serving-scale efficiency.

## Reproduce

From the repository root:

```powershell
python experiments/paper8_tasks/run_task_scope_experiments.py
python experiments/paper8_tasks/run_production_pra.py --phase postprocess
python -m experiments.paper8_tasks.run_native_consumption_geometry --postprocess-only
python -m experiments.paper8_tasks.run_task_management_roadmap --phase postprocess
python -m pytest -q tests/test_task_context.py tests/test_task_scope.py `
  tests/test_session_service.py tests/test_task_planning.py `
  tests/test_task_workflows.py tests/test_task_production_cases.py `
  tests/test_paper8_production_artifacts.py
cd docs/papers/paper8_tasks
latexmk -pdf -interaction=nonstopmode -halt-on-error paper8_tasks.tex
```

The original scope experiment is deterministic and model-free. The production
iteration additionally evaluates frozen `Qwen/Qwen3-0.6B` on CUDA through the
Paper 7 native routing and materialization path at revision
`c1899de289a04d12100db370d81485cdf75e47ca`. Its JSONL checkpoints allow a
crashed run to resume. The native-geometry diagnostic replays frozen routing
through record-bounded width and consumption-layer sweeps. The final task
management iteration evaluates metadata corruption, adaptive widening, JSON and
Markdown preflight, validated online task operations, and hybrid mutation.
The primary causal study still uses oracle task identity and relations; model
acquisition prompts expose identifiers and dependencies explicitly.

## Implementation Map

- `src/pra_hf/task_context.py`: versioned task state, events, DAG validation,
  replay, structural closure, and typed record provenance.
- `src/pra_hf/task_scope.py`: task-local, structural, adaptive, and session
  admission plus cold/warm/hot working-set accounting.
- `src/pra_hf/session_service.py`: abstract, in-memory, and atomic local-disk
  session services resolved by user and session identity.
- `src/pra_hf/task_planning.py`: deterministic complexity gate and validated
  JSON/Markdown preflight parsers.
- `src/data/task_workflows.py`: controlled workflow and interleaving generator.
- `src/data/task_production_cases.py`: typed production cases, confusability,
  join-capacity cases, and independent DAG geometries.
- `experiments/paper8_tasks/run_task_scope_experiments.py`: tables, summaries,
  and figures.
- `experiments/paper8_tasks/run_production_pra.py`: Paper 7 native routing,
  materialization, model-consumption decomposition, accounting, and plots.
- `src/pra_hf/native_geometry.py`: frozen routing anchors, record-bounded
  interval unions, exact layer-native K/V slicing, and semantic coverage.
- `experiments/paper8_tasks/run_native_consumption_geometry.py`: restartable
  width/depth diagnostic and visible selected-record ceiling.
- `experiments/paper8_tasks/run_task_management_roadmap.py`: metadata
  robustness plus model-generated preflight, online, and hybrid acquisition.

Generated data lives in `docs/papers/shared/results/paper8_tasks/`; figures live
in `docs/papers/shared/figures/paper8_tasks/`. The frozen PDF is
`docs/papers/paper8_tasks/paper8_tasks.pdf`.
