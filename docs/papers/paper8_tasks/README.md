# Paper 8: Task-Aware PRA

This paper studies one physical agent session containing multiple interleaved
logical tasks. Task structure scopes candidates before ordinary PRA discovery;
Paper 7 continues to own compact views, exact backing, selective replay, and
native-K/V materialization.

## Reproduce

From the repository root:

```powershell
python experiments/paper8_tasks/run_task_scope_experiments.py
python -m pytest -q tests/test_task_context.py tests/test_task_scope.py `
  tests/test_session_service.py tests/test_task_planning.py `
  tests/test_task_workflows.py
cd docs/papers/paper8_tasks
latexmk -pdf -interaction=nonstopmode -halt-on-error paper8_tasks.tex
```

The experiment is deterministic and does not download or invoke a language
model. It evaluates the oracle task-scope mechanism over five seeds. The paper
does not present these rows as answer-generation accuracy.

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
- `experiments/paper8_tasks/run_task_scope_experiments.py`: tables, summaries,
  and figures.

Generated data lives in `docs/papers/shared/results/paper8_tasks/`; figures live
in `docs/papers/shared/figures/paper8_tasks/`. Rows for online and hybrid task
acquisition are explicit deferred-status artifacts, not synthetic measurements.
