# Paper 7 experiments

**Status:** EXPERIMENTALLY FROZEN - READY FOR EXTERNAL REVIEW

Paper 7 studies typed observations with compact visible views, exact backing
state, retrieval-only addresses, bounded materialization, and size-gated native
index construction. Result artifacts are frozen; the summarizer may regenerate
reader-facing figures and TeX macros without changing historical policy keys.

## Primary quality and active-context study

The calibrated held-out study uses Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca`. `PRA_NATIVE` matches the
reader-facing `FULL_BACKING` control while using fewer active K/V tokens.
`FULL_BACKING` is stored as historical policy key `FULL` in raw CSV files.

Primary artifacts live under:

```text
docs/papers/shared/results/paper7_records/full_pra_calibrated/
```

Regenerate plots and macros from the frozen rows with:

```powershell
python experiments/paper7_records/summarize_full_pra_calibrated.py
```

The original bounded runners are:

- `run_full_pra_reachability.py`: backing-address routing validation;
- `run_controller_calibration.py`: validation-only controller selection;
- `run_calibrated_adaptive.py`: held-out policy and oracle evaluation.

Do not rerun them merely for editorial changes.

## Native-index size gate

`run_native_index_size_gate.py` profiles `1K`, `4K`, `16K`, `64K`, and `256K`
token payloads over five seeds. It compares eager full-body native indexing,
size-gated cheap indexing, lazy selected-region native encoding, and
search/cursor-only handling.

The script uses an instrumented fixed-width Torch encoder through the
production PRA reference API. It measures lifecycle scaling, TTUC, state
transitions, and selected-region work. It is not a pretrained-model or Qwen
latency benchmark.

Artifacts live under:

```text
docs/papers/shared/results/paper7_records/native_index_size_gate/
```

## Historical mechanism studies

The following studies remain appendix evidence rather than the main quality
comparison:

- `run_progressive_context_iteration.py`: lexical progressive-context and
  context-control baseline;
- `run_latent_trigger_cursor_iteration.py`: latent-trigger, cursor, and
  action-conditioned diagnostics;
- `run_inception_experiments.py`: typed compression, backing, transport, and
  cursor mechanism checks.

## Verification

From the repository root:

```powershell
$env:PYTHONPATH = "src;."
python -m pytest -q tests/test_native_index_size_gate.py `
  tests/test_progressive_context.py tests/test_adaptive_context_runtime.py
cd docs/papers/paper7_records
latexmk -pdf -interaction=nonstopmode -halt-on-error `
  paper7_typed_adaptive_context_inception.tex
```

The complete release gate also runs the full test suite and visually inspects
every PDF page. Remaining SDK, remote-store, asynchronous-service, and serving
work belongs to later Paper 4.5, 5.5, and 6 workstreams rather than Paper 7.
