# Paper 8.5 synthetic mechanism diagnostics

These tasks isolate rule activation and observation materialization. They are
structural diagnostics, not model-quality, SWE-bench, or runtime evidence.

| Task | Probe | Mechanism gate |
|---|---:|---:|
| `synthetic_discovery_consumed` | 7 | PASS |
| `synthetic_versioned_state_convergence` | 6 | PASS |
| `synthetic_structured_failure_evidence` | 2 | PASS |

The structured-evidence materializer uses current task/query terms, command
semantics, paths, tracebacks, failure lines, diff headers/hunks, and source
symbols. If it finds no positive evidence, it retains the entire observation
rather than falling back to arbitrary head/tail sampling.

Each task directory contains a complete mini-swe-agent-shaped trajectory that
can be passed to `run_frozen_replay.py` for model-facing qualification.
The separately reduced model-facing smoke is documented in `MODEL_SMOKE.md`.
