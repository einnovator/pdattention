# Paper 4.5 agent benchmark

This campaign enforces the experiment order in the Paper 4.5 benchmark
contract: published result, ordinary no-PRA reproduction, official grading,
compatibility review, gateway pass-through, and only then PRA conditions.

The existing `experiments.agents` package remains the shared normalized result
and official-harness adapter layer. This package adds long-running campaign
state, external-result provenance, admission, and crash-safe reports.

```bash
python -m experiments.paper4_5_agent.run_campaign \
  --config experiments/paper4_5_agent/configs/campaigns/fim14b_r2egym.yaml \
  --max-hours 16 --resume
```

Use `--dry-run` to validate dependencies and create the report skeleton. A
non-baseline cell is marked `BLOCKED` until its exact no-PRA dependency is
`BASELINE_REPRODUCED`. A smoke subset, changed quantization, changed engine, or
unofficial grader can be useful diagnostics, but remains
`BASELINE_ATTEMPTED`.

Distributed hosts can return their normalized `official_result.json` without
replaying inference locally:

```bash
python -m experiments.paper4_5_agent.run_campaign \
  --config experiments/paper4_5_agent/configs/campaigns/fim7b_q4_calibration.yaml \
  --record-cell fim7b-q4-no-pra-smoke \
  --record-result path/to/official_result.json
```
