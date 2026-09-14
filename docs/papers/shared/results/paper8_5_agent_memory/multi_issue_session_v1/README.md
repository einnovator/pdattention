# Persistent multi-issue session pilot

This artifact locks the same three successful real mini-swe-agent trajectories
under two session schedules:

- `fresh_per_issue`: three logical sessions, one issue per session;
- `persistent`: one logical session containing the same ordered issues, with
  explicit issue and workspace boundaries.

The issues are Django 15277, pytest 7982, and scikit-learn 13135. The composer
keeps one system message, assigns globally unique record/turn/causal-group IDs,
normalizes mini-swe-agent's terminal `exit` observation to a role-valid user
observation, and identifies the newest visible task as the active retrieval
query. Incompatible SWE-bench repository snapshots are not merged.

## Tokenizer-exact structural opportunity

These are Qwen3-Coder tokenizer counts over historical successful trajectories.
They are selection-opportunity measurements, not next-action agreement, task
accuracy, K/V reuse, or latency results.

| Schedule/policy | Issues | Final prompt tokens selected/full | Final saving | Cumulative selected/full | Cumulative saving |
|---|---:|---:|---:|---:|---:|
| Persistent FULL | 1 | 9,645 / 9,645 | 0.00% | 164,726 / 164,726 | 0.00% |
| Completed-episode spine | 2 | 14,681 / 21,989 | 33.23% | 380,891 / 534,359 | 28.72% |
| Completed-episode spine | 3 | 15,381 / 33,431 | 53.99% | 632,632 / 1,147,100 | 44.85% |
| Active episode only | 2 | 12,237 / 21,989 | 44.35% | 329,567 / 534,359 | 38.32% |
| Active episode only | 3 | 11,206 / 33,431 | 66.48% | 497,808 / 1,147,100 | 56.60% |
| Fresh FULL | 3 | 11,172 current issue | n/a | 496,540 total | n/a |

The aggressive active-episode arm is 1,268 cumulative tokens (0.26%) larger
than deliberately starting three fresh sessions. It is therefore an automatic
persistent-session hygiene mechanism, not a saving over deliberate reset. A
genuine cross-issue PRA advantage must appear in related same-repository or
dependent same-workspace sequences as preserved quality, reduced rediscovery,
or lower cost per resolved issue. It does not follow from retaining one session
identity. FULL persistent replay, policy replay, and autonomous official
grading remain required.

The registered 1--5 issue campaign is defined by
`experiments/paper8_5_agent_memory/configs/multi_issue_strategy_registry_v1.json`.
Its strict reducer and curve generator are `multi_issue_frontier.py` and
`multi_issue_curves.py`. They require fresh FULL and persistent FULL controls
for every matched sequence and never pool sequence strata.

## Required quality sequence

1. Replay FULL persistent history on the active issue and measure divergence
   from fresh FULL; this isolates cross-issue interference.
2. Replay completed-episode spine and active-episode-only against the same
   contemporaneous FULL persistent reference.
3. Run autonomous 2-, 3-, 4-, and 5-issue schedules only after frozen gates.
4. Cluster uncertainty by ordered issue sequence, not by model call.
5. Transfer the exact schedule and policy IDs to Paper 4.5, where native K/V
   residence, selected K/V, copies, re-encoding, prefix-cache hits, and wall
   time are measured separately.

The first frozen gate is reproducible with
`experiments/paper8_5_agent_memory/run_multi_issue_frozen_gate.sh`. Its default
executes the first active-issue decision in each 2- and 3-issue session; raise
`PAPER85_MAX_DECISIONS` to cover the complete active episode after the smoke
gate passes.
