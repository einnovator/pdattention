# Paper 8.5: engine-independent agent memory

This package studies which logical mini-swe-agent records are sufficient for a
future action. It intentionally does not import an engine adapter or expose K/V
state.

The core context geometry is:

```text
SYSTEM + TASK
+ first H complete causal turns
+ selected middle turns
+ last T complete causal turns
```

`H` and `T` are independent. The middle selector may be none, recency,
lexical, hybrid, or a frozen offline-oracle score. Action--observation pairs are
atomic. Only an oversized tool-observation body can be compacted, and that
materialization decision is recorded separately from record selection.

The experiment ladder is:

1. Structural screening: validate recordization, budgets, head/tail overlap,
   mandatory overflow, and detail accounting on successful trajectories.
2. Frozen counterfactual replay: restore each pre-action workspace and compare
   selected-history next actions with the full-history reference.
3. Autonomous qualification: rerun only frontier policies from a fresh task
   sandbox and grade with the official SWE-bench harness.
4. Freeze `FULL`, `AGENT_BALANCED`, and `AGENT_AGGRESSIVE` logical plans for
   Paper 4.5 runtime realization.

Before positive selection, `dag.py` can construct a conservative resource/effect
DAG. `DAG_CERTIFIED` excludes only bundles backed by complete runtime identity
(cwd, environment, resource versions, complete output, and witnesses). Static
regex inference is diagnostic and fails closed. Even a certificate proves only
operational obsolescence under that abstraction; identical LLM behavior remains
an empirical frozen/autonomous outcome.

mini-swe-agent uses `MiniSweBashSemanticsProvider` because its only tool is
generic Bash. `HarnessMetadataSemanticsProvider` is the portable path: a future
agent SDK supplies generic effects and version metadata, while engines consume
only the resulting logical plan and never embed Bash- or application-specific
reasoning.

For instrumented mini-swe-agent runs, set:

```yaml
environment:
  environment_class: experiments.paper8_5_agent_memory.miniswe_environment.InstrumentedDockerEnvironment
  instrumentation_output_root: /absolute/host/result/checkpoints
  capture_workspace_checkpoints: true
```

The subclass leaves observation text unchanged and adds generic metadata under
the observation's `extra` object. Pre-action checkpoints contain the base HEAD,
separate binary index and worktree diffs, and non-ignored untracked files. This is an exact
repository-state snapshot under that declared scope; ignored build products and
external state are intentionally outside it.

Structural screening is not task-quality evidence. It may use the whitespace
counter only for code-path validation. Scientific rows must pass the exact
frozen model tokenizer through `--tokenizer`.

Example:

```bash
python -m experiments.paper8_5_agent_memory.run_structural_screen \
  --trajectory task1.traj.json task2.traj.json task3.traj.json \
  --tokenizer /path/to/frozen/tokenizer \
  --output structural_screen.json
```

Primary autonomous metrics are official resolution, calls to solution,
selected logical input tokens, avoidable reacquisition calls, repeated
verification, repeated failed edits, and first divergence. Latency, TTFT,
physical K/V copy, and re-encoding belong to Paper 4.5.

Frozen arms must be executed in order. First write a `full.json` artifact. Pass
that artifact through `--reference-replay full.json` for every later arm so
agreement is measured against a contemporaneous FULL generation rather than
only a historical trajectory. `DAG-EXCLUDE@100` supplies the per-decision
*materialized-token* ceilings for the recency control through `--policy
matched_token_tail --matched-budget-replay dag.json`. The control always keeps
the system prompt and original task, then admits a contiguous tail of complete
causal turns. If the oldest admitted turn crosses the ceiling, only an
oversized tool observation may be shortened: complete output lines are removed
from its head first, followed by a tokenizer-measured suffix fallback when one
line alone is too large. Assistant actions and small observations are never
truncated, and the action--observation role sequence stays valid. Every
serialized candidate is counted with the frozen tokenizer and the materialized
total is asserted not to exceed the decision's matched ceiling.

Artifacts report `selected_logical_tokens` (the unabridged contents of every
selected record) separately from `materialized_tokens` (the text actually sent
to the model), plus materialized retention and unused/overshoot counts. The
matched ceiling is read from the source artifact's `materialized_tokens`, not
its logical whole-record count.

This baseline has deliberate limitations. It is a recency control rather than
a semantic selector; it can compact only tool observations above the configured
threshold; it retains output envelopes and suffixes but does not understand
file/diff/test structure beyond line boundaries; and the final oversized-line
fallback searches Unicode character boundaries while checking the exact frozen
tokenizer count. If the immutable prompt or the minimum role-valid boundary
turn cannot fit, it fails closed or leaves the remaining budget unused. It does
not summarize, mutate assistant actions, or claim that truncated text is
behaviorally equivalent to the full observation.

Nominal retention treatments use the opposite boundary rule. Pass
`--round-up-whole-turns` for a 90% (or lower) treatment: ranked complete causal
turns are admitted until realized selected tokens meet or exceed the requested
floor. The artifact reports whole-turn overshoot separately. A run that retains
less than its nominal treatment fraction is not labeled as that treatment.

The replay request specifies only temperature zero by default, matching the
mini-swe-agent baseline; endpoint defaults remain unspecified. `--seed` and
`--max-output-tokens` are opt-in and must be used in both baseline and candidate
arms if a campaign freezes those parameters.

For human and AI policy inspection, export each task as adjacent complete JSON
and economical Markdown files:

```bash
python -m experiments.paper8_5_agent_memory.export_review_history \
  --trajectory task1.traj.json task2.traj.json \
  --output-directory docs/papers/shared/results/paper8_5_agent_memory/human_review
```

The Markdown view excerpts large content with explicit omission counts and a
full-content digest; the JSON retains every canonical record verbatim.
