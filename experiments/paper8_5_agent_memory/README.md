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

`negative_selection.py` adds four separately switchable, causal-group-safe
heuristic families. They are deliberately not promoted to `DAG_CERTIFIED`:

- H1 retires a discovery search only after one of its returned resources is
  consumed by a concrete read and the trajectory subsequently moves to a
  non-read operation; `Kf` delays retirement from that transition.
- H2a retires an actual changed-file write only after a later read/diff exposes
  the same post-write resource version. H2b requires a successful verification
  with an explicit resource dependency. The bare "not rewritten" rule is
  available only as `h2_bare_aggressive` and `Kw` delays it.
- H3 retains the latest `Kr` reads that cover each `(resource, version, span)`;
  disjoint source spans do not supersede one another.
- H4 keeps the latest `Kx` distinct read resources and pins task, mutation,
  failure, and declared-dependency resources. It is a capacity heuristic, not
  a safety rule.

Every retired causal group carries an inactive controller tombstone with rule,
resource, and witness IDs. Tombstones are persisted in the plan/artifact but
are not inserted into the model prompt, because doing that would create a
separate summarization treatment. A predeclared follow-up materialization arm
will compare this binary retirement with a model-visible compact provenance
stub; stub tokens will count against its realized budget. Frozen replay reports immediate reacquisition
of excluded resources and the narrower policy-excess proxy (candidate
reacquires but contemporaneous FULL does not). Autonomous reacquisition and
official task success remain the primary endpoints.

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

Screen heuristic opportunity without loading a model:

```bash
python -m experiments.paper8_5_agent_memory.run_negative_heuristic_screen \
  --trajectory task1.traj.json task2.traj.json \
  --tokenizer /path/to/frozen/tokenizer \
  --head 0 --tail 2 \
  --output negative_structural.json --include-decision-rows
```

Then run isolated frozen arms against one contemporaneous `full.json`:

```bash
python -m experiments.paper8_5_agent_memory.run_frozen_replay \
  --trajectory task1.traj.json --output h1_kf0.json \
  --base-url http://host:port --model MODEL --tokenizer TOKENIZER \
  --reference-replay full.json --policy h1_search_consumed \
  --head 0 --tail 2 --search-delay-turns 0

python -m experiments.paper8_5_agent_memory.run_frozen_replay \
  --trajectory task1.traj.json --output h3_kr1.json \
  --base-url http://host:port --model MODEL --tokenizer TOKENIZER \
  --reference-replay full.json --policy h3_read_superseded \
  --head 0 --tail 2 --same-span-reads-to-keep 1
```

Available combination labels are `safe2_h1_h3`, `h2a_h3`, `h2b_h3`,
`safe3_h1_h3_h2b`, and `all_h1_h2a_h2b_h3_h4`. Use
`--negative-fallback recency|lexical|spine_lexical` only when testing the
second stage at a budget below the residual history. Configuration and budget
semantics are hashed into the resume contract.

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

Autonomous single-task qualification uses a local OpenAI-compatible proxy. The
mini-swe-agent process retains its full canonical history; the proxy derives a
fresh logical plan for each call and forwards an ordinary text-only request.
`FULL` preserves the incoming messages exactly. Negative arms preserve the
system prompt, task statement, and latest causal turn. The runner records
content-only full/selected/materialized token counts, exclusion provenance,
immediate resource reacquisition, action counts, and repeated search/read/test
signatures, then invokes the official SWE-bench grader by default.

The autonomous runner also attempts a fail-closed auxiliary workspace-state
outcome. This is useful when an agent changed the repository correctly but put
a source excerpt, rather than a Git diff, in its submission. The extractor
accepts only the chronologically last pre-action checkpoint from one complete,
contiguous instrumentation session. Normally its paired final execution receipt
must show the same complete workspace fingerprint before and after the action.
mini-swe-agent's terminal submission raises before that receipt is written, so
an exact `N` decision / `N-1` execution sequence has a second guarded path: one
unique trajectory must end in `Submitted`, the final assistant command digest
must equal the unmatched decision, the submitted-output digest must equal the
primary prediction, and the command must begin with the exact submission
sentinel followed only by an allowlisted read/diff pipeline. Extra shell
commands, control operators, redirection, command substitution, editing `sed`,
output-writing or external `git diff` modes, and unclassified programs are
rejected. Every checkpoint digest must match; the untracked archive must be
empty; and at most one of the staged and unstaged patches may be non-empty. An
incomplete latest checkpoint is never replaced with an older one. These
restrictions avoid calling stale, post-mutation, partially captured, or
ambiguously composed state the final workspace.

When admitted, the runner copies the exact index and worktree components to
`auxiliary_workspace_state.index.patch` and
`auxiliary_workspace_state.worktree.patch`, writes the single representable
tracked delta to `auxiliary_workspace_state.patch`, and creates a separate
`auxiliary_workspace_state_preds.json`. Provenance and rejection reasons are
recorded in `auxiliary_workspace_state.json`. The original `agent/preds.json`
and `official_result.json` are never modified or replaced. Pass
`--grade-auxiliary-workspace-state` to invoke a second official grader run; its
separately labeled result is written to
`auxiliary_workspace_state_official_result.json`. `--skip-grading` suppresses
both grader invocations while retaining extraction provenance.

mini-swe-agent 2.4.6 strips observation `extra` fields before HTTP transport.
The proxy therefore joins `execution_*.json` sidecars from the instrumented
Docker environment into a selector-only copy using the complete ordered command
digest sequence. Missing, malformed, mismatched, or multi-session receipts fail
closed: the affected decision falls back to FULL instead of applying a negative
policy. This is recorded as `selection_abstained_for_sidecar`; the switch can be
disabled only as an explicit uninstrumented diagnostic.
Neither metadata nor tombstones are serialized into the model request.

Example for one predeclared task (run `full` first, then change only `--policy`):

```bash
python -m experiments.paper8_5_agent_memory.run_autonomous_swebench \
  --benchmark-card experiments/paper4_5_agent/benchmarks/swebench_verified_easy50_baseline_success14.json \
  --task-index 1 --output /results/paper8_5/task01/full_seed0 \
  --run-id paper85-task01-full-seed0 \
  --upstream-base-url http://MODEL_HOST:PORT/v1 \
  --model qwen3-coder:30b --served-model qwen3-coder:30b \
  --model-revision MODEL_REVISION \
  --tokenizer TOKENIZER_ID --tokenizer-revision TOKENIZER_REVISION \
  --policy full --budget-fraction 1.0 \
  --temperature 0 --top-p 1 --seed 0 --max-calls 40 \
  --docker-executable /usr/local/bin/docker
```

`--tokenizer whitespace --allow-whitespace-tokenizer` exists only for local
structural diagnostics and cannot be treated as paper evidence. `--preflight-only`
writes the task lock, exact policy/generation provenance, and command templates
without starting the proxy, agent, model calls, Docker, or grading.
