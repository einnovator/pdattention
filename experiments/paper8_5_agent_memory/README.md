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
4. Freeze evidence-qualified `AGENT_FULL`, `AGENT_QUALITY`, `AGENT_BALANCED`,
   and `AGENT_ECONOMY` logical plans for Paper 4.5 runtime realization.

## Cross-agent transfer

The locked cross-agent declaration is
`configs/agent_transfer_easy14_v1.json`. It retains the same ordered Easy-14
cohort but qualifies every agent against its own `FULL` control before applying
a memory policy. Report the unconditional 14-task solve rate separately from
conditional preservation on that agent's plain-success identities.

Pi and Kilo are the first typed-tool admission targets, followed by a richer
OpenHands-style harness and the record-native PRA Agent. Standard OpenAI `tool_calls` use the common recordizer directly;
only native event normalization and declared tool semantics may differ between
agents. The frozen selection floors and retirement parameters may not be
retuned during transfer. See
`docs/papers/paper8_5/cross_agent_transfer_protocol.md` for the gates and
measurement contract.

Build and run one Pi admission cell with:

```bash
python experiments/paper8_5_agent_memory/run_pi_swebench.py \
  --instance-id django__django-15368 \
  --reference-trajectory /evidence/task02-full/trajectory.json \
  --model-config experiments/paper8_5_agent_memory/pi_models.example.json \
  --output /results/paper85-pi-task02-full
```

The runner preserves Pi's native system prompt and tools, emits native JSONL,
and exports `model.patch`. Official success remains unset until that patch is
consumed by the common SWE-bench grader.

Kilo uses the same proxy and common recordizer. The reproducible admission
runner, container entry point, and two endpoint examples are
`run_kilo_swebench.py`, `run_kilo_swebench.sh`,
`kilo_config.example.json`, and
`kilo_config.proxy.example.json`. Kilo reports transport completion separately
from command success; reducers must inspect structured exit evidence or output
failure signals rather than equating `completed` with a successful tool result.

OpenHands uses `run_openhands_swebench.py` and a task-derived container with
the pinned SDK's native terminal, file-editor, and task-tracker tools. Its
condenser is explicitly `None` in the FULL control so that an agent-native
summary cannot be mistaken for a PRA selection result. Policy runs declare
portable tool effects through `configs/openhands_tool_semantics_v1.json`:
file-editor operations expose their command and path, task-tracker/think state
is progress, task-tracker updates share the stable
`agent://task-tracker` resource identity, finish is protocol finalization, and arbitrary terminal commands
remain an unknown-effect barrier unless execution middleware supplies complete
resource/effect receipts.

Codex uses `run_codex_swebench.py` and the native OpenAI Responses protocol.
Its FULL-only audit proxy preserves the Responses wire format, fills the frozen
temperature/top-p/seed/output ceiling when the CLI omits them, and records
whether each turn carries complete input or a `previous_response_id`. No
selection policy is applied until that state representation is inventoried;
Chat-Completions policy transport must not be assumed to work for Responses.

The transfer proxy exposes the frozen policy parameters instead of hiding
agent-specific defaults. For Recent Frontier M2/P1, use
`--policy frontier_dag_retirement --boundary-mode boundary_free
--frontier-recent-user-prompts 2 --frontier-protocol-exemplars 1
--frontier-allow-heuristic`. Prompt Pinned E2+F1C uses
`--policy persistent_instruction_epoch_retirement
--completed-instruction-epochs 2 --compact-completed-finalizations`.
`--tool-semantics-json` accepts a generic mapping keyed by native tool name.
Mixed tools may declare `operation_argument`, `operation_map`, and
`resource_arguments`; this lets an OpenHands-style `file_editor` classify
`view` as a read and `str_replace` as a write without embedding its schema in
the PRA runtime.

The PRA Agent admission keeps its typed records and safe execution boundary,
but backs every workspace tool with the official SWE-bench task container:

```bash
PYTHONPATH=src:. python -m experiments.paper8_5_agent_memory.run_pra_agent_swebench \
  --instance-id django__django-15368 \
  --reference-trajectory /evidence/task02-full/trajectory.json \
  --endpoint http://127.0.0.1:11434 \
  --output /results/paper85-pra-agent-task02-full
```

The runner exports `session.json`, normalized `tool_events.jsonl`, `model.patch`,
and `preds.json`. It does not silently treat a generated patch as a solve; pass
`preds.json` to the common official grader. Ordinary OpenAI native `tool_calls`
are projected into the PRA Agent's provider-neutral execution envelope instead
of being mistaken for an empty text answer.

Longer mini-swe horizons are a separate capability experiment. The locked
`benchmarks/easy50_horizon_limit5.json` cohort contains every original Easy-50
failure that stopped with `LimitsExceeded`. Run FULL at 80 calls first and
advance only tasks showing forward progress to 120 calls. These outcomes must
not retroactively enlarge the Easy-14 policy cohort; report extra solves and
their cumulative-token cost separately.

## Multi-issue frontier

The primary operating target is **30--50% failure-aware cumulative input-token
saving with no observed loss in official issue resolution**. During discovery,
"no loss" means that every issue resolved by its paired persistent-FULL control
also resolves under the candidate; an extra candidate success cannot compensate
for a lost FULL success. Confirmation admits no noninferiority margin: the lower
95% ordered-sequence-clustered paired resolution-delta bound must be at least
zero. A setting below 30% can remain a useful mechanism or combination parent,
but it is not a candidate default profile.

The discovery frontier treats one through five ordered issues as the initial
session axis; the current frozen confirmation mechanism cohort extends one
continuous persistent session through all fourteen Easy-14 identities. Each
sequence is run in both `fresh_per_issue` and `persistent` modes where the
comparison requires both controls.
Independent cross-repository sequences are a hygiene/control stratum; related
same-repository and dependent same-workspace sequences are the strata in which
retained prior state can legitimately outperform deliberate fresh sessions.
These strata are never pooled.

Every arm and parameter grid is registered in
`configs/multi_issue_strategy_registry_v1.json`. Simple strategies are tested
alone at two and three issues. A mixture is allowed only after a parent is
nondominated, and only winning frozen settings advance first to four and five
issues, then to the Easy-14 asymptotic mechanism cohort.
This avoids a blind strategy Cartesian product while preserving the rationale
and failure hypothesis for every reported point. Winning settings then advance
to the Easy-14 asymptotic mechanism cohort.

Reduce one or more JSON/JSONL run ledgers with strict pairing, config-digest,
failure-aware and pre-divergence accounting:

```bash
python -m experiments.paper8_5_agent_memory.multi_issue_frontier \
  --registry experiments/paper8_5_agent_memory/configs/multi_issue_strategy_registry_v1.json \
  --input /results/run-ledger.jsonl \
  --output /results/multi-issue-reduction.json
```

Then aggregate repeated executions at the ordered-sequence-family level and
render separate curves for each sequence stratum:

```bash
python -m experiments.paper8_5_agent_memory.multi_issue_curves \
  --reduction /results/multi-issue-reduction.json \
  --output-directory /results/multi-issue-curves
```

The generated plots include official resolution versus failure-aware saving
against both persistent FULL and fresh FULL, all-run and joint-success call
deltas, and first-action divergence against saving accumulated only before the
divergence. Single-sequence intervals are explicitly labelled non-inferential.
The logical-to-runtime handoff and stakeholder evidence requirements are frozen
in `configs/paper4_5_profile_promotion_contract_v1.json`; no candidate receives
a product profile name from frozen replay alone.

Run the frozen autonomous N=1--3 fresh/persistent matrix with:

```bash
PYTHONPATH=src:. python -m experiments.paper8_5_agent_memory.run_autonomous_multi_issue_campaign \
  --spec experiments/paper8_5_agent_memory/configs/autonomous_multi_issue_frontier_v1.json \
  --output /results/paper85-autonomous-multi-issue-v1 \
  --upstream-base-url http://MODEL_HOST:PORT \
  --tokenizer /models/qwen3-coder-tokenizer \
  --docker-platform linux/amd64
```

The runner is resumable at issue boundaries. In persistent mode it exports the
completed trajectory plus selector-only sidecar metadata, binds it to a stable
session ID, and prepends it to every request in the next issue. In fresh mode
each issue has a new session ID and no prefix. Current SWE-bench sequences use a
clean workspace per issue; dependent same-workspace sequences require a
separate benchmark with an explicit workspace-handoff digest and are never
silently inferred.

`persistent_episode_retirement` exposes independent completed-issue floors for
recent turns, mutation turns, verification turns, and clean protocol exemplars:
`--completed-recent-turns`, `--completed-mutation-turns`,
`--completed-verification-turns`, and `--completed-protocol-turns`. A protocol
exemplar is a complete assistant-action/tool-observation turn with no error,
mutation, verification, or finalization role. The floor selects the latest
eligible turn per completed issue. It is a causal deployable rule, not an
oracle add-back; its purpose is to retain ordinary tool-use conditioning when
the recent tail is dominated by recovery and submission turns.

`persistent_instruction_epoch_retirement` additionally accepts
`--completed-instruction-epochs N`.  It keeps the current genuine-user
instruction epoch and the immediately preceding `N` epochs whole while still
pinning every user-authored instruction.  This exposes a causal E0/E1/E2
quality--saving ladder without evaluator task IDs or explicit episode markers.

With `--retire-closed-instructions`, a terminally closed old instruction and
its causal interaction retire atomically instead of leaving an orphaned user
prompt. The frozen, active-context-qualified Easy-14 profile campaign is
registered in `configs/autonomous_easy14_atomic_profiles_v2_ctx131k.json`:
E3 is the conservative
quality candidate, E2 the repeat-qualified balanced candidate, and E1 the
economy candidate. Run persistent FULL before candidates; E1 stops on its first
lost paired FULL success.  The runner queries `/api/ps` before every episode
and fails closed unless the named model is active with at least 131,072 tokens;
the model's advertised maximum is not accepted as evidence of its active
runtime allocation. Export both aggregate and per-task paired metrics,
then render the descriptive N=1--14 prefix curves with:

The autonomous launcher also freezes a 900-second Docker acquisition/startup
ceiling.  This is an infrastructure guard for first-use x86 SWE images on
Apple Silicon; a timeout is quarantined and never scored as a task failure.
The official grader uses its supported `instance` cache level so that grading
one issue does not delete every pre-fetched evaluation image needed by later
episodes.  Containers are still cleaned, and image digests remain recorded.
The harness resolves and platform-checks the image before starting the agent;
`docker run` therefore never performs an implicit pull inside mini-swe-agent's
container-start subprocess.

```bash
PYTHONPATH=src:. python -m experiments.paper8_5_agent_memory.export_multi_issue_evidence \
  --campaign-root /results/easy14-atomic-profiles-v1 \
  --output /results/easy14-atomic-evidence
PYTHONPATH=src:. python -m experiments.paper8_5_agent_memory.plot_easy14_atomic_frontier \
  --evidence /results/easy14-atomic-evidence/evidence.json \
  --output /results/easy14-atomic-evidence
```

These prefix points share one ordered sequence and are descriptive, not
independent observations or confidence intervals.

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
are not inserted into the model prompt. The separate
`--negative-realization observation_receipt` treatment keeps the original H1/H2
assistant action and replaces only its paired observation with a compact,
model-visible resource-map or mutation receipt. It uses the active tokenizer
and fails closed to the complete original group unless the receipt is strictly
smaller than the source observation. Candidate, dropped, source-observation,
receipt, and abstention tokens are reported independently. Frozen replay reports immediate reacquisition
of excluded resources and the narrower policy-excess proxy (candidate
reacquires but contemporaneous FULL does not). Autonomous reacquisition and
official task success remain the primary endpoints.

Before assigning a real-task loss to a heuristic, run the three synthetic
mechanism diagnostics. They isolate consumed discovery, versioned write/read/
verification convergence, and a decisive traceback buried in a long tool
result:

```bash
python -m experiments.paper8_5_agent_memory.run_synthetic_diagnostics \
  --output-directory docs/papers/shared/results/paper8_5_agent_memory/synthetic_diagnostics
```

The `tool_structured_evidence` materializer is the first non-positional result
compactor. It ranks only current-request evidence: failure/traceback lines,
paths, diff structure, source symbols, command resources, and task/query terms.
If none is present, it keeps the whole observation instead of silently falling
back to arbitrary head/tail sampling. If its selected spans cover the complete
observation, it returns the canonical bytes rather than rebuilding the string;
this is required for a true no-op control. Synthetic structural gates are not model
or task-quality evidence; each generated trajectory is also compatible with
the ordinary frozen replay runner.

mini-swe-agent uses an evaluation-only Bash adapter because its only tool is
generic Bash. That adapter now emits portable `operation_kind`, resource access,
version, span, discovery, mutation, completeness, and causal-group metadata.
The state-authority selector consumes only those declarations: it no longer
parses Bash commands. `HarnessMetadataSemanticsProvider` is the portable path
for other agents, while engines consume only the resulting logical plan and
never embed Bash- or application-specific reasoning.

The same `WireAgentMemoryPlan` is realized by the ordinary-text evaluation
proxy and the product mediator. Its source-history and plan digests reject stale
or conflicting application. This means adding embedded/external gateway
placement does not require rerunning every policy arm: byte-identical frozen
plans need only a placement conformance test. A new quality run is required
only when selected identities or model-visible replacement bytes change.

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
`auxiliary_workspace_state_official_result.json`. This second grade is still
run when the primary submission has a patch-apply error: that is the diagnostic
case needed to separate workspace capability from submission-protocol
reliability. The campaign runner accepts the same flag. `--skip-grading`
suppresses both grader invocations while retaining extraction provenance.

If auxiliary grading must run on another host, an `autonomous_runs` entry in
the curve specification may declare `auxiliary_grade`. The referenced receipt
must identify itself as `auxiliary_workspace_grade`, bind the task and exact
`auxiliary_workspace_state.patch` SHA-256, and bind a grader report by path and
SHA-256. The reducer rejects mismatched patches, tasks, reports, and outcomes.
This imports the auxiliary capability result without editing the raw run and
without changing the failed primary-submission endpoint.

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

## Adaptive quality--saving curves

Build the common frozen/autonomous reduction with:

```bash
python -m experiments.paper8_5_agent_memory.tradeoff_curves \
  --spec experiments/paper8_5_agent_memory/configs/tradeoff_current.json \
  --output docs/papers/shared/results/paper8_5_agent_memory/tradeoff_curves
```

The reducer keeps official autonomous accuracy separate from frozen action
agreement, and gross within-run compression separate from paired end-to-end
token saving. Its adaptive decision table prevents dominated or sharply
degrading whole-group heuristics from entering combinations or wider task
cohorts. Controls and mechanism-only arms remain visible without being
mislabelled as profile candidates.

### Locked five-task autonomous curve

The current certificate-only task-quality campaign is declared in
`configs/autonomous_long5_curve_certified_v3.json`. Its benchmark card locks the five
longest tasks from the Easy-14 baseline-success stratum before any treatment
outcomes are observed. It runs two FULL controls per task, then DAG-only,
strict matched-token tail at 90%, whole-record H1+H3 retirement, and the same
retirement decisions with the assistant action/provenance retained and only
the observation payload replaced by a typed receipt. The matched control
preserves immutable prompt records and complete recent turns, trimming only an
oversized boundary tool observation; it reports immutable-prompt overflow
separately when the requested ceiling is impossible. No 80% arm, broader
selector family, second model, or second harness is scheduled before the 90%
and payload/provenance gates pass.
The autonomous contract requires an explicit positive completion-token cap;
the locked Qwen3-Coder campaign uses 1,024, matching its qualified Paper 8.5
controls. An omitted cap is rejected before any task is launched because a
runaway response otherwise changes both task behavior and end-to-end cost.

Run or resume one machine's task partition with:

```bash
PYTHONPATH=src:. python -m experiments.paper8_5_agent_memory.run_autonomous_curve_campaign \
  --spec experiments/paper8_5_agent_memory/configs/autonomous_long5_curve_certified_v3.json \
  --output /results/paper85-autonomous-long5 \
  --upstream-base-url http://127.0.0.1:11434/v1 \
  --tokenizer /exact/tokenizer/snapshot \
  --docker-executable /usr/local/bin/docker \
  --task-index 1 --task-index 2 --reduce
```

When a selector-only correction lands after the two FULL controls have already
completed, `--import-control-state /path/to/campaign_state.json` reuses those
controls instead of rerunning them. The corrected spec must declare
`baseline_pair_campaign_id`; imported artifacts are bound by state-file hash.
The reducer then permits only the disclosed repository-revision difference and
still rejects any model, tokenizer, dataset, harness, environment, generation,
or agent-command mismatch.

The campaign is resumable and arm-major after its controls. It stops an arm
after two official failures, after sub-2% mean gross saving on two tasks, or
after resolution falls below 80% on at least three tasks. Failed candidates
are charged at least their paired FULL input workload in the failure-aware
efficiency coordinate, so early termination cannot appear as a saving. The
runner writes a reducer-ready `tradeoff_spec.json`; `--reduce` emits accuracy
versus within-run, paired end-to-end, and failure-aware token-saving curves,
plus tool-call and first-divergence diagnostics. Endpoint-reported completion
tokens are included only when every call has usage coverage. Pairing freezes
the task/dataset, repository, model/tokenizer, decoding, harness/grader,
environment image, instrumentation mode, and a normalized agent-command
digest; run-local proxy, Docker-executable, instrumentation, and output paths
are deliberately excluded from that digest. FULL-A versus FULL-B endpoint and
trajectory agreement is emitted as its own control table rather than being
attributed to a selection policy. The divergence curve charges only saving on
requests strictly before the first changed action, and overlapping suffix and
full-trajectory frozen cohorts are collapsed to the broadest coverage for
adaptive decisions. Grader-reported malformed-patch/apply errors remain task
failures when the official report supplies a definitive Boolean grade. Only a
run without a definitive grade is excluded from accuracy denominators,
task-failure stopping gates, and efficiency-qualified pairs.

The earlier frozen `configs/autonomous_long5_curve.json` remains available for
reproduction, but its historical arm ID `dag100` names the broader
`task_aware_progress_spine_v4` heuristic. It is not certificate-only DAG
exclusion and must be reported as state-authority heuristic selection. The v3
spec removes that naming ambiguity: `dag_certified100` applies only exclusions
proved from runtime-bound operation identity. `whole_record_h1h3` and
`payload_stub_h1h3` share exactly the same H1+H3 decisions and differ only in
whether they delete the causal group or retain action/provenance plus a
model-visible observation receipt.

For a failed frozen policy decision, generate a one-group-at-a-time oracle
diagnostic queue with:

```bash
python -m experiments.paper8_5_agent_memory.oracle_addback_queue \
  --candidate /results/policy-replay.json \
  --output /results/policy-replay.oracle-addback.json
```

Each queued trial restores exactly one complete causal group at the first
command divergence and can be executed by adding the emitted arguments to
`run_frozen_replay`. Groups are ranked by restored tokens so the first trial
tests the smallest counterfactual intervention. These trials diagnose which
omission caused a divergence; they are explicitly oracle evidence and cannot
be reported as autonomous policy quality.

Oracle headroom uses the complementary `--oracle-omit-group` mode with
`--policy full`. It removes only complete middle causal groups and rejects any
requested system/task/head/tail group. For example, a leave-one-group-out
decision probe is:

```bash
python -m experiments.paper8_5_agent_memory.run_frozen_replay \
  --trajectory /results/trajectory.json \
  --output /results/oracle-turn-0007.json \
  --base-url http://127.0.0.1:8092/v1 \
  --model /exact/model/snapshot \
  --tokenizer /exact/tokenizer/snapshot \
  --policy full --head 2 --tail 4 \
  --min-decision 20 --max-decisions 1 \
  --oracle-omit-group turn:t0007
```

The future action is used only after generation for scoring. Single-group
survivors can seed predeclared pair removals or bounded beam search; oracle
outcomes never enter deployable-policy accuracy curves.

After singleton completion, generate a bounded pair queue from only valid,
exact-command-preserving singleton omissions:

```bash
python -m experiments.paper8_5_agent_memory.oracle_headroom_queue \
  --input-directory /results/task/decision_24 \
  --target-depth 2 --beam-width 8 \
  --output /results/task/decision_24/pair_queue.json
```

The primary queue uses exact commands. A separately labelled secondary queue
may use `--qualification semantic_transition`, which requires the same typed
operation and target resources and explicitly recognizes syntax-check variants
such as `ast.parse` versus `py_compile`. Both layers remain visible; semantic
equivalence must never be reported as exact reproduction.

After executing the queued pairs into the same directory, rerun with
`--target-depth 3`. Only successful depth-two parents are expanded, giving a
bounded beam search rather than a combinatorial subset sweep.

Execute a digest-bound queue with one tokenizer load and resumable per-subset
artifacts:

```bash
python -m experiments.paper8_5_agent_memory.run_oracle_subset_queue \
  --trajectory /results/task/trajectory.json \
  --reference-replay /results/task/full_a.json \
  --queue /results/task/decision_24/pair_queue.json \
  --output-directory /results/task/decision_24 \
  --base-url http://127.0.0.1:8092 \
  --model /exact/model/snapshot \
  --tokenizer /exact/tokenizer/snapshot
```

For a full leave-one-middle-bundle-out decision sweep, use the resumable
`run_oracle_singletons.sh` driver. `PAPER85_LAST_GROUP` is the last unprotected
middle turn index after applying the declared head and tail floors:

```bash
PAPER85_PYTHON=/qualified/bin/python \
PAPER85_MODEL=/exact/model/snapshot \
PAPER85_TOKENIZER=/exact/tokenizer/snapshot \
PAPER85_BASE_URL=http://127.0.0.1:8092 \
PAPER85_TRAJECTORY=/results/task/trajectory.json \
PAPER85_REFERENCE=/results/task/full_a.json \
PAPER85_DECISION=24 PAPER85_LAST_GROUP=20 \
PAPER85_OUTPUT=/results/task/decision_24 \
  sh experiments/paper8_5_agent_memory/run_oracle_singletons.sh
```

### Recordizer reliability audit

Prepare a blinded, stratified worksheet before crediting semantic policies to
automatic record types:

```bash
python -m experiments.paper8_5_agent_memory.recordizer_audit prepare \
  --trajectory /results/task01/trajectory.json \
  --trajectory /results/task02/trajectory.json \
  --trajectory /results/task03/trajectory.json \
  --sample-size 150 \
  --seed 850 \
  --output /results/recordizer-audit
```

The sample balances task identity, automatic primary role and trajectory
tercile. It keeps tool-grounded and prose-inferred labels explicit and emits
economical head/tail previews rather than copying large observations into the
worksheet. After a blinded human fills the `human_*` columns, score per-role
precision/recall and exact multilabel agreement with:

```bash
python -m experiments.paper8_5_agent_memory.recordizer_audit score \
  --audit-csv /results/recordizer-audit/recordizer_audit.csv \
  --output /results/recordizer-audit/recordizer_audit_scores.json
```
