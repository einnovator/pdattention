# AGENTS.md — Paper 8.5 Inception
## Agent Memory Selection under Bounded Context
### Engine-Independent Retention Policies for Software-Engineering Agents

Repository: `einnovator/pdattention`

Suggested branch:
`research/paper8-5-agent-memory`

Primary first harness:
`SWE-agent/mini-swe-agent`

Related PRA papers:
- Paper 2.x: retrieval and routing for prompt/document memory.
- Paper 3.x: materialization, composition, and RAG/native-memory tradeoffs.
- Paper 4.5: engine/runtime realization and native-K/V correctness.
- Paper 8: agent/task/session record model.
- Paper 9: subagent lineage, effects, invalidation, and shared memory.

This paper studies **which agent-history records should be retained for the next model call**.

It does **not** study engine-specific K/V reuse, TTFT/latency, zero-copy page attachment, backend-specific cache lifecycle, or native sparse-kernel speed. Those remain Paper 4.5 / engine concerns.

The core independent variable is the logical selection policy:

```text
full canonical agent history
        ↓
agent-memory selector
        ↓
selected logical records/spans
        ↓
ordinary model call
```

The paper's goal is to identify selection/materialization policies that preserve task success while reducing active history and avoiding inefficient recovery behaviors such as repeated source reads, repeated tests, and repeated failed edits.

---

# 1. Scientific question

At model turn `t`, let the canonical agent history be:

```text
H_t = [r_1, r_2, ..., r_n]
```

where each `r_i` is an agent record.

A policy computes:

```text
S_t = π(H_t, Q_t, B)
```

where:
- `H_t`: canonical history;
- `Q_t`: task/current active state;
- `B`: memory budget;
- `S_t`: selected records/spans.

Central question:

> Which historical records are necessary for a coding agent to preserve task success and interaction efficiency under bounded context?

This is a semantic policy problem, not a cache-engine problem.

---

# 2. Why mini-swe-agent first

Use mini-swe-agent as the first harness because its control loop is unusually clean:

```text
linear message history
→ model query
→ bash action
→ environment observation
→ append to history
```

Relevant properties:
- one linear message history;
- history equals the trajectory passed to the model;
- bash-only actions;
- subprocess-style independent action execution;
- minimal harness-managed hidden state;
- easy trajectory serialization and replay.

Do not turn mini-swe-agent into a complex framework for this paper. Keep the harness minimal and inject only the recordization/selection boundary.

---

# 3. Paper boundary

## Paper 8.5 owns
- logical recordization of agent history;
- memory-selection policies;
- record-level and span-level materialization policies;
- retention budgets;
- frozen-trajectory counterfactual screening;
- autonomous reruns;
- agent-specific metrics;
- dependency-horizon analysis;
- recovery/reread behavior;
- oracle record-utility analysis;
- transfer to one richer open-source harness;
- transfer of selected policies into PRA Agent.

## Paper 8.5 does not own
- native K/V transport;
- same-state cache equivalence;
- physical K/V copy;
- engine-specific position handling;
- offload/restore;
- page aliasing;
- latency benchmarking.

Paper 4.5 later consumes qualified Paper 8.5 policies.

---

# 4. Canonical agent record model

Normalize mini-swe-agent linear messages into typed logical records.

Minimum record types:

```text
TASK
ASSISTANT_ACTION
TOOL_OBSERVATION
SOURCE_VIEW
MUTATION
VERIFICATION
PROGRESS
ERROR_OR_REJECTION
FINALIZATION
```

Recommended semantics:

## TASK
Original problem statement and immutable constraints.

## ASSISTANT_ACTION
Executed command or intended shell action.

## TOOL_OBSERVATION
Environment output paired with an action.

## SOURCE_VIEW
Observation that actually exposes source/content, e.g. `cat`, `sed -n`, source-bearing traceback, diff, or test failure. Path-only `find`, `ls`, or filename-only `grep` should not automatically replace a more informative source-content record.

## MUTATION
Edit/patch/write operation that changes the workspace.

## VERIFICATION
Test/build/lint/check result.

## PROGRESS
Current hypothesis, plan, target, diagnosis, or intermediate conclusion.

## ERROR_OR_REJECTION
Format error, rejected action, command failure, harness error.

## FINALIZATION
Submission/final response.

---

# 5. Causal bundles

Some records must be retained/dropped as groups.

At minimum:

```text
ASSISTANT_ACTION
      +
TOOL_OBSERVATION
```

For edits:

```text
MUTATION
   ↓
VERIFICATION
```

For source acquisition:

```text
SOURCE_REQUEST
   ↓
SOURCE_VIEW
```

Each record should contain:
- stable record ID;
- turn ID;
- causal group ID;
- source URI/file where applicable;
- source span where applicable;
- mutation target;
- verification target;
- created-at logical step;
- record type.

Selection policies should not break mandatory causal groups unless explicitly testing that ablation.

---

# 6. Engine-independent selector API

Define a pure logical interface.

```python
class AgentMemorySelector(Protocol):
    def select(
        self,
        *,
        task: AgentTask,
        records: Sequence[AgentRecord],
        active_state: AgentState,
        budget: AgentMemoryBudget,
    ) -> AgentMemoryPlan:
        ...
```

`AgentMemoryPlan` contains only:

```text
selected record IDs
optional source spans
priority
selection reason
causal group IDs
requested budget
realized budget
```

It must not contain tensors, K/V arrays, pages, engine slots, or backend handles.

A serializer converts the selected logical plan into ordinary text messages for the main Paper 8.5 experiments. Later Paper 4.5 maps the same plan to native engine objects.

---

# 7. Text realization for the main study

Main study realization:

1. maintain the complete canonical trajectory;
2. apply the memory selector at each model turn;
3. serialize only selected records in canonical causal order;
4. query the same model with ordinary text;
5. execute the produced action;
6. append resulting full canonical records to the underlying history;
7. repeat.

Important limitation:

> Text reserialization recontextualizes selected records. Paper 8.5 measures logical memory sufficiency, not native-K/V equivalence.

State this explicitly.

---

# 8. Core policy ladder

Implement policies from simple to structured.

## P0 — FULL
All history. Reference.

## P1 — TOKEN_TAIL
Last `B` tokens.

## P2 — TURN_TAIL
Last `N` complete turns.

## P3 — RECORD_TAIL
Last `M` records.

## P4 — CAUSAL_TAIL
Last `N` complete causal bundles.

## P5 — PROGRESS_SPINE
Always retain:
- task;
- active tail;
- recent complete turns;
- latest source-view bundle;
- latest mutation bundle;
- latest verification bundle;
- latest progress/hypothesis record.

## P6 — LEXICAL
Task/current-state BM25 over historical records.

## P6b — DAG_CERTIFIED_EXCLUSION
Build a resource/effect DAG and remove only records with a machine-checkable
certificate of operational obsolescence under a declared state abstraction.
This negative-selection stage runs before lexical, dense, or hybrid positive
selection.

## P7 — DENSE
Embedding similarity over historical records.

## P8 — HYBRID
Structured progress spine plus lexical/dense retrieval.

## P9 — MODEL_SELECTOR
Small/model-assisted record selector. Do not unlock until strong parameter-free baselines are measured.

## P10 — ORACLE
Retrospective utility-based upper bound.

---

# 8.1 Head--middle--tail policy geometry

Do not conflate the immutable beginning of an agent run with its recency tail.
Every structured policy must expose two independent controls:

```text
SYSTEM + TASK
+ first H complete causal turns
+ selected middle turns
+ last T complete causal turns
```

`H` and `T` are independent. Selection and retrieval operate only on the
middle interval after removing overlap between the head and tail. The initial
system/task records remain pinned even when `H=0`; `H` counts completed
assistant--observation turns after the task statement.

Required pilot values:

```text
H in {0, 1, 2, 4}
T in {1, 2, 3, 5, 10}
```

The first sweep may reduce this grid using frozen replay, but autonomous rows
must report both values. This design tests whether early localization and plan
formation remain useful after they leave a conventional tail.

Head/tail retention is at causal-turn granularity, with one exception: an
oversized tool observation may use the same typed detail policies as the PRA
gateway. Even then preserve the action, return code, mutation/verification
status, resource identity, failure headline, and stable parent/causal IDs.
Only the observation body may be reduced through natural child spans,
matched spans, or bounded head/tail detail. Report record selection and detail
materialization as separate decisions and token counts.

If mandatory system/task/head/tail state exceeds the requested budget, never
silently break a causal group. Emit the requested budget, realized budget, and
mandatory-overflow count and classify the cell as rounded-up or infeasible.

---

# 8.2 Resource/effect DAG and negative exclusion

Run negative exclusion before budgeted positive selection:

```text
canonical history
-> certificate-backed DAG exclusion
-> uncertain remainder
-> head/middle/tail positive selector if still above budget
```

Use immutable, versioned resource nodes and typed edges:

```text
ACTION -> OBSERVATION                  returns
RESOURCE_VERSION -> ACTION             reads_from
ACTION -> RESOURCE_VERSION             creates/writes/deletes
RESOURCE_VERSION_OLD -> VERSION_NEW     superseded_by
OBSERVATION -> PROJECTION               reports
PROJECTION_OLD -> PROJECTION_NEW        subsumed_by
```

Every effect records provenance as declared tool semantics, runtime tracing,
static heuristic, or unknown. `UNKNOWN` is an exclusion barrier. A certificate
records the rule ID, proof scope, witnesses, cwd/environment identity,
resource-version fingerprints, output completeness, and assumptions.

Never describe the certificate as proof of identical LLM behavior: removing
duplicate text still changes positions, attention, and repetition. The
defensible claim is **certificate-backed operational obsolescence under the
declared resource-state abstraction**.

Keep three tiers separate:

- `DAG_CERTIFIED`: operationally obsolete under a complete certificate;
- `DAG_RECOVERABLE`: a known command can reacquire the state;
- `DAG_HEURISTIC`: supersession or dependency evidence exists but is incomplete.

Only the first tier participates in `DAG-EXCLUDE@B=100`. The latter two require
ordinary counterfactual and autonomous quality qualification. Do not call this
arm `PRA-100`, which is reserved for Paper 4.5's exact full-history runtime gate.

Initial rules fail closed:

- never exclude system, task, incomplete/current turn, mandatory head/tail,
  unresolved error, latest unverified mutation, or an unknown effect;
- require complete stdout/stderr, return code, no timeout/truncation, canonical
  cwd, and input resource-version fingerprints;
- an unchanged full read can subsume an earlier covered read only with byte and
  span identity;
- git status/diff requires separate HEAD, index, and worktree fingerprints;
- a later pass never automatically erases a prior failure;
- a write makes old state non-authoritative but does not prove its diagnostic
  information irrelevant;
- preserve complete action--observation causal bundles.

After DAG exclusion, apply the ordinary selector against the original absolute
budget. Report certified, recoverable, and heuristic token removals separately.

mini-swe-agent has only one generic Bash tool. Implement Bash inference as a
special harness-side `ToolSemanticsProvider`; do not put Bash parsing into the
PRA engine. Regex/static classification is diagnostic, while certification
requires declared semantics or runtime tracing. The portable SDK vocabulary is
generic: read/write/delete/verify effects, resource/version IDs, dependencies,
completeness, confidence, certificates, and optional record dispositions such
as protected, certified-excludable, or preferred-for-selection.

Paper 4.5 engines consume the resulting frozen `AgentMemoryPlan` and generic
record metadata. They validate and realize it; they do not understand Bash,
Excel, browser, database, agent, or application-specific rules. This boundary
must remain explicit so future agents and tool categories can supply their own
semantics providers without changing the engine contract.

---

# 9. Budget axis

Report both relative and absolute budgets.

Relative:

```text
100%
90%
75%
50%
25%
```

Absolute:

```text
2K
4K
8K
16K
32K
```

For every request persist:
- full-history tokens;
- selected tokens;
- retention fraction;
- record count;
- causal bundle count;
- selected bytes/characters if useful.

Do not treat percentage alone as sufficient.

---

# 10. Selection vs materialization factorial

Separate `which records?` from `how much detail from each record?`.

Materialization policies:

## M0 — WHOLE_RECORD
Retain full record.

## M1 — MATCHED_SPAN
Retain only matching source interval.

## M2 — HEAD_TAIL
Retain first/last bounded portions.

## M3 — CHILD_SPAN
Use hierarchical child spans for oversized records.

## M4 — SUMMARY
Optional summary/compression arm.

Primary first matrix is selector × whole-record. Test materialization only for top selectors.

---

# 11. Two-phase experiment design

## Phase I — frozen-trajectory counterfactual screening

Use successful FULL-history trajectories.

At each historical model call:
1. restore exact repository/environment state;
2. reconstruct canonical records available at that turn;
3. apply candidate selector/materializer;
4. query model deterministically;
5. compare next behavior against baseline trajectory.

Measure:
- exact response match;
- exact command match;
- command equivalence;
- target file match;
- edit-target match;
- first divergence;
- logit divergence if available;
- whether omitted records reappear as dependencies.

This phase is for high-throughput policy screening. It is not a substitute for autonomous evaluation.

## Phase II — autonomous reruns

For qualified policies:
1. reset task sandbox;
2. run mini-swe-agent autonomously with selector active;
3. grade official task outcome;
4. measure calls, recovery behaviors, selected context, and failures.

Autonomous runs are the primary task-quality evidence.

---

# 12. Initial benchmark

Start with:

```text
SWE-bench Verified Easy-14
```

because known baseline-success tasks and trajectories already exist in Paper 4.5.

Then expand to:
- Easy-50;
- broader SWE-bench Verified sample.

Do not jump to full Verified before policy bugs are resolved on Easy-14/50.

---

# 13. Models

Discovery:
- one fixed small/medium coding model already used in current agent work.

Replication:
- one stronger model;
- one different family if practical.

Freeze model revision, tokenizer, temperature, decoding parameters, mini-swe-agent revision, task environment image, and prompt templates.

Do not confound initial selector discovery with many model families.

---

# 14. Primary task metrics

## Official success
SWE-bench resolved / official grader. Primary endpoint.

## Calls to solution
Number of model calls.

## Actions
Number of shell actions.

## Trajectory tokens
Total selected input tokens consumed across the run.

## Final patch quality
Official patch/test outcome.

Do not declare a policy superior solely from token reduction.

---

# 15. Agent-specific efficiency metrics

## Avoidable rereads
Count reacquisition of previously observed information:
- repeated `cat` of same file/range;
- repeated overlapping `sed`;
- repeated `grep` for same fact;
- repeated test solely because prior result was forgotten.

Persist:

```text
reacquisition_calls
reacquisition_source_tokens
reacquisition_fraction
```

## Repeated verification
Same/equivalent test invoked again without relevant intervening mutation.

## Repeated failed edit
Reattempt caused by forgotten prior mutation or error.

## Source localization regression
Broad whole-file reread replacing previously known targeted span.

These may be more sensitive than final task success.

---

# 16. Behavioral failure taxonomy

At minimum:

```text
MEMORY_TASK_LOSS
MEMORY_SOURCE_LOSS
MEMORY_MUTATION_LOSS
MEMORY_VERIFICATION_LOSS
MEMORY_PROGRESS_LOSS
MEMORY_CAUSAL_PAIR_BREAK
MEMORY_STALE_HYPOTHESIS
MEMORY_REDUNDANT_REREAD
MEMORY_REDUNDANT_TEST
MEMORY_WRONG_FILE
MEMORY_REPEAT_FAILED_EDIT
MEMORY_FORMAT_REGRESSION
MEMORY_RECOVERY_SUCCESS
MEMORY_RECOVERY_FAILURE
```

---

# 17. Dependency-horizon analysis

For each consequential action at turn `t`, estimate the age of the historical record(s) needed to support it.

```text
dependency_age = t - created_step(record)
```

Report distributions by record type:
- source view;
- mutation;
- verification;
- progress;
- task.

Questions:
- how far back do coding agents actually need memory?
- do source observations have longer lifetimes than tests?
- how long do mutation records remain useful?
- are progress/hypothesis records short-lived or persistent?

This should be a central scientific analysis.

---

# 18. Oracle utility study

Construct a retrospective upper bound using successful trajectories.

Start at bundle granularity.

For each historical bundle `g` and future decision point `t`:

```text
U(g,t) = degradation when g is removed
```

Possible outcomes:
- exact next action;
- action equivalence;
- task-oriented score;
- logit/response divergence.

Do not begin with token-level ablation.

Hierarchy:

```text
record type
→ causal bundle
→ source span
```

The oracle answers:

> How compact can history be if we know which records will matter later?

Use two explicitly different oracle targets:

## O1 -- next-action oracle

At a frozen decision point, search subsets of middle causal bundles while
holding system/task/head/tail fixed. Score exact command, command equivalence,
target file, edit intent, and reference-action likelihood where available.
Future action data is used only by the offline scorer and must never appear in
the model request.

Begin with leave-one-bundle-out and role ablations. Because bundle utility can
be non-additive, use pairwise ablations and bounded beam/branch-and-bound search
on small histories before calling a set minimal. A union of individually
important bundles is an importance oracle, not automatically a sufficient-set
oracle.

## O2 -- rollout oracle

For a small predeclared subset of tasks and frontier decision points, resume
the agent from the exact repository snapshot and score eventual official task
success plus calls and recovery behavior. This expensive oracle checks where
next-action equivalence is too strict or too weak. It is an upper bound and is
never a deployable policy.

---

# 19. Policy ablations

For the Progress Spine, ablate:

```text
-task
-source
-mutation
-verification
-progress
-recent-turns
```

Measure task success, calls, rereads, tests, and first divergence.

---

# 20. Lexical/dense/hybrid retrieval

For retrieval policies, use query:

```text
task
+
current progress/hypothesis
+
latest active observation
```

Candidate records:
- prior source views;
- prior mutations;
- prior verifications;
- progress records;
- other observations.

Compare BM25, dense embedding, and RRF hybrid.

Keep retrieval inside agent history only. Do not search the external repository/corpus through this policy.

---

# 21. Model selector

Only after P5–P8 baselines.

Potential input:
- task;
- active state;
- compact metadata for candidate records.

Output:
- record IDs / priorities.

Train or prompt against oracle utility labels and successful trajectory dependencies.

Keep selector cost separate from primary model cost.

---

# 22. Statistical protocol

For discovery:
- deterministic temperature 0;
- paired task-level comparisons.

For autonomous final evaluation:
- multiple task identities;
- multiple model seeds only if decoding is stochastic;
- paired bootstrap over tasks.

Do not call repeated bootstrap samples independent seeds.

Easy-14 = mechanism/pilot evidence. Publication-quality claims should expand to Easy-50 or larger.

---

# 23. Main result tables

## Table 1 — policy × budget

```text
Policy
Budget
Resolved
Calls/task
Input tokens
Selected fraction
Rereads
Repeated tests
```

## Table 2 — semantic role ablation

```text
Policy variant
Resolved delta
Calls delta
Reread delta
First-divergence rate
```

## Table 3 — frozen trajectory screening

```text
Policy
Exact next-action %
Equivalent next-action %
Median retained tokens
First-divergence turn
```

## Table 4 — materialization

```text
Selector
Materializer
Resolved
Selected tokens
Calls
Rereads
```

---

# 24. Main figures

1. Task success vs retained context.
2. Calls-to-solution vs retained context.
3. Avoidable rereads vs retained context.
4. Policy quality frontier.
5. Dependency-age distribution by record type.
6. Progress-spine ablation effects.
7. Frozen-screening quality vs autonomous task quality.

---

# 25. Success criteria

A policy is scientifically interesting if it achieves one of:

## S1 — quality preservation
Same task success as FULL with materially less context.

## S2 — interaction efficiency
Same task success but fewer calls/rereads than simple tail truncation at matched context budget.

## S3 — graceful degradation
Task quality degrades more slowly than token-tail/turn-tail baselines as budget shrinks.

## S4 — role-aware advantage
Structured semantic-role policy significantly outperforms pure recency at matched tokens.

Strong target:

```text
>= 25% reduction in selected input tokens
with no meaningful loss in official task success
and no increase in calls-to-solution
```

Do not force this threshold if data suggests a different frontier.

---

# 26. Handoff to Paper 4.5

Paper 8.5 should publish a small frozen policy set, e.g.:

```text
FULL
AGENT_BALANCED
AGENT_AGGRESSIVE
```

Paper 4.5 then tests only:

```text
Can engine E realize this exact AgentMemoryPlan correctly and efficiently?
```

Paper 4.5 should not retune semantic selection policy.

The logical `AgentMemoryPlan` must be backend-neutral and hashable.

---

# 27. Transfer beyond mini-swe-agent

After mini-swe-agent qualification:

## Stage 2
Test top policies on one richer open-source agent, preferably SWE-agent or OpenHands.

Goal: test transfer of semantic-role findings, not redo the full policy grid.

## Stage 3
Test in PRA Agent, where typed records/tasks/progress/tool effects are native and subagents can later be integrated.

---

# 28. Reproducibility artifacts

Persist:
- task ID;
- repository revision;
- mini-swe-agent revision;
- model revision;
- prompt templates;
- full canonical trajectory;
- typed recordization;
- selector configuration;
- materializer configuration;
- selected record IDs/spans per turn;
- token counts;
- environment snapshot;
- output action;
- official grader result;
- failure taxonomy labels.

No policy should depend on untracked local state.

---

# 29. Definition of done — inception

Paper 8.5 inception is complete when:
1. branch exists;
2. mini-swe-agent is pinned;
3. canonical agent-record schema exists;
4. engine-independent selector API exists;
5. FULL, token-tail, turn-tail, causal-tail, and progress-spine policies work;
6. frozen-trajectory replay works on at least 3 baseline-success tasks;
7. autonomous selector execution works on at least 3 tasks;
8. task success, calls, rereads, and selected-token metrics are emitted;
9. initial budget sweep exists;
10. draft paper compiles.

---

# 30. Definition of done — publication

Publication-quality Paper 8.5 requires:
1. Easy-50 or larger autonomous evaluation;
2. at least FULL, recency, causal, structured-spine, lexical, dense, hybrid, and oracle arms;
3. frozen screening plus autonomous validation;
4. semantic-role ablation;
5. dependency-horizon analysis;
6. at least one materialization comparison;
7. one second-model replication;
8. one second-harness reduced transfer;
9. frozen qualified policy set for Paper 4.5;
10. clean artifacts, tests, figures, and PDF.

---

# 31. Main scientific hypothesis

Primary hypothesis:

```text
token recency
<
causal recency
<
typed progress spine + retrieval
```

at matched context budgets.

The expected first benefit may appear not as a large task-success increase, but as:

```text
same task success
+
fewer model calls
+
fewer source rereads
+
fewer repeated tests
+
less retained history
```

This is a valid and important agent-memory result.
