# AGENTS.md — Paper 8 v2
## Task-Aware PRA for Interleaved Single-Session Workflows

## Gate and scope

Inception / implementation plan.

Start only after the stable Paper-7 typed-record, addressability, progressive-materialization, cursor, backing-store, and cache mechanisms are available through the SDK path Paper 8 will use.

Paper 8 is strictly:
- one physical session;
- multiple logical tasks;
- interleaved context stream.

Out of scope:
- separate-session subagents (Paper 9);
- PRA context-stream trees/DAGs across sessions (Paper 9);
- cross-session memory (Paper 10);
- inventing a new general planner;
- reopening Paper-7 materialization research.

## Working title

**Task-Aware Progressive Retrieval Attention: Virtualizing Interleaved Agent Context by Execution Scope**

## Core thesis

Prior PRA effectively assumes:

`session/context ≈ one coherent task`.

Agent sessions multiplex tasks.

Task identity, status, provenance, and explicit dependency relations should become a first-class **scope** signal before ordinary PRA discovery.

Required architecture:

`Acquire task structure -> Select task scope -> PRA discover -> Paper-7 materialize -> LLM`

Do not collapse these stages.

## 1. Responsibility boundary

### Harness/session manager = authoritative execution state

Own:
- physical session event log;
- canonical task graph/state;
- workspace;
- task status/version;
- tool execution/outcomes;
- constraints;
- evidence/result relationships;
- authorization;
- task transition validation;
- task graph mutation commit.

Model task calls are proposals until validated.

### PRA/model runtime = replayable context projection

Own:
- task-aware candidate partitions;
- task facets;
- indexes;
- routing;
- record/chunk selection;
- bounded active/native K/V;
- Paper-7 view/materialization state.

PRA is not a workflow engine.

Require:
- stable session/task/record IDs;
- monotonic event sequence;
- versions;
- idempotent replay;
- snapshots;
- invalidation;
- deterministic reconstruction.

## 2. Mandatory reuse of Paper 7

Do not create a second task-specific memory/materialization stack.

Use the inherited Paper-7 record abstraction conceptually:

`R7 = {id, type, provenance/policy, address views, visible view, backing state}`

Paper 8 extends with task provenance:

`R8 = {R7, task provenance/relations}`.

Paper 8 primarily changes **scope**.

Existing PRA handles **discovery**.

Paper 7 handles **detail/materialization escalation**.

## 3. Scope x materialization decomposition

Treat context policy as two orthogonal axes.

### Scope breadth

`TASK_LOCAL -> STRUCTURAL_CLOSURE -> RELATED_TASKS -> SESSION_GLOBAL`

### Materialization depth

`INDEX/GIST -> COMPACT -> SELECTED_DETAIL -> FULL/NATIVE_KV`

Primary experiment should test whether the same active-KV/token budget is better spent:
- broadly across many tasks at shallow detail;
- narrowly on structurally relevant tasks at deeper detail.

Do not make one opaque “context size” variable.

## 4. Task completion is context demotion

Do not describe task completion merely as summarization/compression.

Use:

`execution history -> compact completed-task result + retrieval addresses + exact backing state`

The completed task is demoted from hot active state while preserving:
- provenance;
- evidence refs;
- result refs;
- retrieval addresses;
- exact backing records.

Paper 7 owns later selective expansion.

## 5. Canonical TaskState

Prefer one projected authoritative state object:

```text
TaskState {
  task_id
  version
  status
  description
  constraints[]
  parent_task_id?
  depends_on[]
  after[]
  blocker_refs[]
  evidence_refs[]
  output_refs[]
  result_ref?
  completion_condition?
  created_seq
  updated_seq
}
```

Status v1:

`pending | active | blocked | completed | cancelled`

Prefer `result_ref` to embedding a large `result_summary` in authoritative task state.

The result reference points to a Paper-7 typed record.

## 6. Structural closure

Define explicitly:

`Closure(T) = {T, parent(T), unresolved dependencies(T), blockers(T), required evidence/results(T)}`

Bound relation depth if needed.

Important:
- siblings are NOT automatically in closure;
- unrelated completed tasks are NOT in closure;
- related/session-global retrieval is a later fallback.

Primary causal comparison:
- TASK_LOCAL;
- TASK_STRUCTURAL.

This tests whether hard isolation loses legitimate dependency evidence and whether explicit structure restores it without reopening the session.

## 7. Deterministic task transitions first

Oracle phases receive harness events:

- TASK_ACTIVATE
- TASK_BLOCK
- TASK_RESUME
- TASK_COMPLETE
- TASK_CANCEL

These deterministically trigger:
- task scope recomputation;
- mandatory-set update;
- working-set promotion/demotion;
- cache/materialization update.

Do not require model callbacks for the core mechanism experiment.

## 8. Task-structure acquisition is a separate problem

Do not assume the execution model reliably calls task-management tools.

Evaluate task acquisition separately.

Required modes:

### T0 ORACLE
Correct task graph supplied by benchmark/harness.

### T1 SINGLE
Entire request remains one task.

### T2 PREFLIGHT_JSON
Harness performs optional two-pass decomposition before execution using JSON/schema-constrained output.

### T3 PREFLIGHT_MD
Same, but model writes a simple Markdown task grammar parsed deterministically.

### T4 ONLINE_TOOLS
Execution model creates/updates tasks during execution.

### T5 HYBRID
Preflight graph initializes tasks; online task tools may mutate graph later.

The core task-aware PRA claim must pass under T0 before task-generation quality is studied.

## 9. Preflight complexity gate

Two-pass mode:

`request -> complexity gate -> optional decomposition model call -> task records -> normal execution`

If gate says simple:
- create one task;
- do not pay second round trip.

If gate says complex:
- run decomposition before long generation/tool use.

Start with simple deterministic signals or a small classifier:
- multiple deliverables;
- explicit lists;
- conjunctions;
- sequencing language;
- dependency markers;
- prompt length/structure.

Measure:
- gate precision/recall;
- false-positive extra planning calls;
- false-negative missed decomposition;
- cost per downstream success.

Do not over-engineer the gate before oracle/preflight value is established.

## 10. Preflight JSON mode

Use schema-constrained output where supported.

Validate:
- unique task IDs;
- valid dependency IDs;
- acyclicity;
- required description;
- no self-dependency;
- valid statuses/relations;
- constraints/result-flow fields.

Malformed output must not silently become authoritative state.

Record parse/validation failures.

## 11. Preflight Markdown mode

Provide a deliberately simple grammar.

Example:

```text
## Task 1
Description: ...
Depends on: none

## Task 2
Description: ...
Depends on: Task 1
```

Parse deterministically.

The goal is to test whether models that struggle with JSON can still create useful task structure.

Do not assume JSON is always superior.

## 12. Online model-facing task tools

Candidate set:

- task_create
- task_update
- task_link
- task_complete
- task_get_state

Use `task_get_state`, not ambiguous `task_query`, for authoritative workflow inspection.

PRA context retrieval should remain PRA's job; do not introduce a semantic task retrieval tool merely to fetch task context.

Task tools should be always-visible/fixed system capabilities if model/tool experiments require them.

## 13. Workflow complexity is a primary independent variable

Do not treat “number of tasks” as sufficient.

Generate/evaluate workflow families:

### W0 Atomic
One task.

### W1 Linear sequential
`T1 -> T2 -> ... -> Tn`

### W2 Parallel independent
`{T1,...,Tn}`, no dependency edges.

### W3 Fork
One predecessor fans out to multiple children.

### W4 Join
Several predecessor results are required by one downstream task.

### W5 General DAG
Multiple forks/joins and arbitrary acyclic dependencies.

Record:
- `N_tasks`;
- `N_edges`;
- critical-path depth `D`;
- maximum parallel width `W`;
- join count `J`;
- fork count `F`;
- records/steps per task;
- total execution steps.

## 14. Result joins are mandatory

Join workflows are central because pure task-local filtering is insufficient.

For a downstream join task, structural closure should select all required predecessor outputs while excluding unrelated completed-task history.

Measure:
- join-input recall;
- join completeness;
- irrelevant predecessor contamination;
- active KV/tokens.

This is one of the strongest expected demonstrations of explicit task relations.

## 15. Interleaving complexity

Independently vary:
- task-switch frequency;
- resumption distance in records;
- resumption distance in tokens;
- semantically similar task count;
- shared entities;
- records/task;
- delayed dependency outputs;
- completed-task history volume;
- hot previous-task state.

Do not correlate all complexity dimensions in one generator. Use controlled factorial/sliced experiments.

## 16. Hot wrong-task contamination

Mandatory experiment.

Task A is recently active and has detailed/native K/V hot.

Switch to semantically similar Task B.

Compare:
- ordinary PRA over full session;
- task-aware PRA.

Measure whether A's hot state contaminates B merely because it is already resident.

Metrics:
- cross-task attention/materialization;
- wrong-task record recall;
- answer/action error;
- demotion cost.

## 17. Cold/warm/hot resumption

For old task resumption distinguish:

### Cold
Backing state exists; native encoding absent.

### Warm
Native encoding cached but inactive.

### Hot
Relevant K/V remains accelerator resident.

Measure:
- TTFT;
- K/V reused;
- K/V promoted;
- K/V demoted;
- H2D/transfer if applicable;
- resumed-task recall/success.

Do not average these states together.

## 18. Task resumption curve

Headline curve:

x:
`intervening task records/tokens/switches`

y:
`resumed-task evidence recall` and/or `task success`

Compare at matched active-context budget:
- PRA_SESSION;
- PRA_TASK_LOCAL;
- PRA_TASK_STRUCT;
- PRA_TASK_ADAPTIVE.

Expected task-aware curves should degrade more slowly with distance if the thesis holds.

## 19. Main PRA baselines

Reader-facing systems:

### FULL / TRUNCATION
Controls.

### PRA_SESSION
Ordinary PRA over the complete interleaved session.

### PRA_TASK_LOCAL
Only active-task candidates.

### PRA_TASK_STRUCT
Active task + structural closure.

### PRA_TASK_ADAPTIVE
Structural closure plus controlled widening:
`task -> dependencies/parent -> related tasks -> session global`

Do not reopen Paper-7 controller calibration here. Use a frozen inherited materialization policy.

Keep tags-ignored, routing variants, and relation/materialization ablations secondary.

## 20. Task facet

Test a persistent task facet `q_task` alongside local token/query facet `q_token`.

But do not make learned task embeddings responsible for relations already known exactly.

Known dependencies/parents/blockers should be structural channels.

Semantic PRA should operate within or after structural scoping.

## 21. Structure acquisition metrics

For PREFLIGHT/ONLINE/HYBRID report:
- decomposition-needed gate precision/recall;
- task count error;
- missing task rate;
- duplicate task rate;
- dependency edge precision/recall/F1;
- fork/join recovery;
- critical-path/depth error;
- constraint coverage;
- output/result-flow preservation;
- JSON/Markdown parse failure;
- validation rejection;
- extra model calls;
- planning input/output tokens;
- planning latency;
- downstream task-aware PRA success.

Do not require exact graph identity as the sole success criterion.

A different decomposition can be functionally valid.

## 22. Core task-aware PRA metrics

Quality:
- relevant-record recall/precision;
- cross-task contamination;
- dependency/evidence recall;
- join completeness;
- fallback rate;
- task completion;
- required-action recall;
- constraint errors;
- dependency errors;
- stale-state errors.

Systems:
- active native tokens/KV;
- backing/CPU bytes;
- peak GPU memory;
- K/V promotion/demotion/reuse;
- task-switch TTFT;
- resumption TTFT;
- routing latency;
- materialization latency;
- tokens/FLOPs/cost per successful task/session.

Paper-7 interaction:
- compact/full expansion count;
- evidence recovery after demotion/expansion.

## 23. Complexity curves

Required curves where feasible:

1. task count `N_tasks` vs contamination/success;
2. dependency depth `D` vs dependency recall/success;
3. maximum parallel width `W` vs contamination;
4. join count/fan-in vs join completeness;
5. resumption distance vs recall;
6. completed history size vs active KV;
7. task switches vs task-switch latency;
8. scope breadth x detail depth quality/cost frontier.

Do not report only one aggregate benchmark number.

## 24. Scope x detail budget experiment

Paper 8 inherits Paper-7 materialization.

Compare matched total budgets across configurations like:

- narrow scope + deeper detail;
- structural scope + moderate detail;
- global scope + shallow/index detail.

Question:

Is explicit task structure a better way to spend the same PRA/KV budget?

This is a major systems experiment.

## 25. Oracle-first causal order

Mandatory order:

1. T0 ORACLE + PRA_SESSION/TASK_LOCAL/TASK_STRUCT/TASK_ADAPTIVE.
2. Complexity scaling under oracle structure.
3. Metadata corruption/staleness robustness.
4. Preflight JSON/Markdown.
5. Online task tools.
6. Hybrid.

Do not jump directly to model-managed tasks.

## 26. Metadata robustness

Test:
- wrong task tag;
- missing task tag;
- stale task version;
- missing dependency edge;
- spurious dependency edge;
- delayed result ref;
- task status mismatch.

Measure degradation and fallback behavior.

If task awareness is brittle to small metadata errors, weaken claims.

## 27. Relationship to Paper 6.5

Task state may condition which tool/skill palette is relevant, but do not reopen large-catalog tool retrieval in Paper 8.

Use frozen Paper-6.5 capability mechanisms.

Possible experiment:
- same tool catalog;
- task-conditioned candidate scope;
- measure candidate/tool context savings and wrong-task tool contamination.

Keep secondary unless it materially supports the task-scope thesis.

## 28. Relationship to Paper 7

Paper 7 owns:
- compact vs full typed record views;
- backing originals;
- addresses;
- selective materialization;
- cursor/search expansion;
- model/context escalation policy.

Paper 8 must not duplicate these.

Paper 8 owns:
- task-scope selection;
- task relations;
- task-lifecycle-driven promotion/demotion;
- interleaving/resumption.

Use frozen Paper-7 policies for primary Paper-8 experiments.

## 29. Implementation phases

### A. Substrate inheritance
Verify Paper-7 SDK integration.

### B. Oracle task protocol
TaskState, task provenance, dependency graph, replay/event protocol.

### C. Task-aware scope
TASK_LOCAL and structural closure.

### D. PRA systems
PRA_SESSION, PRA_TASK_STRUCT, PRA_TASK_ADAPTIVE under matched budgets.

### E. Lifecycle
Activation/block/resume/completion demotion; cold/warm/hot states.

### F. Complexity benchmark
Atomic, linear, parallel, fork, join, DAG; interleaving and resumption ladders.

### G. Preflight decomposition
Complexity gate + JSON + Markdown modes.

### H. Online task tools
Evaluate only after preflight/oracle mechanisms are understood.

### I. Hybrid
Preflight + online mutation.

## 30. Required benchmark artifacts

Create:
- `task_workflow_cases.jsonl`
- `task_graphs_oracle.jsonl`
- `task_complexity_manifest.csv`
- `task_interleaving_manifest.csv`
- `task_record_provenance.csv`
- `task_scope_results.csv`
- `task_structural_closure_results.csv`
- `task_resumption_results.csv`
- `task_hot_contamination_results.csv`
- `scope_detail_frontier.csv`
- `preflight_gate_results.csv`
- `preflight_json_results.csv`
- `preflight_markdown_results.csv`
- `online_task_tool_results.csv`
- `hybrid_task_results.csv`
- `task_structure_downstream_utility.csv`
- `task_switch_costs.csv`
- `task_cache_state_results.csv`

## 31. Required tests

Add tests for:
- stable task IDs;
- task state versioning;
- idempotent replay;
- DAG acyclicity;
- dependency validation;
- structural closure correctness;
- sibling exclusion;
- join predecessor inclusion;
- task provenance filtering;
- task completion demotion;
- result_ref recovery;
- cold/warm/hot state transitions;
- hot wrong-task demotion;
- scope widening;
- stale metadata handling;
- JSON decomposition parser/schema;
- Markdown decomposition parser;
- hybrid mutation validation.

Then run full suite.

## 32. Main hypotheses

H1 Interleaving hurts PRA_SESSION faster than task-aware PRA.

H2 Task scope reduces contamination without losing required evidence.

H3 Structural closure beats hard local scope on dependency/join workflows.

H4 Gains increase with semantically similar and hot distractor tasks.

H5 Task completion is a useful demotion boundary when backing addressability is preserved.

H6 Explicit relations beat semantic rediscovery of known dependencies.

H7 Task-aware active working set remains bounded/sublinear as history grows.

H8 Benefits increase with workflow complexity: task count, depth, width, joins, resumption distance, and interleaving.

H9 Preflight decomposition recovers substantial oracle benefit for models that do not reliably manage tasks online.

H10 JSON vs Markdown planning reliability is model-dependent; downstream utility is the decisive metric.

H11 Hybrid preflight+online mutation helps only if online changes add real execution-discovered structure.

## 33. Falsification

Weaken the thesis if:
- PRA_SESSION matches task-aware variants at matched budget under strong interleaving;
- structural closure expands close to session-global in realistic DAGs;
- hard/task-aware filtering loses required dependency evidence;
- joins routinely fail despite explicit relations;
- active KV does not stay bounded as history/tasks grow;
- completion demotion harms later task use;
- task switching overhead approaches savings;
- hot distractor K/V remains contaminating despite task scope;
- metadata errors make the mechanism brittle;
- preflight planning cost exceeds downstream savings;
- generated task structures do not improve PRA;
- gains occur only with synthetic labels.

## 34. Claim discipline

Do not claim invention of:
- task lists;
- workflow DAGs;
- decomposition prompting;
- planning;
- agent task tools.

Target claim:

**Explicit typed task state and workflow relations can serve as a first-class context-scope signal, allowing PRA to reconstruct bounded logical task contexts from an interleaved single-session stream while reusing ordinary PRA discovery and progressive materialization and preserving authoritative execution state in the harness.**

## 35. Stop gate for first empirical iteration

Before model-managed task tools, require:
- oracle task protocol implemented;
- workflow complexity benchmark built;
- PRA_SESSION/TASK_LOCAL/TASK_STRUCT compared;
- join workflows tested;
- resumption curve measured;
- hot wrong-task contamination measured;
- active-KV/task-switch accounting complete;
- scope x detail frontier measured;
- structural closure shows a meaningful quality/cost advantage or is falsified.

Only then proceed to preflight and online task acquisition.
