# Paper 9 — Selectively Permeable Subagent Contexts with PRA

## Status

**Inception / research plan.**

Paper 9 starts after Paper 8 has a stable task-aware agent harness and typed/compacted tool-result representation. The purpose of Paper 9 is **not** to invent a novel way to spawn subagents. The harness should initially use the ordinary and widely deployed mechanism: the parent LLM emits a subagent/delegation tool call, the harness creates a child agent loop/session, and the child executes with configurable subagent semantics.

The research contribution is on the PRA side: representing agent lineage and subagent context streams explicitly, preserving compacted records as addressable state, and making context boundaries selectively permeable so that valid ancestor/descendant records—and where possible their native model K/V state—can be reused without copying or replaying whole histories.

---

## Relationship to the PRA series

Paper 9 depends on mechanisms and evidence developed across the PRA roadmap, especially Papers 7 and 8.

### Earlier foundations

- **Papers 2.x**: semantic/lexical/hybrid routing, chunk selection, retrieval quality, multi-gist and graph-aware selection experiments, and the principle that logical context can exceed materialized physical context.
- **Papers 3.x**: adaptive/progressive context control, retry/select/materialize policies, summary/index variants, and policy-level decisions about what representation to keep active versus externally addressable.
- **Paper 4.5**: practical SDK, agent/gateway/engine integration, deployment modes, coding-agent evaluation, and separation between agent-side behavior, gateway behavior, and engine-side PRA support.
- **Paper 5 / 6.x**: persistent PRA records/resources, externalized context, lifecycle and tooling infrastructure, typed/versioned resources, safe execution, scalable tools/skills, and persistent sessions.
- **Paper 7**: typed adaptive context and structured records. Provides the generic record substrate and progressive materialization model that Paper 9 extends across multiple agents.
- **Paper 8**: task-aware agent harness, tool-call/tool-result records, task records, compaction of tool outputs, task-sensitive materialization and storage policies, and a clean separation between harness state and PRA runtime state within a single session.

### Paper 9 extension

Paper 9 adds:

1. first-class `agent_uuid` and agent lineage metadata;
2. parent/child subagent lifecycle records;
3. multiple agent context streams represented as a tree or DAG;
4. configurable visibility across stream boundaries;
5. generic side-effect/resource metadata for validating reuse;
6. cross-agent tool-result reuse;
7. cross-agent native K/V reuse where supported by the engine integration;
8. selective post-hoc reading/materialization from completed subagents;
9. consistency rules for sequential and parallel subagent execution.

Paper 9 should deliberately **not** absorb Paper 10 cross-session memory. The scope is multiple agents/subagents belonging to the same higher-level session or run.

---

# Core thesis

Conventional subagent systems use hard context boundaries:

```text
parent context
   |
   | delegate(task)
   v
isolated child context
   |
   | final answer / summary / artifacts
   v
parent context
```

This isolation is useful but creates a lossy boundary. The parent usually cannot selectively recover the child’s detailed context without replaying it, and the child usually cannot directly reuse detailed parent context unless the harness copies or restates it.

Paper 9 studies a different abstraction:

> **Subagent contexts remain isolated execution streams, but their records are selectively permeable through PRA. Valid records can be referenced and materialized across agent boundaries without copying entire histories.**

A subagent is therefore still an ordinary harness-level agent, but PRA represents the execution topology as an addressable context graph.

---

# Architectural principle: strict separation of responsibilities

This is critical. Do **not** make the PRA runtime understand coding-agent-specific concepts such as `file_read`, `file_write`, `grep`, Git, shell commands, IDE operations, or a particular agent framework.

The architecture should remain layered.

## 1. PRA Agent Harness

The PRA agent harness is responsible for agent behavior and tool semantics.

It should initially behave like a conventional subagent harness:

```text
Parent LLM
  -> tool call: spawn_subagent(...)
Harness
  -> allocates child agent
  -> chooses child model / tools / workspace / context policy
  -> starts child loop
Child LLM
  -> tool calls
  -> observations
  -> completion
Harness
  -> returns summary/result/artifacts to parent
```

The harness additionally emits typed PRA records for important lifecycle events.

Responsibilities:

- expose `spawn_subagent`, `resume_subagent`, `stop_subagent`, `inspect_subagent`, etc.;
- maintain configurable subagent semantics;
- assign child `agent_uuid` and task identity;
- select model/tool profile/permissions/workspace;
- interpret concrete tool semantics;
- attach generic effect/resource metadata to tool descriptors and results;
- emit subagent lifecycle records exactly as it emits ordinary tool-call/tool-result/task records;
- decide what context references to make visible to a child;
- receive reuse/materialization decisions from the PRA runtime;
- enforce execution permissions and unsafe side-effect boundaries.

The harness is the place where `file_read`, `file_write`, SQL, HTTP, Git, or application-specific tools are known.

## 2. PRA Runtime

The PRA runtime may often be a **remote service** relative to the harness. It therefore must remain generic and transport-friendly.

The PRA runtime should understand only generic record/context semantics:

- sessions;
- agents and parent/child lineage;
- tasks;
- typed records;
- record visibility;
- resource identities;
- generic effects (`PURE`, `READ`, `WRITE`, `UNKNOWN` initially);
- dependencies;
- validity metadata;
- reuse policy;
- timestamps / logical ordering / causal ordering;
- stream topology;
- materialization and routing policy;
- compacted versus native/detail representations;
- engine capabilities such as native K/V persistence/reuse.

The runtime should **not** need code paths such as:

```python
if tool.name == "file_read": ...
if tool.name == "git_diff": ...
```

Instead the harness translates a concrete tool into generic metadata.

## 3. PRA Engine integration

The engine integration handles model-near operations:

- native K/V storage;
- K/V identifiers and lifecycle;
- per-layer materialization;
- routing over candidate records/chunks;
- reuse of already encoded records across agent streams;
- model-specific positional/RoPE handling;
- bounded active context;
- cache residency/eviction.

The engine does **not** spawn subagents and should not understand agent workflows beyond metadata needed to index and materialize records.

---

# Harness-level subagent semantics

Implement these incrementally and expose capability status explicitly.

```text
Dimension              Initial / planned options
--------------------  ----------------------------------------------------
Child context          empty | selected | parent_snapshot | logical_fork
Child model            same | cheaper | specialist | explicit
Tools                  inherited | restricted | specialist | explicit
State                  ephemeral | persistent | resumable
Execution              synchronous | asynchronous
Scheduling             sequential | parallel | DAG
Return                  final | summary | artifacts | typed_records
Communication          child->parent | bidirectional | peer-to-peer
Workspace              shared | isolated | worktree/branch | explicit
Lifetime               one-shot | resumable
Recursion               prohibited | bounded_depth | unrestricted
Ancestor visibility    none | selected | routable | inherited
Descendant visibility  none | completed_only | routable
Sibling visibility     none initially; experimental later
```

Do not block the paper on implementing all semantics. Add a capability table in the SDK and paper:

```python
SubagentCapabilities(
    spawn="supported",
    resume="supported",
    parallel="experimental",
    dag="not_implemented",
    ancestor_record_visibility="experimental",
    descendant_record_visibility="experimental",
    cross_agent_result_reuse="experimental",
    cross_agent_kv_reuse="experimental",
)
```

---

# Record model additions

All PRA records should be extended with agent identity.

```text
RecordMetadata
  record_uuid
  session_uuid
  agent_uuid
  task_uuid?              optional
  created_at
  logical_clock?          preferred
  parent_record_uuid?     optional
  type
  representation
```

Do not assume a single agent per session anymore.

## Agent lifecycle records

### `AgentStartRecord`

```text
agent_uuid
session_uuid
parent_agent_uuid?        null for root agent
parent_task_uuid?
spawn_call_record_uuid?
model_descriptor
workspace_id?
tool_profile_id?
context_policy
start_time
```

### `AgentStopRecord`

```text
agent_uuid
status
stop_time
summary_record_uuid?
result_record_uuids[]
artifact_record_uuids[]
```

### Optional `AgentResumeRecord`

Use only if persistent/resumable children are implemented.

---

# Generic tool effect model

Paper 9 needs enough semantics to reason about validity without embedding tool-specific knowledge in PRA.

Start deliberately small:

```text
EffectType = PURE | READ | WRITE | UNKNOWN
```

- `PURE`: deterministic/local computation with no mutable external dependency, subject to descriptor constraints.
- `READ`: observes one or more resources.
- `WRITE`: mutates one or more resources and may invalidate previous reads.
- `UNKNOWN`: conservative default; no cross-agent reuse unless explicitly overridden.

Avoid a large effect calculus in the first implementation.

---

# Resource model

A tool descriptor may declare a resource domain and a way to derive a resource identity from arguments.

Example:

```yaml
name: file_read

effects:
  type: READ
  resource_domain: os.FILE
  resource_key:
    args: [path]

reuse:
  enabled: true
  max_age_seconds: 300
  consistency: ancestry
  validation: stat
```

Example mutation:

```yaml
name: file_write

effects:
  type: WRITE
  resource_domain: os.FILE
  resource_key:
    args: [path]

invalidates:
  - domain: os.FILE
    match: same_resource
```

The PRA runtime receives normalized metadata such as:

```text
resource.domain = os.FILE
resource.key    = /repo/src/parser.py
```

It does not know that this is a file path semantically.

---

# Tool-result dependency metadata

Results may depend on one or multiple resources.

A simple read:

```text
ToolResultRecord
  effect = READ
  dependencies = [ os.FILE:/repo/src/parser.py ]
```

A grep/search over a directory may depend on a coarser resource:

```text
dependencies = [ os.FILE_TREE:/repo/src ]
```

or, if available, a resolved set:

```text
dependencies = [
  os.FILE:/repo/src/a.py,
  os.FILE:/repo/src/b.py,
  os.FILE:/repo/src/c.py,
]
```

Prefer exact identities where practical, but allow coarse-grained resource identities.

---

# Reuse model

There are three distinct levels of reuse. Measure them separately.

## Level 1 — tool execution reuse

A child asks for a tool operation whose valid result already exists in an ancestor-visible stream. The harness/runtime serves the previous result instead of invoking the external tool.

Savings may include:

- process launch;
- network round-trip;
- filesystem access;
- remote API latency;
- repeated command execution.

## Level 2 — record payload reuse

The previous tool-result record is referenced/materialized rather than copied/re-serialized into a new context stream.

Savings may include:

- serialization;
- transport;
- duplicated storage;
- repeated tokenization.

## Level 3 — native K/V reuse

If the engine already has native K/V for the record under compatible model/encoding conditions, materialize the stored K/V into the child’s logical context without re-prefill.

Potential savings:

- tokenization;
- model prefill;
- K/V reconstruction;
- repeated host->device transfer depending on placement.

This is the most PRA-specific mechanism.

---

# Validity and consistency

Do not claim reuse is safe merely because no write exists on the direct parent chain. Parallel agents and external processes complicate correctness.

Start with explicit consistency modes.

```text
ANCESTRY
  Only mutations in the child-visible ancestor causal chain invalidate.
  Suitable for isolated snapshots/worktrees or strongly controlled environments.

SESSION_TREE
  Any causally prior mutation in the relevant shared session/resource domain may invalidate.
  Suitable for a shared workspace managed exclusively by the harness.

EXTERNAL
  The resource may change outside PRA/harness control.
  Reuse requires explicit TTL/version/stat/etag/content-hash validation, or is disabled.
```

Potential validation mechanisms:

- TTL / `max_age`;
- mtime + size;
- inode/version tuple;
- content hash;
- ETag/version ID;
- application-provided version token;
- immutable workspace snapshot ID.

Use conservative invalidation by default.

---

# Context visibility model

Avoid defining subagent context as a full parent-copy.

A child context should be modeled as:

```text
child_local_records
+
explicitly inherited records
+
ancestor-visible addressable records
```

Possible visibility policies:

```text
NONE
SELECTED
ROUTABLE
INHERIT_ALL_REFERENCES
```

`ROUTABLE` is the most interesting PRA mode: ancestor records are not physically inserted into the child context, but are candidates for PRA routing/materialization when relevant.

Likewise, a parent may optionally route into completed descendants.

---

# Completed subagents

A completed subagent should become an effectively immutable context branch unless explicitly resumed.

The parent should be able to:

- retrieve the child summary;
- inspect typed result/artifact records;
- route into detailed child tool results;
- materialize evidence from the child’s context stream;
- inspect child decisions without replaying the whole transcript.

This is a central Paper 9 use case.

---

# Initial benchmark / dataset design

The benchmark must contain tasks where parent-child reuse is intentionally frequent enough to produce measurable performance differences. Do not rely only on arbitrary SWE-bench instances, because identical cross-agent reads may be too sparse to reveal the mechanism cleanly.

Create a controlled benchmark suite first, followed by naturalistic coding-agent validation.

## Benchmark name

Working name:

**Subagent Context Reuse Benchmark (SCRB)**

Alternative names:

- PRA-SAR: Subagent Ancestor Reuse
- CARB: Cross-Agent Reuse Benchmark
- PARBench: Permeable Agent Record Benchmark

Use `SCRB` unless a better name emerges.

---

# Dataset family A — Shared Repository Orientation

### Goal

Measure repeated reads of repository-wide context that many children need.

### Task construction

Each repository contains:

- `README.md`;
- `AGENTS.md` or contributor instructions;
- package/build config;
- architecture docs;
- shared interfaces/types;
- test conventions;
- several target modules.

The parent first performs an orientation pass and reads common files. It then spawns 2–8 children for independent subtasks.

Each child receives only its task description and ancestor visibility; without reuse it naturally issues the same `file_read`/`grep` operations.

### Example

Parent:

```text
Read project instructions and architecture, then delegate:
1. inspect parser bug
2. inspect serializer bug
3. inspect CLI bug
4. inspect tests
```

Likely common reads:

```text
AGENTS.md
pyproject.toml
src/core/types.py
src/core/interfaces.py
tests/conftest.py
```

### Metrics

- repeated tool calls avoided;
- repeated bytes avoided;
- prefill tokens avoided;
- K/V entries reused;
- wall-clock latency;
- time-to-first-useful-child-result;
- total model compute;
- answer/task correctness.

---

# Dataset family B — Shared Definition / Interface Fan-out

### Goal

Create high-value repeated reads of one or several large files.

### Construction

Create repositories where many task-specific modules implement or consume a common interface/specification.

Examples:

```text
large protocol schema
large AST definition
large API interface
large generated type file
large database model definition
large configuration schema
```

Parent reads the common definition once before delegation.

Children solve separate tasks that all require the same definition.

Vary shared-definition size:

```text
2K, 8K, 32K, 64K tokens
```

Vary children:

```text
2, 4, 8, 16
```

This family should produce a near-ideal upper bound on cross-agent K/V reuse.

---

# Dataset family C — Repeated Search/Grep

### Goal

Measure reuse for expensive or voluminous search results.

Parent performs searches such as:

```text
grep for symbol usages
search route registrations
search SQL table references
find interface implementations
search error code references
```

Children later issue equivalent searches.

Test:

- exact same query;
- semantically equivalent but syntactically different queries, initially no reuse unless canonicalized;
- same query after unrelated writes;
- same query after relevant writes.

This family directly tests invalidation correctness.

---

# Dataset family D — Read-after-unrelated-write

### Goal

Demonstrate that selective invalidation preserves reuse.

Sequence:

```text
parent reads A and B
child 1 writes A
child 2 reads B
```

Expected:

- A result invalidated;
- B result remains reusable.

Compare against a coarse baseline that invalidates the entire `os.FILE` domain.

Metrics:

- false invalidations;
- stale reuse errors;
- preserved cache hit rate.

---

# Dataset family E — Read-after-relevant-write

### Goal

Test safety.

Sequence:

```text
parent reads A
child 1 writes A
child 2 later reads A
```

Expected:

- parent result must not be reused under shared-workspace semantics;
- child 2 sees the updated content.

Include parallel and sequential schedules.

Measure stale-read rate; target must be zero for supported consistency modes.

---

# Dataset family F — Isolated worktrees / branch semantics

### Goal

Show that workspace semantics can increase safe reuse.

Parent reads common immutable base files. Each child operates in its own Git worktree/branch.

Ancestor reads from the common base may remain valid even when sibling agents mutate their own worktrees.

Compare:

1. shared workspace;
2. isolated worktree;
3. immutable snapshot.

Hypothesis: isolation greatly simplifies validity and increases cross-agent cache reuse.

---

# Dataset family G — Completed-child post-hoc inspection

### Goal

Measure parent access to detailed evidence from stopped subagents.

Pattern:

```text
parent delegates investigation to N children
children finish and return compact summaries
parent receives a synthesis question requiring evidence from one child
```

Examples:

- “Which exact line caused child 3 to reject approach X?”
- “Show the test output supporting child 2’s conclusion.”
- “Compare the exact API definitions inspected by children 1 and 4.”

Baselines:

1. summary only;
2. full transcript copied back;
3. transcript re-fetch/replay;
4. PRA descendant routing/materialization.

Measure correctness, parent context growth, latency, and tokens materialized.

---

# Dataset family H — Hierarchical delegation

### Goal

Test ancestor reuse across depth > 1.

```text
root
  -> child A
       -> grandchild A1
       -> grandchild A2
  -> child B
```

Shared repository instructions/interfaces originate in root.

Measure whether grandchildren can reuse valid root records through the lineage graph without copying root context into every branch.

Depth sweep:

```text
1, 2, 3, 4
```

---

# Dataset family I — Expensive remote tools

### Goal

Show that the mechanism is not filesystem-specific.

Use controlled mock or local services with tunable latency for:

- HTTP GET;
- database reads;
- package metadata lookup;
- static-analysis service;
- test-index service.

Configure deterministic resources and version tokens.

Inject latencies such as:

```text
10ms, 50ms, 200ms, 1s
```

Cross-agent reuse then produces an easily measurable wall-clock benefit.

This family validates the generic resource/effect abstraction.

---

# Dataset family J — Naturalistic coding-agent tasks

After controlled experiments, evaluate on existing coding-agent task sets.

Candidate sources:

- selected SWE-bench Verified tasks;
- Mini-SWE-agent reproducible subsets;
- repository-level bug-fix tasks used in Paper 4.5;
- custom multi-issue repository bundles;
- agent benchmark tasks where decomposition into parallel investigation is natural.

Do **not** expect high cache-hit rates automatically. Report observed reuse frequency honestly.

Add an instrumentation-only pass first to answer:

```text
How often do independently spawned subagents naturally repeat reads/searches already performed by ancestors?
```

This determines external validity of the controlled benchmark.

---

# Benchmark generation strategy

For controlled families, generate tasks from real open-source repository snapshots where licensing permits, but manipulate task orchestration rather than source content when possible.

Each task should include:

```text
repository snapshot ID
task DAG
root instructions
expected child subtasks
workspace policy
expected reusable resources
expected invalidation events
ground-truth final answer or patch/tests
```

Do not require agents to follow a fixed tool-call script for the main benchmark. Instead:

- provide task structure that makes reuse likely;
- instrument actual tool behavior;
- additionally run a scripted microbenchmark to establish upper-bound systems performance.

---

# Main experimental matrix

At minimum compare:

```text
A. Conventional subagents, isolated contexts
B. Conventional subagents + harness-level tool memoization
C. PRA records, no cross-agent visibility
D. PRA ancestor record visibility, payload reuse
E. PRA ancestor visibility + native K/V reuse
F. PRA + selective invalidation/resource tracking
```

Where practical also include:

```text
G. Parent-context copy/snapshot baseline
H. Full transcript sharing baseline
```

This separation is essential. Otherwise gains from ordinary tool-result caching could be incorrectly attributed to PRA.

---

# Key hypotheses

## H1 — Cross-agent result reuse

Tasks with repeated ancestor reads/searches will reduce tool executions and wall-clock latency when child agents can reuse valid ancestor records.

## H2 — Native K/V reuse

For sufficiently large reused records, native K/V reuse will reduce child prefill cost beyond harness-level memoization.

## H3 — Bounded context advantage

PRA reuse will preserve bounded active physical context while avoiding repeated prefill/copying of shared context into every child.

## H4 — Selective invalidation

Resource-level invalidation will preserve substantially more reusable state than domain-wide invalidation while maintaining zero stale-read errors under supported consistency assumptions.

## H5 — Completed-child permeability

A parent can answer post-hoc evidence questions more accurately and with lower context growth by routing into completed descendant records than by relying on child summaries alone.

## H6 — Scaling with fan-out

Benefits should grow with:

```text
shared context size
× number of children
× repeated-access probability
```

subject to cache residency and routing overhead.

---

# Analytical cost model

Include a simple model.

Let:

```text
S = tokens in reusable shared record set
N = number of child agents
p = probability a child reuses the set
C_prefill(S) = model cost to prefill S
C_tool = external tool cost
C_route = PRA lookup/routing cost
C_mat = materialization cost when K/V already exists
```

Without reuse, approximate duplicated cost:

```text
N * p * (C_tool + C_prefill(S))
```

With PRA reuse:

```text
N * p * (C_route + C_mat)
```

One-time parent cost remains.

Report break-even regions rather than claiming universal benefit.

---

# Metrics

Record at least:

### Correctness

- task success;
- patch/test success;
- stale-read count;
- invalid reuse count;
- post-hoc evidence QA accuracy.

### Agent/tool behavior

- tool calls per agent;
- duplicate tool calls;
- avoided tool calls;
- reads/searches reused;
- invalidations;
- false invalidations;
- cache hit/miss reason.

### Model cost

- logical tokens;
- physically materialized tokens;
- prefill tokens;
- decode tokens;
- reused K/V token-equivalents;
- per-layer K/V bytes reused if measurable.

### Runtime

- wall-clock latency;
- tool latency;
- routing overhead;
- materialization overhead;
- time-to-first-child-result;
- total session runtime.

### Storage

- duplicate payload bytes avoided;
- K/V storage overhead;
- cache residency;
- evictions.

---

# Instrumentation requirements

Every reuse decision should produce a structured trace record:

```text
ReuseDecisionRecord
  request_agent_uuid
  source_record_uuid
  source_agent_uuid
  resource_identity
  decision: HIT | MISS | INVALID
  reason
  validation_mode
  age
  causal_position
  kv_reused: bool
  payload_reused: bool
  tool_execution_avoided: bool
```

Miss reasons should be explicit:

```text
NO_MATCH
NOT_VISIBLE
EXPIRED
WRITE_INVALIDATION
EXTERNAL_VALIDATION_FAILED
MODEL_INCOMPATIBLE
KV_EVICTED
UNKNOWN_EFFECT
POLICY_DISABLED
```

This trace is required for scientific analysis.

---

# Implementation order

## Phase 0 — Paper-only simulator / trace analysis

Before changing the engine, instrument the Paper 8 harness to estimate natural repeated tool calls across parent/subagent traces.

## Phase 1 — Agent lineage records

Add `agent_uuid`, `AgentStartRecord`, `AgentStopRecord`, parent-agent linkage, and context-stream indexing.

## Phase 2 — Harness-level ancestor result reuse

Implement exact-match result reuse with conservative validity checks. No K/V reuse yet.

## Phase 3 — Resource-level invalidation

Add `READ/WRITE/UNKNOWN`, resource identity, TTL/version validation, and ancestry/shared-workspace consistency.

## Phase 4 — PRA routing across agent streams

Allow child routing into visible ancestor records and parent routing into completed descendant records.

## Phase 5 — Native K/V reuse

Reuse stored K/V for compatible records across agents.

## Phase 6 — Parallel/DAG semantics

Add causal ordering, sibling interaction rules, and shared-workspace invalidation.

## Phase 7 — Naturalistic coding-agent evaluation

Run selected Paper 4.5 / SWE-style agent tasks and report actual reuse rates and end-to-end benefit.

---

# Non-goals for first Paper 9 implementation

- cross-user reuse;
- cross-session long-term memory (Paper 10);
- arbitrary distributed cache coherence;
- automatic semantic equivalence of different tool calls;
- optimistic concurrency control across arbitrary hosts;
- full database transaction semantics;
- peer-to-peer subagent messaging as a primary contribution;
- novel subagent planning algorithms;
- training/fine-tuning for delegation unless needed only as a control.

---

# Safety and correctness defaults

1. Unknown side effects => no reuse.
2. Shared mutable external resources => validate or disable reuse.
3. Never trade correctness for cache-hit rate in the main reported system.
4. Always report the consistency assumptions of each experiment.
5. Tool descriptors remain declarative; harness plugins own concrete semantics.
6. Preserve an option to disable every cross-agent reuse optimization independently.

---

# Paper narrative

The paper should not be framed as “PRA adds subagents.” Existing agent systems already do subagents well enough for the baseline.

The narrative should be:

1. subagents are normally separate context streams;
2. hard context boundaries prevent efficient reuse and make summaries lossy;
3. Paper 8 already turns tool outputs/tasks into typed, compactable, addressable records;
4. Paper 9 extends that record model to agent lineage and multiple context streams;
5. declarative resource/effect metadata allows safe reuse without PRA learning tool-specific semantics;
6. PRA can selectively route/materialize records across agent boundaries;
7. native K/V reuse can avoid repeated prefill in addition to repeated tool calls;
8. the benefit is workload-dependent and is quantified with controlled and naturalistic benchmarks.

The strongest conceptual phrase is:

> **selectively permeable context boundaries**

A second useful phrase is:

> **subagent context as a graph of addressable record streams rather than isolated transcript blobs**

