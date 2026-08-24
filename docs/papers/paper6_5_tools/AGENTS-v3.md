# AGENTS.md — PRA Paper 6.5: Persistent Agent Context with Tools and Skills

## Mission
Test a systems-level consequence of PRA: an LLM agent session need not be represented as an ever-growing append-only prompt. A bounded native workspace can operate over persistent, typed, addressable memory containing tools, skills, system instructions, history, artifacts, clipboard objects, and eventually parent/child session state.

Working title: **Persistent Agent Context with Progressive Retrieval Attention: Addressable Tools, Skills, History, and Hierarchical Sessions**

This is not a proposal for another general-purpose agent framework. Build a controlled mini harness for experiments, then integrate through the PRA OpenAI-compatible serving endpoint into an established open-source harness.

## Dependencies in the PRA line
Paper 6.5 should consume, not duplicate:
- Paper 1/1.5: native-KV transport and positional correctness;
- Paper 2: pretrained-family integration;
- Paper 2.5/2.6: iterative and hybrid reference discovery;
- Paper 3/3.5: materialization and adaptive effort;
- production/productization paper: persistent cache/residency, batching, and OpenAI-compatible endpoint.

Keep the numbering “6.5” provisional if the repository roadmap changes.

## Core hypotheses
H1. Tool/skill catalogs can grow much faster than native prompt context while selection/task quality remains stable.
H2. Explicit and approximate resource references benefit from hybrid symbolic/token-native/semantic discovery, reusing Paper 2.6's token-aware resolver and calibrated referent confidence.
H3. Stable tool/skill definitions should be encoded/cached once and selectively materialized rather than repeatedly tokenized in every turn.
H4. `#__head` plus persistent history can bound active native context as a project session grows.
H5. Compaction can become an optional index/summary operation rather than destructive replacement of history.
H6. Subagents can inherit selected parent memory by reference with much lower transfer cost than copying full prompts.
H7. Transparent PRA behind an OpenAI-compatible endpoint can improve context economics with minimal harness modification; PRA-native references may improve it further.

## Resource model
Support typed, versioned identities such as:
- `!!ref:tool:<namespace>:<name>:<version>!!`
- `!!ref:skill:<namespace>:<name>:<version>!!`
- `!!ref:clipboard:<id>!!`
- `!!ref:artifact:<id>!!`
- `!!ref:system-prompt:<version>!!`
- `!!ref:session:<session>/<object>!!`
- implicit `#__head`.

Separate identity from cached representation. Cache keys must include content/version, model, tokenizer, routing configuration and relevant positional encoding fingerprints.

## Reference-resolution ladder
Evaluate:
1. explicit typed URI — deterministic resolution;
2. exact resource name;
3. approximate lexical/token/alias match;
4. semantic description;
5. contextual reference to previously used resource.

Preserve discovery provenance and calibrated referent confidence from Paper 2.6. Confidence must support downstream decisions rather than merely ranking candidates:

- high confidence -> select/materialize, and only then consider execution;
- intermediate confidence -> search/disambiguate/ask;
- low confidence -> abstain.

For side-effecting tools, selection confidence is necessary but not sufficient for execution authorization.

## Phase A — controlled catalog benchmark
Build a minimal OpenAI-style agent loop with generated and curated tools/skills.

Scale catalogs across roughly 8, 32, 128, 512, 2K and, if feasible, 8K+ resources. Vary:
- definition/schema length;
- number of parameters;
- name similarity;
- semantic similarity;
- aliases;
- distractor families;
- duplicate-looking tools;
- skill depth/supporting files.

Prompt classes:
- explicit URI;
- exact name;
- partial/fuzzy name;
- semantic paraphrase;
- ambiguous reference;
- multi-tool task;
- tool + skill task;
- wrong/nonexistent requested resource.

Separate **resource selection** from **argument generation/execution** so weak small-model argument filling does not obscure the memory question.

## Baselines for Phase A
- all tool/skill definitions eagerly in prompt;
- native provider/tool schema interface where comparable;
- harness-side lexical retrieval;
- harness-side embedding/semantic retrieval;
- progressive-disclosure/skill-description baseline;
- transparent PRA with ordinary prompts;
- PRA-native explicit references;
- hybrid PRA from Paper 2.6;
- oracle selected definitions.

Match model, task and active-definition budgets where meaningful.

## Phase B — persistent-session benchmark
Create long synthetic and realistic project trajectories with stable facts, changed facts, artifacts, tool use, decisions, and revisited earlier requirements.

Compare:
- append-only until context limit;
- truncation/sliding window;
- periodic lossy compaction;
- summary + conventional retrieval;
- PRA bounded `#__head` + persistent history.

Scale by turns and total historical tokens far beyond the native context where feasible.

Measure whether the model can recover old requirements, distinguish superseded from current state, and preserve provenance.

## Phase C — mutation, freshness and invalidation
Agent memory is mutable. Test:
- tool schema version changes;
- skill updates;
- revoked resources;
- superseded project decisions;
- deleted artifacts;
- alias collisions.

A stale cached K/V hit must never silently masquerade as current state. Add versioned invalidation tests before making persistent-session claims.

## Phase C.1 — Historical-reference robustness and conflict
Extend wrong-reference stress testing from tools to persistent history. Construct sessions containing:

- repeated entity/project names at different times;
- superseded requirements;
- contradictory old/new decisions;
- renamed artifacts;
- copied branches with similar content;
- summaries that omit a decisive detail;
- stale cached representations that remain physically resident.

Measure whether the resolver retrieves the current authoritative state, preserves access to historical provenance, and assigns lower confidence to ambiguous/superseded references. Report stale-memory use separately from ordinary retrieval misses.

## Phase D — session trees / subagents
Model a child session as:
`M_child = R(M_parent) ∪ ΔM_child`
where `R` is a permission-scoped read view and `Δ` is child-local state.

Compare:
- no inherited context;
- manually written delegation prompt;
- compacted parent summary;
- copied full relevant history;
- PRA shared references.

Measure child task success, prompt-transfer tokens, parent-memory materialization, latency, and merge-back cost. Add ambiguous-parent-reference cases and measure whether the child searches/asks/abstains rather than confidently inheriting the wrong branch or stale decision. Keep writes explicit; do not let a child mutate parent memory implicitly.

## Phase E — real harness integration
Prefer OpenAI-compatible integration first:
`existing harness -> OpenAI-compatible PRA endpoint -> PRA runtime/model`

Two modes:
1. **Transparent PRA:** harness sends ordinary OpenAI-like messages/tools; server discovers/cache-manages resources without PRA-specific syntax where possible.
2. **PRA-native:** harness explicitly supplies stable resource/session references.

Select the real harness at experiment time based on maintenance, reproducibility, tool/skill support, and OpenAI-compatible model endpoint support. Do not fork a large harness unless the endpoint boundary proves insufficient.

## Measurements
Quality: resource-selection accuracy/recall/MRR, task success, argument validity, old-fact recovery, supersession accuracy, subagent success, wrong-selection severity, and recovery after incorrect candidate selection.

Confidence/safety: Brier score, ECE/reliability, selective accuracy, risk-coverage curves, ask/abstain rate, false-act rate, and confidence conditioned on discovery provenance and resource side-effect class.

Context: native prompt tokens, tool-schema tokens, skill tokens, direct-history tokens, logical-history tokens, requested/materialized tokens, active fraction.

Systems: tokenize/encode time, cache hit rate, cold/warm latency, routing/index time, H2D transfer, attention time, throughput, peak GPU memory, host memory, cache bytes.

Session: compaction count, information-loss failures, stale-memory failures, number of rollovers, history length, session duration.

Delegation: bytes/tokens copied to child, shared-reference count, child materialization, merge-back size.

Report both logical memory growth and active native working-set growth. The main systems hypothesis is bounded active context, not zero storage cost.

## Security and isolation
Treat reference resolution as privileged. Tests must cover:
- batch-row isolation;
- session/tenant isolation;
- inaccessible references;
- revoked references;
- forged URI attempts;
- stale version IDs;
- cache-key collisions.

The host remains responsible for authentication, permissions, side effects, tool execution and durable object storage. PRA supplies model-local discovery/materialization.

## Two-level success gates
### Gate A — context/discovery systems result
A positive PRA-agent-context result requires equal or better resource/history retrieval quality at materially lower native-context/materialization cost, or better quality at matched cost, with statistical support and survival under tokenization, ambiguity, stale-state, and isolation stress tests.

### Gate B — end-to-end agent result
A headline agent-capability result additionally requires improved end-to-end task success, lower failure cost, or better long-session/subagent continuity on at least one realistic benchmark. If Gate A passes but Gate B does not, report a context/discovery systems result rather than improved agent capability.

### Gate C — safe selective action for side-effecting tools
Any claim about side-effecting tool use requires evidence that calibrated selection plus ask/abstain/execution gating controls false actions. Better top-1 tool retrieval alone is insufficient.

## Critical falsification criteria
Narrow or reject the thesis if:
- tool/skill retrieval failures erase prompt savings;
- standard harness retrieval/compaction matches quality at lower cost;
- persistent cache construction/reuse does not amortize;
- long PRA sessions accumulate unacceptable stale/conflicting-memory failures;
- subagent shared references do not improve transfer economics or contextualization;
- OpenAI-compatible integration requires so much harness-specific logic that the claimed boundary disappears.

## Codex phases
0. Audit production-serving/OpenAI endpoint and current `#__head` behavior.
1. Build deterministic tool/skill catalog generator and selection-only benchmark.
2. Add PRA resource registration/cache adapter.
3. Run catalog scaling, tokenization-robustness, ambiguity, and confidence-calibration experiments.
4. Add argument generation/execution tasks with explicit side-effect classes and execution gates.
5. Add persistent-session benchmark and rollover, including contradictory/superseded history.
6. Add mutation/invalidation and stale-confidence tests.
7. Add session-tree prototype with ambiguous inheritance cases.
8. Integrate one maintained open-source harness.
9. Run matched end-to-end comparisons, risk-coverage analysis, and update paper.

Every phase must leave machine-readable traces and reproducible scripts.

## Paper discipline
Do not call storage “free.” Distinguish prompt tokens, encoded source tokens, resident K/V, transferred K/V, and active attention K/V. Do not claim never-ending sessions from finite experiments; claim bounded native working-set behavior over measured session lengths. Separate selection accuracy from execution competence and transparent integration from PRA-native integration.
