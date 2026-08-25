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
H8. A bounded graph-expanded capability set may improve multi-step planning relative to Top-1 or reactive disclosure at matched total context cost.
H9. Category, family, tag, keyword, and directional schema relations may recover useful capability neighborhoods beyond similarity Top-K.
H10. Discovery policy and disclosure breadth are independent controls and must be evaluated separately.
H11. Native K/QK discovery is deployment-dependent and must remain optional for generic external-agent integrations.

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


# V4 ADDENDUM — Discovery Policy, Indexing, and Model Strategy

## A. PRA SDK discovery policy
Paper 6.5 should test discovery-policy selection, not only retriever selection. The PRA SDK should expose a resource discovery hint:

- `auto`: runtime chooses from metadata, query evidence, confidence and cost;
- `explicit`: typed URI/handle only;
- `token`: token-native exact/approximate matching;
- `index`: pre-built token/lexical index;
- `semantic`: contextual gist discovery;
- `hybrid`: bounded token/index + semantic union/reranking;
- `adaptive`: start with a cheap/high-confidence policy and escalate when uncertain.

Allow the hint at resource-collection, namespace, reference, and request level. Add `strict=True` to disable fallback. Otherwise a hint is a preference, not an unconditional command.

Resource metadata may expose routing hints such as `stable_name`, `aliases`, `indexable`, `semantic_only`, `expected_reuse`, `side_effect_class`, and version. These are system metadata, not gold relevance labels.

### Policy hypothesis
No discovery mechanism should be assumed universally best:
- explicit/name-heavy requests favor token lookup;
- large stable catalogs favor pre-built indexes;
- paraphrase/implicit intent requires semantic routing;
- hybrid can improve recall but costs more;
- adaptive retry should be concentrated on uncertain cases.

Evaluate a policy pipeline:

`query + resource metadata -> choose policy -> retrieve -> calibrate confidence -> optionally escalate -> materialize`

Representative escalation:

`explicit -> token -> index -> semantic -> hybrid -> ask/abstain`

but treat the ordering as an experimental variable.

Primary question: can SDK/user hints plus confidence-aware fallback approach oracle per-query policy choice without paying hybrid cost on every request?

## B. Index construction and reuse
Large tool/skill registries are unusually suitable for persistent indexes because definitions usually change less frequently than requests.

Evaluate:
1. no persistent index;
2. exact token/name/alias postings;
3. token-IDF and token n-gram index;
4. BM25/FTS word-level baseline;
5. semantic gist index;
6. combined token + gist registry.

Measure build/update cost separately from query cost. Report queries-per-rebuild and amortized cost. Version and invalidate indexes together with source text and cached K/V.

An index is a discovery accelerator, not a replacement for PRA materialization. Index hits resolve to PRA resources whose detailed K/V can remain nonresident until selected.

## C. Model strategy: mechanism first, pretrained validation second
Do not begin with a full agent harness and a strong pretrained model. Separate discovery competence from language/tool competence.

### M0 — deterministic resolver
No LLM is required for pure lookup. Scale exact/approximate token, index, hybrid and oracle routing to very large catalogs.

### M1 — toy custom transformer
Train a small decoder on synthetic tool-use language. Use opaque identities such as `tool_zaf_193`, with semantics available only in the supplied definitions. This removes pretraining priors and permits causal study of:

`discover -> materialize -> construct call -> observe result -> continue`

The toy model is evidence about mechanism, not real-world agent capability.

### M2 — Qwen3-0.6B primary pretrained bridge
Use Qwen3-0.6B for the first pretrained PRA experiments because the existing PRA integration already targets Qwen3 and the instruct model supports agent/tool calling. Run both opaque synthetic tools and realistic tool schemas.

### M3 — execution and typed observation
Insert a deterministic safe harness after model-side call construction. Treat
the call as untrusted, require request-scoped host authorization, and preserve
each result as a versioned `observation` resource with producer and schema
provenance.

### M4 — multi-tool planning and reactive discovery
Test three-to-five-step composition with just-in-time tool disclosure. Compare
reactive refresh, static required-tool disclosure, and no-refresh controls.

### M5 — speculative capability disclosure
Separate discovery from disclosure. Build a typed capability graph from
category/family/tag/keyword relations plus directional producer-consumer schema
edges. Compare P0--P9 fixed policies, including matched-budget reactive and
up-front speculative disclosure. Do not train an adaptive controller unless
static policies establish task-success headroom without unsafe exposure.

### M6 — PRA-native semantic resource discovery
Treat native mean K and Paper-2.8 QK routing as optional co-located,
shared-memory, model-server, or replicated-query integrations. Generic SDK use
must remain independent of native state. Prefer exporting low-rank projected
queries or identity-only server results, never raw Q/K by default.

### M7 — persistent session and typed history
Move persistent-history experiments after the tool discovery/disclosure and
execution boundaries are characterized.

Cross-model replication remains required after these dense Qwen gates; do not
run every catalog ablation on every model.

## D. Synthetic-to-real experiment ladder
Separate:
1. **lookup only** — output resource ID;
2. **lookup + schema/arguments** — materialize definition and emit a call;
3. **lookup + safe execution + observation** — deterministic mini harness;
4. **tool + skill composition** — select both resource types;
5. **persistent session** — resources + history + rotating `#__head`;
6. **real harness** — OpenAI-compatible PRA endpoint.

Opaque synthetic tools are mandatory controls because pretrained models may already know common tools such as `get_weather`, `calculator`, or `create_issue`.

## E. Policy-selection benchmark
Cross discovery policy with prompt/reference type and catalog structure.

Prompt strata:
- explicit URI;
- exact resource name;
- approximate/typo/alias;
- semantic paraphrase;
- implicit contextual reference;
- mixed tool + skill;
- ambiguous/nonexistent resource.

Catalog strata:
- small/dynamic: indexing may not amortize;
- large/stable: index should be favorable;
- semantically dense;
- lexically/name dense;
- frequently mutated/versioned.

Compare:
- fixed token;
- fixed index;
- fixed semantic;
- fixed hybrid;
- user/SDK hint;
- heuristic `auto`;
- confidence-triggered adaptive fallback;
- oracle per-query policy.

Measure quality, latency, retrieval stages invoked, fallback count, index/KV reuse, active materialization, and **oracle-policy regret**. The target is the quality/cost policy frontier.

## F. Skills need a separate hierarchical experiment
Tools are usually compact callable schemas. Skills can contain long instructions, examples, scripts and supporting documents.

Evaluate progressive disclosure:

`skill identity/summary -> instructions -> selected supporting files/examples`

Compare eager full skill injection, description-only disclosure, token/index selection, semantic gist selection, hybrid/adaptive selection, and oracle skill/supporting-file selection.

Measure both:
1. correct skill selection;
2. correct within-skill materialization.

Test whether the persistent index should cover only the skill registry or also internal sections/files.

## G. Additional measurements
Add:
- selected discovery mode;
- hint compliance;
- fallback/escalation count;
- retrieval stages/query;
- oracle-policy regret;
- policy accuracy by prompt/catalog stratum;
- index build/update time;
- queries per rebuild;
- index bytes;
- amortized index cost;
- selection quality vs total discovery/materialization cost.

Always log the actual executed discovery path. Never label a result simply `hybrid` or `token` if the runtime silently escalated.

## H. Additional falsification criteria
Narrow the adaptive-policy claim if:
- one simple fixed policy dominates across realistic strata;
- SDK hints do not improve quality/cost over a reasonable automatic default;
- adaptive fallback invokes expensive hybrid search so often that it loses its cost advantage;
- index build/update cost does not amortize under realistic catalog reuse;
- policy-selection overhead is comparable to the retrieval savings it is intended to create.

## I. Revised implementation order
0. Audit current OpenAI endpoint, `#__head`, Paper 2.6 discovery interface, and existing SDK search/index options.
1. Build deterministic synthetic tool/skill catalog generator with opaque identities.
2. Implement `auto|explicit|token|index|semantic|hybrid|adaptive` policy API and strict/fallback semantics.
3. Add token indexes, semantic gist registry, provenance/confidence records, and cache/resource registration.
4. Run M0/M1 lookup scaling, robustness, ambiguity, policy-oracle and index-amortization experiments.
5. Run Qwen3-0.6B opaque + realistic selection experiments.
6. Add schema/argument generation only after lookup behavior is characterized.
7. Add safe execution/observation mini harness.
8. Add hierarchical skill experiments.
9. Add persistent sessions, mutation/invalidation and index-update tests.
10. Add session-tree/subagent experiments.
11. Replicate selected conditions on a second small model and SmolLM3-3B.
12. Integrate one maintained open-source harness through the OpenAI-compatible endpoint.
13. Run matched end-to-end, risk-coverage and policy-regret analyses.

## J. Current implementation status

M0 and M1 are complete. M1 uses a 506,400-parameter decoder with PRA native-K/V
at layers 1 and 3, five seeds, 3,000 steps per seed, and catalog sizes 8, 32,
and 128. The host owns stable-URI discovery and temporary slot binding; the
model owns schema-conditioned argument construction. Oracle and idealized
discovered memory are at parity, while shuffled and disabled memory fail, so
the selected definition is causally used. This is a synthetic mechanism gate,
not evidence for pretrained agents, production discovery, or side-effecting
execution.

M2--M4 are complete on frozen Qwen3-0.6B over five deterministic prompt
presentations. Selected/oracle schema disclosure obtains 20/20 exact calls;
shuffled, irrelevant, and empty controls obtain 0/20, while eager disclosure
obtains 17/20. The safe in-memory executor accepts 20/20 selected calls and
creates typed observations; 17/20 continuations include every returned value.
Reactive JIT completes 10/15 multi-tool workflows, versus 3/15 static
required-tool disclosure and 0/15 without refresh.

M5 is implemented and its stop gate is negative. The combined graph reaches
.933 required-tool recall without destructive exposure but obtains 0/15 static
model task success; reactive JIT remains 10/15. Retain fixed graph policies but
do not train an adaptive disclosure controller from this evidence.

M6 is implemented as a co-located mechanism with projected-query and
identity-only server boundary types. Indexed lexical discovery obtains 1.000
single-step Top-1 and token routing obtains .889 multi-step required-tool
recall. Native mean/full QK and zero-shot Paper-2.8 projections are worse, so
native discovery is not the SDK default. Cross-model replication and M7
persistent-history experiments remain open.

## K. M5/M6 measurement and falsification addendum

M5 must report required/useful recall, useful precision, related and irrelevant
fractions, unsafe exposure, plan coverage, initial and total disclosed tools,
mid-execution retrievals, graph edge provenance, definition tokens, and task
success. Reject speculative disclosure when it only increases context or when
reactive JIT is equal or better at lower cost.

M6 must report model/tokenizer/layer/projection provenance, deployment mode,
index bytes, routing latency, Top-1/MRR, required and successor recall, and
whether raw native state crosses the process boundary. Reject native discovery
when external token/index methods match or exceed quality with a simpler
integration boundary. Both rejection conditions are met by the current frozen
cohort and must remain visible in the paper.
