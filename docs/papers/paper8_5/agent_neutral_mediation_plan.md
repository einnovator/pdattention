# Agent-neutral mediation plan

## Product boundary

PRA uses one mediation core in two placements:

1. **Embedded (preferred):** request mediation runs inside the PRA runtime,
   beside the exact tokenizer and the resident logical/K/V state.
2. **External fallback:** the same mediator runs in `pra gateway` only when the
   target engine cannot host PRA mediation. In this topology it assumes the
   context-mediation role otherwise provided by the Headroom gateway.

An embedded runtime and an external gateway must not both mediate the same
request. A content-addressed stamp rejects different second-hop policies and
skips an identical second application.

## Reusable layers

| Layer | Inputs | Agent-specific? | Engine-specific? |
| --- | --- | --- | --- |
| Agent compatibility adapter | Native messages and tool events | Yes, thin and optional | No |
| Portable record/effect schema | Causal groups, operations, resource versions, completeness | No | No |
| Policy planner | Typed history, exact token counter, frozen policy config | No | No |
| Mediation/realization | Frozen selected IDs, replacements, tool disclosure | No | No |
| Native execution | Logical IDs and selected K/V | No | Yes |

mini-swe-agent's adapter recognizes its nonstandard assistant-to-user Bash
protocol and emits the portable schema. Bash parsing is confined to
`experiments/paper8_5_agent_memory/miniswe_semantics.py`; common PRA code never
imports it. Standard OpenAI tool-call traffic needs no agent adapter. A future
agent should normally emit `pra_record` and generic tool-effect metadata
directly. A standard OpenAI function may also put its static category and
operation class in `function.x-pra-semantics`; execution middleware adds the
realized resources, versions, spans, and completeness to its result record.

## Avoiding wholesale retests

Selection experiments produce a `WireAgentMemoryPlan` containing the source
history digest, selected record IDs, optional observation replacements, policy
parameters, and a plan digest. The same plan is replayed unchanged through the
ordinary-text harness, embedded mediation, and external fallback. Therefore:

- changing placement requires only mediator conformance tests;
- changing an engine requires 100% parity and physical-reuse qualification;
- changing a policy, model, agent, task, tokenizer, or model-visible receipt
  requires new quality evaluation;
- changing only mediation placement while preserving the plan digest and
  serialized model input requires conformance testing, not a new task-quality
  campaign;
- shadow mode may collect candidate decisions without altering model input;
- active mode is enabled only after the corresponding frozen quality gate.

Old quality rows are not rerun merely because mediation moved. They are
invalidated only when their model-visible selected IDs or replacement bytes
change.

## Implementation and experiment sequence

1. Complete the generic state-authority planner over declared metadata; unknown
   and incomplete effects retain full text.
2. Run mini-swe-agent policy discovery in ordinary-text mode: FULL controls,
   isolated H1/H2/H3/H4 curves, useful combinations, head/middle/tail controls,
   and oracle add-back.
3. Freeze frontier plans at requested savings levels and verify byte-identical
   realization in embedded and external placement.
4. Transfer the frozen policy/configuration to a second open-source agent and
   the PRA agent. Only the compatibility adapter and evaluation driver may be
   agent-specific.
5. Qualify one real engine per family at FULL/PRA-100 before 90% policy tests.
   Paper 8.5 reports task quality and logical token tradeoffs; Paper 4.5 reports
   K/V hits, copies, re-encoding, TTFT, memory, and lifecycle.
6. Promote result compaction and tool disclosure from shadow independently.
   Each feature has a separate plan component and digest so a feature change
   does not silently contaminate an earlier policy arm.

Selection evidence (rule, resource, version, completeness, and witness IDs)
stays in request/plan metadata and is not counted as prompt text. When a chat
protocol requires an observation to remain paired with its action, active
compaction emits a minimal role-valid protocol stub. A richer semantic receipt
is a separate model-visible intervention and must not be substituted into a
previously qualified arm.

Automatic history planning is implemented over portable effect metadata.
Automatic multi-tool disclosure is deliberately still shadow-only: the frozen
plan can carry selected tool names, but graph-root selection and schema
dependency closure require a separate qualification corpus. This omission is
irrelevant to mini-swe-agent's one-Bash-tool study and will be exercised when
transferring to a multi-tool agent.

## Acceptance gates

- Common modules contain no mini-swe-agent, Bash, browser, database, or other
  tool syntax.
- FULL and PRA-100 preserve exact model-visible request bytes and action
  trajectory for deterministic engines.
- A below-100% engine run matches a fresh-prefill reference consuming the same
  selected records and positions; it need not match FULL.
- Every receipt is tokenizer-smaller than the observation it replaces.
- Missing semantics, stale versions, incomplete output, stale plans, or double
  mediation fail closed.
- Transfer claims require at least two agents and a real-engine replication;
  mini-swe-agent alone is policy-development evidence.
