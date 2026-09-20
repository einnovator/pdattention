# Cross-agent Easy-14 transfer protocol

## Question

The transfer study asks whether Paper 8.5's logical memory rules preserve task
quality outside mini-swe-agent. It is not an agent leaderboard. Every candidate
is first measured under its own unmodified `FULL` history. A memory treatment is
then paired only with task identities that the same agent solved under `FULL`.

The ordered cohort remains the locked Easy-14 card. Results must expose both
the unconditional solve rate over all 14 tasks and conditional preservation on
that agent's plain-success subset. An agent that fails `FULL` supplies agent
capability evidence, not evidence against the memory policy.

## Admission order

1. **Pi** is the first transfer because its native `read`, `bash`, `edit`,
   `write`, `grep`, `find`, and `ls` calls expose standard call/result identity
   with little hidden orchestration.
2. **OpenHands** is the richer open-source transfer. Its runtime state,
   summarization, and workspace services must be inventoried before activation.
3. **PRA Agent** is the required typed-record transfer.
4. **Kilo** is an optional additional open-source replication. Codex or Claude
   can provide external validity only when model, prompt, context management,
   and event traces are sufficiently auditable; they are never pooled with the
   fixed-model primary estimate.

The executable declaration is
`configs/agent_transfer_easy14_v1.json`. It admits one smoke task, then three
plain controls, then frozen policies on agent-specific plain successes, and
only then all Easy-14 identities.

## What remains frozen

Instruction pinning, causal-group atomicity, recent/mutation/verification/
protocol floors, retirement parameters, and token accounting do not change
between agents. Only the compatibility boundary may change:

- native event to portable-record normalization;
- declared semantics for each tool schema;
- role-valid realization of selected native messages.

Standard OpenAI tool traffic is handled directly by `OpenAIRecordizer`.
mini-swe-agent remains the exceptional adapter because it transports Bash
actions and observations as assistant/user text. The autonomous proxy's
`openai_tools` mode preserves `tool_calls`, `tool_call_id`, tool names, and any
provider fields while selecting complete causal groups.

## Cross-agent measurements

Exact trajectories are meaningful only for repeated controls of the same
agent. Across agents, report:

- official SWE-bench resolution;
- cumulative logical input tokens and paired saving;
- native tool events and normalized causal actions;
- calls/actions to solution;
- first treatment-induced divergence against that agent's `FULL` control;
- reacquisition of retired resources;
- hidden agent state and any native context compaction.

Native auto-summarization or context truncation is disabled when possible. If
it cannot be disabled, the arm is labelled `agent-managed context` and is not
pooled with raw-FULL controls.

## Expected strategy adjustments

Multi-tool agents need schema-declared effects rather than Bash parsing. A
single assistant message may issue several calls; the action and every result
remain one atomic causal group. Parallel reads can be retired only when all
results are complete and later versioned evidence dominates them. Agent-local
plans, todo lists, browser state, indexes, or summaries are classified as
model-visible records, auditable hidden state, or excluded non-causal telemetry
before a policy run.

These are normalization changes, not opportunities to retune a failed policy.
Any semantic rule learned only after inspecting transfer failures is declared
as a new exploratory policy and evaluated on held-out task identities.
