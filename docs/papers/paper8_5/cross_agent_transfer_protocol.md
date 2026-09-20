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

## Admission order and current status

1. **Pi** is the first transfer because its native `read`, `bash`, `edit`,
   `write`, `grep`, `find`, and `ls` calls expose standard call/result identity
   with little hidden orchestration. Its controlled Task-2 `FULL` admission is
   complete and officially resolves the issue. The second staged FULL control
   also resolves. Task 4 reaches the completion-token ceiling before writing
   and exports an empty patch, so Pi completes plain-three at 2/3 and policy
   pairing is restricted to Tasks 1 and 2.
2. **Kilo** is the second open-source transfer. Its controlled Tasks 1 and 2
   resolve officially. Task 4 cleanly reaches the correct diagnosis but hits
   the completion-token ceiling before issuing its edit, so Kilo completes
   plain-three at 2/3 and policy pairing is restricted to Tasks 1 and 2. Kilo
   reports every native tool event as transport-completed even when command
   output contains a semantic failure; portable records must therefore carry
   exit or failure evidence independently of transport state. External
   telemetry, plugins, project configuration, and persistent session storage
   are disabled for hermetic controls.
3. **PRA Agent** is the required typed-record transfer. The SDK is record-native,
   and its new SWE-bench container adapter now resolves controlled Tasks 1 and
   2. Task 1 exposed and fixed a second transport boundary: an unambiguous
   terminal serialized tool decision was previously mistaken for a final
   answer. After two further fail-closed serialization repairs, Task 4 still
   terminates without a patch. PRA Agent therefore completes plain-three at
   2/3, and policy pairing is restricted to Tasks 1 and 2.
4. **OpenHands** is the supplementary richer-tool transfer. The admission uses
   the current SDK directly inside the task-derived container, not the
   unmaintained standalone CLI. Its native terminal, file-editor, and
   task-tracker tools remain enabled, while the SDK condenser is explicitly
   `None` so hidden summarization cannot masquerade as a PRA result. Its clean
   Task-1 FULL control officially resolves after 33 model requests; the
   remaining plain-three controls are pending.

Codex or Claude can provide external validity only when model, prompt, context
management, and event traces are sufficiently auditable. They are never pooled
with the fixed-model primary estimate and cannot replace the open, same-model
Pi/Kilo/PRA-Agent transfer.

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

Two additional invariants follow from the completed admissions. First, an
action that fails, its error observation, and the recovery action form one
atomic progress unit: retiring only the failure can make a recovered path look
unfinished, while retiring only the recovery can resurrect a known-bad action.
Second, token totals are paired within agent. Different system prompts and tool
schemas make raw cross-agent token totals incomparable even when the model,
task and tokenizer are identical.

A third invariant is that provider stop reason is not semantic completion.
Pi terminates a reasoning-only turn whose provider stop reason is `length`;
portable evaluation records this as an incomplete agent outcome, not a valid
final answer. A separately declared 2,048-token diagnostic continues past that
point but still submits no patch after three stream-termination retries, so a
larger ceiling alone is not a repair. Completion-horizon diagnostics remain
separate from the frozen primary control.

Kilo independently reproduces this invariant on Task 4: the model states the
correct edit direction, reaches `length`, and the agent exits with no tracked
change. Its earlier external TLS stall is quarantined rather than attributed to
the task; disabling nonessential telemetry and plugin state makes the retry
hermetic. Agent controls therefore require both semantic completion handling
and removal or explicit accounting of external control-plane dependencies.

OpenHands adds two richer-agent requirements. First, optional provider usage
fields belong to accounting, not semantic execution: OpenHands 1.49.2 aborted
on an absent `cache_creation_tokens` field until the telemetry boundary
zero-defaulted it. Second, task trackers and other agent-maintained plans are
versioned causal resources. The generic tool declaration assigns repeated
tracker updates the stable `agent://task-tracker` identity, while file-editor
commands declare operation and path. Arbitrary terminal commands remain
unknown-effect barriers unless execution middleware supplies complete effect
receipts. These are portable metadata rules, not agent-specific selection
code.

A fourth invariant is that a provider-native `tool_calls` object and the
runtime's exact terminal serialized action projection are two encodings of the
same decision, but arbitrary prose is not. PRA Agent therefore accepts the
serialized form only when the PRA-owned sentence terminates the response and
its name and arguments pass the ordinary disclosure, schema, and authorization
checks. On Task 1 this changes an empty-patch stop after 10 requests into an
official solve after 14 requests without changing the model or task. Such a
repair belongs to agent qualification; the failed pre-fix trajectory cannot be
counted as evidence against a memory policy.

Task 4 adds a fifth invariant. A terminal provider-specific function block is
an attempted action, while malformed action arguments are a rejected action;
neither is semantic task completion. The runtime accepts a well-formed
terminal Qwen function block, but never repairs malformed JSON for execution.
Instead it stores a visible rejection and asks for a fresh action. The final
corrected FULL control still stops after correctly localizing the required
string conversion but before issuing a write, so its failure remains in the
agent baseline and is excluded from the policy denominator.
