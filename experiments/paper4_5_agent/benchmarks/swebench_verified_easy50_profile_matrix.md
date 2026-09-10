# Easy-50 baseline-success profile matrix

This diagnostic and qualification matrix is task-major. It uses only the 14
instances resolved by the admitted seed-0 no-PRA Easy-50 baseline, in the
frozen order of `swebench_verified_easy50_baseline_success14.json`.

For each instance, complete every admitted condition before starting the next
instance:

1. same-engine plain control;
2. direct native PRA, 100% budget (`near_passthrough`);
3. direct native PRA, 75% budget (`high_quality`);
4. direct native PRA, 50% budget (`balanced`);
5. direct native PRA, 25% budget (`economy`), only if 50% passes the task gate.

The execution order is therefore task-major, not condition-major:
`task-01/plain -> task-01/100 -> task-01/75 -> task-01/50 -> task-01/25`,
then the same sequence for task 02. A condition failure is a measured outcome;
an infrastructure failure leaves that task prefix incomplete and must be
repaired or rerun before advancing.

G11 gateway conditions are added only after a direct native profile passes the
single-task parity gate. Every cell uses mini-swe-agent 2.4.6, Qwen3-Coder 30B
Q4_K_M, the official SWE-bench 4.1.0 Docker grader, one task per chunk, and a
50-step limit. Each ARM execution explicitly pre-pulls the exact official
`linux/amd64` task image because official grading removes the tag after a cell.

Endpoint qualification traffic is executed once and preserved as a preflight
receipt. The llama-server process is then restarted before every measured arm,
and the runner replays that receipt without issuing another generation probe.
The earlier `plain_seed0_clean_v4` diagnostic is not a matched control: its
server log contains a 12-token generation preflight in the request slot before
the task, changing the first task prompt from 2,067 to 2,066 evaluated tokens.
It must not be compared with receipt-based PRA arms.

Native agent-history consumption additionally requires a causal chat-template
split: the resource slot contains the rendered system plus selected historical
turns, and the request slot receives only the exact remaining suffix. The
request slot is erased before every attachment. Engine configurations must pass
two distinct token-for-token gates before admission: dense-versus-native parity
for immutable context, and live-sequential-versus-attached parity after the
engine has generated history. The latter must compare exact resident token IDs,
not `/slots.n_prompt_tokens`, and must forward any final sampled token that was
returned to the client but not yet evaluated into KV. Re-prefilling a transcript
is not an acceptable substitute for this live-state gate. On the M4 llama.cpp
target, dense rematerialization at `ubatch=256` was not exact, while a direct
five-seed live-state attachment probe with the boundary-token bridge was 5/5
exact at short context and 1/1 exact near 2K tokens. These engine probes do not
admit the agent cell until the current plain and 100%-PRA task pair also matches.

At a 100% selection budget, PRA must degenerate to ordinary same-slot prefix
caching: it submits the complete logical prompt to the live request slot and
allows llama.cpp to reuse the exact common prefix. It does not copy or rebuild
detached resource state. This arm qualifies transport and presentation; only
the 75%, 50%, and 25% arms exercise selected-history attachment.

The initial user task is structural context, not optional trajectory history.
Every segment of that message (the PR description and mini-swe-agent contract)
is pinned in causal position at every budget; selection operates only over the
subsequent dynamic assistant/observation history. The first attempted 75% arm
(`direct_native_75_seed0_v17`) is invalid because it omitted `m1-0-user`, which
contained the PR description, and produced a runaway second response. Frozen
selection replay fails closed if any pinned task segment is absent or changed.

Live-state implementations must also support transcript backtracking. When
mini-swe-agent rejects a malformed assistant action, it appends a format-error
user turn without retaining the rejected assistant text in the logical
trajectory. The request can therefore replace, rather than extend, the tail of
the resident sequence. The llama.cpp path computes the exact token common
prefix and relies on `cache_prompt` to trim the rejected KV suffix before
continuing; an empty append is not an infrastructure error.

Selection is atomic at causal-turn granularity. An assistant action and all
following user/tool observations before the next assistant action form one
bundle. Retrieval ranks bundles and may only admit every segment of a bundle or
none of it; it cannot retain two assistant actions while dropping the
observation between them. Parser-generated user-only format-error runs remain
valid history because mini-swe-agent intentionally omits the rejected assistant
text. Before inference or resource materialization, the direct engine performs
a structural role check and renders the complete reconstructed request through
the model's own `/apply-template` endpoint. Both checks fail closed. Frozen
selection replay retains the exact ordered bundle segments and digest.

The active tail is stronger than an ordinary selectable bundle: the latest
retained assistant action and every following user/tool observation are all
mandatory visible messages. Making only the current observation mandatory can
drop the command that produced it. On task 01 that defect changed the second
model response and immediately caused a format error, so all profile attempts
made before the active-tail correction are diagnostic-only even when their
rendered role sequence is accepted by the chat template.

Selection contract v4 adds a progress spine. The task-aware retrieval query
combines the pinned goal with the current action and observation; independently
of retrieval, the selector pins the two latest completed causal turns, the most
recent mutation turn, and the most recent verification/submission turn. Pinned
and retrieved records are then restored to exact causal order. This prevents a
small early saving from deleting the only evidence of what the agent has
already tried or changed.

The previous `direct_native_75_seed0_pinned_v18` arm is also invalid: selecting
independent segments produced adjacent assistant turns, and Qwen's template
rejected request 24. The corrected implementation was replayed against all 31
user-ending prefixes of the admitted task-01 trajectory at 75%, 50%, and 25%;
all 93 reconstructed requests passed the physical Qwen3-Coder llama.cpp chat
template. The v4 progress-spine replay repeats that 31-by-3 validation with the
active tail and progress pins enabled. This remains a template-grammar gate;
task efficacy is measured separately.

Changed selected history uses an in-place resource delta. The engine submits
the replacement rendered prefix to the same pinned resource slot with
`cache_prompt=true`; llama.cpp retains the exact common token prefix and trims
the stale suffix. It must not delete the resource slot before each changed
digest. A physical cold-versus-delta gate cached 2,450/2,455 prefix tokens,
evaluated five replacement tokens instead of 2,455, and produced the same 32
sampled tokens with `physical_kv_copy=false`. Receipts separately report the
resource update mode, total selected-prefix tokens, cached common-prefix
tokens, and newly evaluated replacement tokens.

The prefix-cache factorial toggles only conversational request-slot reuse.
Detached PRA resource residency and its in-place resource-delta update remain
enabled in both arms. A cache-off endpoint is prohibited from entering the
same-slot live-continuation path; that path always uses `cache_prompt=true` and
would otherwise contaminate the OFF control. The cache-on arm replays the
cache-off arm's exact selection fixture. Headroom is a separate external
control pinned to version 0.37.0 and the same underlying llama.cpp model.

After task `n` is complete in every condition, the ordered prefix `1..n` is a
valid paired checkpoint. Reports must publish the exact prefix size and task
IDs; they must not compare partially completed conditions or pool the separate
36-task baseline-failure stratum.

Mini-swe-agent execution exceptions, Docker startup failures, missing
trajectories, failed endpoint preflights, and engine stalls are infrastructure
failures. They are recorded but never normalized to an unsuccessful solution
or admitted into a paired checkpoint.

Current task-01 gate: v4 at 75% resolves 1/1 after 33 actions with 22.3%
estimated logical-context saving. It creates, inspects, and submits the correct
patch. V4 at 50% reaches the 50-step limit without a submission despite 45.9%
estimated saving. The needed CharField path is present in selected records, but
the model ignores it, launches redundant whole-tree searches, and corrupts a
method boundary while editing. Therefore 25% is withheld, and the next arm is
the frozen 75% G11 replay that isolates gateway mediation from selection.

The task-01 frozen G11 replay now passes. Direct native and G11 are exact on all
33 logical-request digests, ordered selected-resource bodies, response texts,
and engine tuples (`prefix_cache_hit`, cached/native/wire token counts, and
`physical_kv_copy`). Both produce the same 689-byte patch and resolve 1/1 under
the official grader. The first two G11 attempts are excluded diagnostics: one
exposed an advertised-but-unimplemented resource-delta reconstruction, and the
next exposed erroneous engine-session invalidation when inline history moved to
detached records. Both are covered by regression tests. An initial grader-only
ARM image pull error was also excluded; the unchanged prediction resolves after
the declared `linux/amd64` regrade, and the platform is now propagated through
both agent and grader environments. Host swap pressure makes the 1,189.7 s G11
wall time unsuitable for a latency claim. The next task-major checkpoint is
direct 75% plus frozen G11 on task 02; after the native/G11 series, run matched
G00-only and Headroom controls as separate strata.
