# Distance-conditioned mini-swe-agent diagnostic

This campaign is preregistered before interpreting model output.  Broader
policy exploration is frozen until it explains when removing resource-
disconnected history increases calls-to-solution.

## Questions

1. Does behavioral disturbance decrease as a removed closed instruction
   component moves from one to five prompts behind the active prompt?
2. Does disturbance decrease when one complete no-path action--observation
   group moves from 10 to 30 completed tool turns behind the next decision?
3. Is the disturbance caused by orphaned prompts, loss of protocol/workflow
   examples, loss of current-task evidence, or unstable FULL generation?
4. Can a bounded natural behavioral floor preserve calls and resolution while
   still yielding 25--60% history omission as session length grows?

## Distance semantics

- Task distance 1 is the immediately preceding genuine-user instruction
  interval. The active interval is distance 0. No evaluator task ID is given
  to the selector.
- Tool-call distance 1 is the most recent complete action--observation group
  before the next model decision.
- A task-distance arm removes the prompt and its entire completed interaction
  atomically. This prevents the known orphaned-prompt confound.
- A tool-distance arm removes one complete causal group while preserving the
  surrounding prompt and terminal closure.
- Every target must have no path to the active prompt in the generic
  information-flow DAG. Missing evidence abstains. Historical traces without
  workspace-lineage receipts are labelled heuristic, never certified.

## Stage A: frozen next-action bracket

The initial N=6 request interleaves FULL with:

- `TASK_D1`, `TASK_D3`, and `TASK_D5`;
- `TOOL_D10`, `TOOL_D20`, and `TOOL_D30`;
- atomic DAG frontiers `M1`, `M2`, and `M3`.

Dry-run opportunity on the exact tokenizer is:

| Arm | Materialized-token saving |
|---|---:|
| Task D1 | 21.84% |
| Task D2 | 9.37% |
| Task D3 | 11.50% |
| Task D4 | 0%; fail-closed abstention |
| Task D5 | 12.30% |
| Tool D10 | 6.15% |
| Tool D20 | 0.84% |
| Tool D30 | 1.04% |
| DAG M1: active prompt only | 55.01% |
| DAG M2: active + one prior prompt | 33.17% |
| DAG M3: active + two prior prompts | 23.80% |

Stage A measures exact request and selected-message hashes, action validity,
command hash, operation class, and divergence from the contemporaneous FULL
repeat. It is not task-quality evidence.

The initial D10/D20/D30 tool bracket is explicitly a pilot, not the age-effect
estimate: those groups contain 1,647, 226, and 279 tokens and have different
operation classes.  The complete candidate ledger identifies two matched
follow-ups:

- read groups at distances 8, 16, and 28 contain 275, 282, and 283 tokens;
- edit groups at distances 3, 15, and 43 contain 142, 121, and 156 tokens.

These triples hold coarse operation semantics and removed token mass much more
nearly constant while varying chronology.  Any final distance claim must use
the matched triples, not the pilot D10/D20/D30 contrast.

The task-distance matched follow-up uses D2, D3, and D5, whose atomic closed
components contain 2,512, 3,083, and 3,295 tokens. D1 remains a useful
near-history stress point but is almost twice as large at 5,852 tokens.

## Variance rule

Temperature zero is not accepted as an explanation by itself. FULL repeats
must use byte-identical model-visible messages, observed model digest,
generation settings, and ordered observations. If FULL command hashes vary,
the cell is a backend-stability observation and no removal effect is inferred.
For a stable FULL request, a repeat-stable arm difference is attributable to
the changed history. Token log probabilities will be captured on the smallest
reproducing pair when the endpoint exposes them; they do not replace the
action- and task-level outcomes.

## Stage B: autonomous calls-to-solution

Only the smallest Stage-A arms that reproduce a stable difference advance.
Each selected arm is paired from the same immutable prefix and clean workspace.
Primary outcomes are:

- official resolution;
- total model/tool calls;
- calls to first source mutation;
- post-mutation verification and submission calls;
- repeated searches, reads, failed edits, tests and submissions;
- candidate-own and paired cumulative input work.

An arm is useful only when it loses no paired success, does not increase total
calls, and retains meaningful omission. The target band is 25--60%, depending
on task/session length.

## Mitigation ladder

Mitigations are tested one causal factor at a time:

1. atomic component retirement, eliminating unfinished-old-request signals;
2. one natural valid action--observation exemplar;
3. one natural mutation--verification--completion workflow spine;
4. one complete recent successful instruction component;
5. two complete recent successful components.

Synthetic post-cache summaries are excluded from strict native-K/V promotion.
The first mitigation that preserves success and calls becomes the bounded
behavioral floor; older closed disconnected components remain eligible for
retirement, so saving can still increase with session length.

Cross-agent transfer begins only after the mini-swe-agent root cause and one
nondominated mitigation are repeat-qualified.

## Current diagnostic results

The prior Task-5 FULL cohort exposes a necessary reproducibility control.  An
identical selected-message request, model digest, temperature and seed was
identical for five actions, then diverged on call 6.  One FULL execution solved
in 9 calls; two later FULL executions followed the same alternative trajectory,
used 18 calls, and failed.  Relative to the successful FULL trajectory, the
alternative added four reads, three unclassified diagnostic actions, one write,
one diff, one failed tool action, and a second submission attempt.  This is an
observed FULL-control split, not evidence that selection caused nine extra
calls.  The exact call-6 request has been reconstructed for repeated logprob
replay.

That replay now reproduces the split under locked Ollama 0.34.2 on the 48-GB
M4 Pro. Five byte-identical FULL requests yield two exact response clusters:
the first request chooses the command from the historical 9-call success
branch; the following four choose the command from the historical 18-call
failure branch. The first response-token divergence occurs after 99 common
tokens. It is not a near tie or an adequate basis for a generic
``floating-point noise'' explanation. In the first response, the decoder
emits ` this` at reported logprob -1.625 even though the reported top candidate
is ` the` at -0.509; the next four responses emit the reported top candidate.
The reported distributions are similar, but they are not the effective
sampler distribution and cannot establish sampler causality.

The cold/warm isolation is now replicated. After unloading the model, two
independent cold executions of the exact `a009e752...` request produce the
same complete response hash and successful-branch command. The immediate warm
execution produces the alternative response and command. Three warm
executions with temperature `1e-7` are also byte-identical to the alternative
branch. This is an exact 2/2 cold cluster versus a stable warm cluster. It
matches the failure shape reported upstream for Ollama's first request after
load, while the adjacent Ollama logprob report explains why a chosen token can
differ from the highest *reported* raw-logit probability. Therefore the
observed 9-versus-18-call FULL split is a backend decoding-state control, not
admissible selection evidence. Policy comparisons must declare and stabilize
cold/warm state as well as model, messages, tokenizer, seed and temperature.
Near-zero temperature is not promoted as the workaround. The apparent
autonomous Task-3 model failure at request 5 was subsequently isolated as a
client-transport timeout: Python `urllib` on the Medium Mac timed out, while
the exact captured 14,385-token request completed through `curl` in 166.55 s
and returned the expected response. Frozen next-action repeatability still
does not qualify a decoding configuration for autonomous use, but the timeout
must not be counted as a policy or model failure. Autonomous comparisons now
use the same `curl` transport and a 600 s upstream ceiling.

The first stable distance cell removes one 275-token read group eight complete
tool turns behind the current decision.  It saves 1.03% of the request and
matches the exact FULL command in 2/2 trials, with stable 2/2 interleaved FULL
controls.  The matched distance-16 and distance-28 read cells hold removed
token mass near 280 tokens and are run before any autonomous promotion.

One important structural case is now isolated.  The component four user
instructions behind the current task contains 9,089 tokens, is resource-
disconnected, but lacks a valid submission receipt.  The strict closed-only
policy therefore retains it, even though several later user instructions have
superseded it.  A diagnostic-only superseded-unfinished arm can retire that
whole component atomically, yielding 33.92% next-request omission.  This arm
directly tests whether an old apparently unfinished request, rather than loss
of useful evidence, is responsible for later workflow disturbance.

The Task-6 frozen decision confirms that isolation: retiring only the 9,089-
token superseded-unfinished component preserves the exact next command in 2/2
treatment trials (33.92% saving), and retiring it together with the oldest
closed component also preserves the exact command in 2/2 trials (46.21%
saving).  By contrast, removing the closed component three prompts back saves
11.50% and preserves the search operation but changes the observation contract
from a `head -10` bounded search to an unbounded search.  Exact command,
operation class, and observation contract are therefore reported separately.

The first admitted autonomous cell uses Task 4, whose historical persistent
FULL control officially solved in 14 calls.  The treatment atomically retires
the no-path, superseded-unfinished component two prompts back.  It officially
solves in 9 calls with 51.61% cumulative logical/materialized saving (76,694 of
158,495 content tokens retained), zero reacquisition events, and no upstream
errors.  Relative to FULL it reaches the first mutation at call 7 rather than
12, removes four reads and one miscellaneous diagnostic action, and leaves the
two-call mutation-to-submission tail unchanged.

The treatment repeat is exact at the episode-content level: 2/2 official
solves, 9 calls in each execution, identical 20-message role/content
trajectories, and the same 51.61% cumulative saving.  The paired historical
FULL success used 14 calls and first mutated at call 12; both treatments first
mutate at call 7.  This repeat-qualified cell still covers one task identity
and one model/backend, so it motivates transfer rather than a production
default.

A second task identity now has a decode-state-qualified frozen decision. For
Task 3, one discarded FULL warmup is followed by three interleaved FULL and
three treatment requests at temperature `1e-7`. All three FULL responses are
byte-identical. Atomically retiring the immediately preceding 9,089-token
no-path, superseded-unfinished instruction component saves 64.73% of the
request and matches the exact FULL command, operation, and observation
contract in 3/3 trials.

The corrected autonomous FULL--treatment pair is now complete. FULL officially
resolves Task 3 in two 13-call repeats and sends 199,232 cumulative input
tokens per repeat. The FULL controls match exactly through call 12 and differ
only in the terminal submission command spelling. Atomic
distance-1 retirement resolves in two exact 15-call repeats. Each materializes
95,934 of 232,269 tokens: 58.70% candidate-own saving and 51.85% paired saving
relative to FULL, after paying for the two extra calls. The extra calls do not represent
repository rediscovery. Both arms issue the same first search and inspection
and both initially target the wrong `sed` line. FULL never changes the source
file; after observing the no-op it hand-builds a valid patch and submits it.
The selective arm continues inspecting, edits the actual source line, verifies
it, and submits the workspace diff. Its delta is two reads plus one write minus
one separate diff action. Therefore the raw 13-versus-15 comparison is a real
trajectory difference but not evidence that retirement destroyed current-task
state or made the agent less capable. The reducer now reports effective source
mutations, patch-construction provenance, candidate-own saving, and paired
saving. Both arms are therefore repeat-qualified within one
task/model/backend identity, not a
cross-task accuracy estimate.

## Current root-cause disposition

| Hypothesis for extra calls | Current evidence | Disposition |
|---|---|---|
| Old request appears unfinished | A 9,089-token disconnected interval has no valid completion receipt; closed-only selection retained it. Atomic retirement preserves the frozen next action and gives the 2/2 nine-call Task-4 treatment. | Supported as a policy defect in the old closure rule; use later-instruction supersession plus no-path evidence. |
| Missing one-command format | Removing matched old read groups at tool distances 8, 16 and 28 preserves the exact next command in 6/6 trials. | Not caused by age alone. Retain the system contract and at least one natural valid action--observation exemplar before broader removal. |
| Missing search--inspect--edit--verify--submit workflow | Removing a closed component three prompts back preserves the search operation but changes bounded `find ... | head` to unbounded `find`. | Plausible conditioning effect; operation-class equality is insufficient. Keep one recent complete successful workflow component. |
| Insufficient evidence confidence | The repeat-qualified Task-4 treatment removes four reads and reaches the same mutation five calls earlier, with no reacquisition. On Task 3 both arms make the same no-op edit; the selective arm repairs the real file while FULL fabricates the submitted patch. | No current-task evidence was removed in either cell. Count effective source mutation and submission provenance before calling added calls a confidence loss. |
| Missing error-recovery example | No admitted cell yet removes the sole relevant failed-action/recovery pair. | Pending targeted frozen ablation; failed action, error observation and recovery remain atomic. |
| Missing stopping example | The Task-4 treatment leaves the two-call mutation-to-submission tail unchanged. | No stop regression in the admitted cell; preserve one valid verification/submission spine until replicated. |
| Temperature-zero model variance | Exact FULL request branches 9 versus 18 calls historically; 2/2 cold requests and warm requests form different exact response clusters. | Confirmed backend-state confound. Do not attribute call deltas to selection without stable paired FULL controls. Separately, use a transport capable of completing long cold/prefix-miss requests. |

The present policy candidate is deliberately small: preserve the active
instruction and its complete progress spine, preserve the system/tool
contract, keep one recent complete successful workflow component, keep
failed-action/recovery and mutation/verification/submission groups atomic, and
retire only older no-path instruction components atomically. No synthetic
summary is inserted into the model-visible stream. This is expected to retain
25--60% depending on session length while avoiding the partial-component
deletions that created the strongest earlier confounds.
