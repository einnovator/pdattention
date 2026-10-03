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
