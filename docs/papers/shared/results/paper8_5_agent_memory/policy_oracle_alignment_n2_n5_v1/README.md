# Boundary-free policy/oracle alignment audit

This is a structural diagnostic over the locked persistent-FULL prefix before
episodes 3--6 of the N=6 independent-cross-workspace sequence.  The selector
receives the ordinary boundary-free transcript.  Evaluator episode identities
are used only after selection to score where excluded tokens originated.

The upper-bound structural oracle removes prior assistant/tool interaction
detail, keeps the active issue whole, pins every user instruction, and retains
one terminal finalization causal group for every prior instruction.  It does
not claim that oracle-aligned removal is behaviorally invisible to an LLM.
Counts use the locked local Qwen3-Coder tokenizer.

| Completed issues in prefix | Policy | Whole-input saving | Active interaction removed | Oracle precision / recall | Orphaned prior prompts |
|---:|---|---:|---:|---:|---:|
| 2 | global R4/M2/V2/P1 | 53.7% | 65.6% | 25.6% / 100% | 1 |
| 2 | E0 | 16.2% | 0.0% | 84.9% / 100% | 1 |
| 2 | E0+F1 | 13.8% | 0.0% | 100% / 100% | 0 |
| 3 | global R4/M2/V2/P1 | 59.3% | 37.2% | 83.4% / 93.8% | 2 |
| 3 | E0 | 58.9% | 0.0% | 89.5% / 100% | 2 |
| 3 | E0+F1 | 52.7% | 0.0% | 100% / 100% | 0 |
| 4 | global R4/M2/V2/P1 | 60.5% | 40.2% | 86.9% / 100% | 2 |
| 4 | E0 | 59.2% | 0.0% | 88.9% / 100% | 3 |
| 4 | E0+F1 | 52.6% | 0.0% | 100% / 100% | 0 |
| 5 | global R4/M2/V2/P1 | 63.4% | 79.1% | 68.9% / 100% | 3 |
| 5 | E0 | 49.6% | 0.0% | 88.2% / 100% | 4 |
| 5 | E0+F1 | 43.7% | 0.0% | 100% / 100% | 0 |

## Immediate findings

1. Global R4/M2/V2/P1 is genuinely too aggressive inside the active issue.  It
   removes 37--79% of active-issue interaction tokens at N=3--5.
2. E0 does not remove active-issue interaction.  Its structural defect is that
   every old user prompt remains visible while every old completion turn is
   removed.  The N=5 request therefore starts with five consecutive task
   prompts, four of which appear unanswered.
3. E0+F1 fixes that defect by retaining one finalization action--observation
   bundle per old instruction epoch.  It exactly matches this structural oracle
   at every audited prefix and still saves 43.7% at N=5.
4. Raw path matching is unsafe across workspaces: generic names such as
   `file.txt`, `patch.txt`, `pyproject.toml`, and `setup.cfg` collide, and two
   Django images also expose the same `/testbed/...` path.  Once resource IDs
   are scoped by the runtime-traced environment fingerprint, cross-workspace
   overlap is zero for all four prior/active comparisons at N=5.

The behavioral follow-up therefore tests closure realization directly.  Broad
R/K sweeps remain suspended until this causal defect is resolved.

## Frozen next-action closure ablation

The first part of that experiment is complete on the frozen Django-16145
request used by the earlier E2 same-prefix diagnostic.  All arms consume the
same canonical 15,861-token history before selection and use the same
Qwen3-Coder model with temperature 0, top-p 1, and seed 0.  Requests were
interleaved across three repeats.

| Arm | Selected tokens | Saving | Valid one-command actions | Unique valid commands |
|---|---:|---:|---:|---:|
| FULL | 15,861 | 0.00% | 3/3 | 1 |
| E0 | 7,621 | 51.95% | 0/3 | 0 |
| E0+F1 | 8,627 | 45.61% | 3/3 | 1 |
| E2 | 11,918 | 24.86% | 3/3 | 1 |

The E0 response is directly diagnostic rather than merely different.  It lists
all five preserved issue statements as simultaneous unrelated problems and
emits four candidate command blocks, violating mini-swe-agent's exactly-one-
command protocol.  E0+F1 instead identifies only the active Django issue and
emits one repository-localization command in all three repeats.  Thus the
failure is caused by orphaned old task prompts, not by aggressive removal from
the active task.  One closure bundle restores protocol validity while retaining
the desired 30--50% saving region.

The single autonomous E0+F1 fork then falsified whole-bundle retention as a
deployable policy.  It saved 43.10% cumulatively but submitted after three
actions without modifying the repository.  The old finalization bundle had
made the prior prompt look closed, but also exposed the historical submission
sentinel as an action exemplar.

Two compact closure receipts were tested next on the identical frozen request:

| Closure realization | Materialized tokens | Saving | Valid one-command actions | Diagnostic outcome |
|---|---:|---:|---:|---|
| synthetic Bash `true` turn | not retained by v1 reducer | not reported | 3/3 | all 3 copy `true`; reject |
| text-only assistant/user closure | 7,709 | 51.40% | 0/3 | active issue understood, but no command; reject |
| atomic closed-epoch retirement | 1,507 | 90.50% | 3/3 | one identical active-issue search command |

The initial compact-receipt reducer reported logical selected tokens rather
than post-replacement materialized tokens; that value is deliberately not
reconstructed after observing the outcome.  The last treatment adds no
synthetic message and consumes no evaluator boundary.  It recognizes genuine user
instructions from record provenance, proves an older instruction epoch closed
only when that epoch contains complete terminal/finalization evidence, and then
retires the old prompt and its interaction detail atomically.  If closure is
absent, it fails closed and retains the old prompt.  Its command is not byte-
identical to FULL, but is semantically equivalent: both search the repository
for the `runserver` implementation.

The active-only atomic fork fails official grading after 22 calls despite
71.03% cumulative saving.  It preserves the complete current epoch but makes
an incorrect address-parsing edit.  This failure, together with two successful
13-call E2 forks, suggests that natural successful trajectories can act as
behavioral exemplars even when their task facts are unrelated.

We therefore run one bracket rather than a parameter sweep: keep the two most
recent complete epochs exactly as E2 does, but atomically retire every older
terminally closed epoch including its user prompt.  This `E2_ATOMIC` treatment
selects 8,707 of 15,861 frozen-prefix tokens (45.10% saving) and produces one
identical valid current-task search in 3/3 trials.  Its first autonomous fork
resolves officially in 16 calls and sends 185,387 of 299,851 cumulative
message-content tokens, a 38.17% saving against its own full-history
counterfactual.  An independent reset-workspace repeat is exact: it also
resolves in 16 calls, sends the same 185,387 of 299,851 tokens, and produces
the same patch.  Atomic E2 therefore resolves 2/2 at 38.17% saving, but takes
three more calls than the two E2 successes and four more than the historical
independent-session success, so it has not passed the no-call-increase gate.

The next test is a held-out task with the policy frozen.  The repeated result
promotes atomic E2 beyond a one-run diagnostic, but two executions of one task
are not an accuracy estimate.
