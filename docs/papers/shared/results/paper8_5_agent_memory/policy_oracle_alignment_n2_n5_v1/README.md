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

The next behavioral experiment is therefore a frozen same-prefix comparison of
E0, E0+F1, and a compact E0+F1 closure receipt.  Broad R/K sweeps are suspended
until this causal test is complete.

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

This remains next-action evidence.  The next single expensive test is one
autonomous E0+F1 fork from the identical prefix and reset workspace.  Only if it
resolves without call inflation should the policy advance to repeated N=6
sequences.
