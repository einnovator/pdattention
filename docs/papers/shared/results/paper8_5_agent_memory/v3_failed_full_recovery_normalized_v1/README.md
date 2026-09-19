# Cold-normalized M2/P1 diagnostic on failed persistent-FULL controls

This is a separately declared explanatory diagnostic, not a policy-preservation
cohort.  The frozen boundary-free Recent Frontier M2/P1 policy is applied to
the five identities whose exact-prefix persistent-FULL control failed in the
held-out eight-issue sequence.  Before every cell, the runner unloads the
Ollama model with `keep_alive=0`, performs an OpenAI-compatible warmup, requires
three successful generation probes, and verifies an active context capacity of
at least 131,072 tokens.  The exact source prefix is reused.

| Episode | Identity | Diagnostic role | Failed FULL calls | M2/P1 calls / outcome | Candidate-own saving | Call delta | First action same |
|---:|---|---|---:|---|---:|---:|---|
| 1 | `scikit-learn__scikit-learn-13135` | empty-prefix repeatability control | 14 | 14 / fail | 0.00% | 0 | yes |
| 3 | `sphinx-doc__sphinx-8721` | cross-task retirement | 12 | 14 / pass | 36.57% | +2 | yes |
| 4 | `sympy__sympy-23534` | cross-task retirement | 14 | 40 / fail | 38.01% | +26 | yes |
| 5 | `psf__requests-2317` | cross-task retirement | 13 | 10 / fail | 49.67% | -3 | no |
| 7 | `django__django-12741` | cross-task retirement | 8 | 14 / fail | 58.40% | +6 | yes |

The candidate trajectories materialize 1,300,918 of 2,253,858 message-content
tokens, a 42.28% ratio-of-sums saving on their own realized trajectories.  They
use 92 calls versus 61 in the failed FULL controls, primarily because the SymPy
candidate reaches the 40-call cap.  M2/P1 recovers one of four cross-task
failures and does not change the empty-prefix control.

The Sphinx recovery is semantically concrete.  The failed FULL trajectory puts
the epub guard outside `collect_pages`, where it references unavailable state;
M2/P1 puts the guard inside `collect_pages` and passes the official grader.
Both arms have the same first action and diverge at the third action while
reading the same source file.  Their first assistant messages also explicitly
refer to prior, unrelated tasks, directly demonstrating cross-task cognitive
interference under FULL retention.

The empty-prefix control reproduces all 14 source action hashes and the same
official failure.  This invalidates the earlier non-normalized apparent
recovery and proves that the historical “plain-success” label is not a
contemporaneous guarantee: that label came from a 32K RTX/Ollama execution,
whereas this campaign uses a 131K Mac runtime.  Temperature zero does not make
different runtime regimes or historical executions interchangeable.

Therefore this diagnostic supports a narrow conclusion: retiring old task
detail can remove harmful cross-task interference, but it is not a general
repair for unstable or capability-limited FULL trajectories.  It is excluded
from the primary preservation denominator.  The next accuracy estimate uses a
second predeclared task order with cold exact-prefix FULL controls before each
M2/P1 candidate.

Primary machine-readable evidence is in `diagnostic_state.json`; each episode
directory contains its request-level selection ledger, exported typed
trajectory, patch, official result, metrics, and run manifest.
