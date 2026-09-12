# Human action audit

This audit classifies generated Bash actions against contemporaneous FULL.  It
does not infer task success from a single frozen decision and does not count
reasoning-text differences as action differences.

| Arm | Unchanged | Comment-only | Likely operationally equivalent | Different / plausibly useful | Format-invalid | Submission-critical |
|---|---:|---:|---:|---:|---:|---:|
| DAG-EXCLUDE@100 | 22 | 0 | 0 | 0 | 0* | 1 |
| Matched causal tail | 20 | 0 | 0 | 0 | 0 | 3 |
| Lexical@90 | 10 | 3 | 6 | 2 | 0 | 2 |
| Spine + lexical@90 | 10 | 3 | 3 | 3 | 1 | 3 |

`*` DAG and FULL share the same missing-command response at decision 22, so it
is not a policy-induced validity divergence.

Key observations:

- DAG first differs at decision 21 after omitting `m30`--`m31`, an earlier
  post-mutation source verification duplicated by a later read.  It displays
  the diff instead of saving `fix_patch.txt`, but decision 23 returns to the
  exact submission action.
- The causal-tail ceiling omits `m4`--`m5`, the early full-file read and its
  truncated large observation.  At decisions 21--23 it respectively displays
  rather than saves the diff, tries to submit a nonexistent `patch.txt`, and
  creates the patch without the submission marker.  Because it underfills far
  below the DAG arm, this association is not evidence of causal importance.
- Lexical first differs at decision 6 after omitting `m6`--`m7`, the attempted
  reproduction and `ModuleNotFoundError: numpy`; it substitutes another
  environment diagnostic.  Later material divergences coincide with omitted
  mutation, verification, and submission-state bundles.
- Spine + lexical first differs at decision 8 after omitting `m6`--`m9`.  Its
  decision 9 is an unterminated code block, decisions 16--18 are plausible
  alternative verification actions, and decisions 19/21/22 affect completion
  or submission state.

The next oracle add-back candidates are therefore failed-environment status,
successful mutation, post-mutation verification, and patch/submission-artifact
state.
