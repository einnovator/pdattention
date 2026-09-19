# Exploratory M2/P1 recovery on failed FULL controls

This separately declared diagnostic applies the frozen M2/P1 policy to the five
identities whose contemporaneous exact-prefix FULL controls failed in the
eight-identity held-out campaign.  It is explanatory evidence only: FULL did
not establish an eligible success, so recovery is not counted as treatment
preservation or as an accuracy win.

| Identity | Prefix role | FULL calls/outcome | M2/P1 calls/outcome | Candidate-own saving | Call delta |
|---|---|---:|---:|---:|---:|
| `scikit-learn__scikit-learn-13135` | empty-prefix repeatability control | 14 / fail | 23 / pass | 0.00% | +9 |
| `sphinx-doc__sphinx-8721` | cross-task retirement | 12 / fail | 14 / pass | 36.40% | +2 |
| `sympy__sympy-23534` | cross-task retirement | 14 / fail | 40 / fail | 35.39% | +26 |
| `psf__requests-2317` | cross-task retirement | 13 / fail | 12 / fail | 49.25% | -1 |
| `django__django-12741` | cross-task retirement | 8 / fail | 10 / fail | 58.74% | +2 |

Only one of four cross-task probes recovers.  The empty-prefix control is more
important for diagnosis: it materializes 100% of every request, starts with no
earlier issue, matches FULL's first action hash, yet changes from failure to
success.  Therefore cross-task interference cannot explain the 3/8 FULL rate by
itself; backend/trajectory repeatability is a material component even at
temperature zero.  The original historical success stratum also came from a
different 32K RTX/Ollama execution regime, while this campaign uses a 131K Mac
endpoint.

This exploratory pass did not repeat the source campaign's explicit
keep-alive-zero unload and warmup before every task.  Its results are preserved
rather than discarded, but they are not the matched runtime-state diagnostic.
The separately predeclared `paper85-v3-m2p1-failed-full-recovery-normalized-v1`
campaign adds the unload, warmup, three generation probes and 131,072-token
active-context gate before each cell.

`diagnostic_state.json` binds the source campaign state and each exact prefix,
then reports outcomes, call deltas, first-action hashes and candidate-own
saving.  Per-identity directories contain the official result, metrics,
manifest, submitted patch, selection trace and persistent trajectory export.
