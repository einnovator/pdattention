# Baseline-conditional held-out confirmation: first three identities

This bundle is the first execution of
`paper85-v3-baseline-conditional-confirmation8-ctx131k`.  The campaign keeps
one continuous ordinary-role session and applies a predeclared missingness
rule: when the exact-prefix FULL control does not resolve an identity, the
policy is withheld and that exact FULL episode is carried into the session.
The next identity is then qualified from the shared prefix.  A baseline failure
is never scored as a selector failure or success.

The first three held-out identities produced:

| Identity | Exact-prefix FULL | Recent Frontier M2/P1 | Calls F/P | Input tokens F/P | Paired saving |
|---|---:|---:|---:|---:|---:|
| `scikit-learn__scikit-learn-13135` | 0/1 | withheld | 14/--- | 97,604/--- | --- |
| `django__django-14089` | 1/1 | 1/1 | 21/21 | 265,272/265,272 | 0% |
| `sphinx-doc__sphinx-8721` | 0/1 | withheld | 12/--- | 215,640/--- | --- |

Task 2 is an exact abstention control, not yet a saving point.  M2 retains the
two most recent genuine user prompts, and at session length two there is no
older instruction epoch to retire.  The FULL and policy arms have identical
assistant-command, assistant-content, and selected-message digests at all 21
decisions.  Thus the result validates carry-forward/session identity and exact
behavior under abstention, while contributing one eligible policy identity and
zero saving.

Task 3 required a 600-second transport ceiling after an infrastructure-only
180-second timeout during the first 16K-token prefill.  The clean retry is the
reported FULL result.  Task 4 paused before launch when the `.6` endpoint
became unreachable during cold normalization; it is absent from the outcome
table and must resume in an immutable retry directory.

The primary estimator remains conditional preservation on identities whose
exact-prefix FULL controls resolve.  Unconditional session economics include
the FULL fallback episodes.  These first three identities do not yet estimate
an accuracy--saving frontier because only one treatment is eligible and that
treatment correctly abstains at 0% saving.

Files:

- `campaign_state.json`: complete retry, health, pairing and pause ledger;
- `task01_full_fallback/*`: first baseline-ineligible FULL episode;
- `task02_full_control/*`: successful exact-prefix FULL control;
- `task02_recent_frontier/*`: successful exact treatment with no retirement;
- `task03_full_fallback/*`: second baseline-ineligible FULL episode.
