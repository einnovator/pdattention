# Baseline-conditional held-out confirmation: eight-identity completion

This bundle is the completed execution of
`paper85-v3-baseline-conditional-confirmation8-ctx131k`. The frozen campaign
keeps one continuous ordinary-role session and applies a predeclared
missingness rule: when the exact-prefix FULL control does not resolve an
identity, the policy is withheld and that exact FULL episode is carried into
the session. A baseline failure is never scored as a selector failure or
success.

| Identity | Exact-prefix FULL | Recent Frontier M2/P1 | Calls F/P | Input tokens F/P | Paired saving |
|---|---:|---:|---:|---:|---:|
| `scikit-learn__scikit-learn-13135` | 0/1 | withheld | 14/-- | 97,604/-- | -- |
| `django__django-14089` | 1/1 | 1/1 | 21/21 | 265,272/265,272 | 0.00% |
| `sphinx-doc__sphinx-8721` | 0/1 | withheld | 12/-- | 215,640/-- | -- |
| `sympy__sympy-23534` | 0/1 | withheld | 14/-- | 331,711/-- | -- |
| `psf__requests-2317` | 0/1 | withheld | 13/-- | 368,125/-- | -- |
| `sympy__sympy-16886` | 1/1 | 1/1 | 10/10 | 328,501/142,732 | 56.55% |
| `django__django-12741` | 0/1 | withheld | 8/-- | 288,082/-- | -- |
| `django__django-15277` | 1/1 | 1/1 | 6/6 | 236,111/89,534 | 62.08% |

The primary baseline-conditional estimator uses only identities whose
contemporaneous exact-prefix FULL controls resolve. On those three eligible
identities, M2/P1 preserves 3/3 official resolutions, uses 37 versus 37 calls,
and reduces cumulative message-content input from 829,884 to 497,538 tokens:
**40.05% paired saving**. Task 2 is an exact abstention control. Task 6 diverges
at action 1 and Task 8 at action 3, yet both treatment trajectories solve in
the same number of calls as FULL. This distinguishes outcome preservation from
exact-trajectory agreement.

The result is not an unconditional 100% accuracy claim. Exact-prefix FULL
itself resolves only 3/8 identities in the continuous session. Including the
five carried FULL fallback episodes, the whole-session logical saving is
15.59% (2,131,026 full-history tokens versus 1,798,700 materialized tokens).
Thus the experiment supports a conditional memory-policy claim and exposes a
separate persistent-session baseline-interference problem.

Transport-only Task 4 attempts are quarantined in `campaign_state.json`. The
clean completion used an existing-key SSH tunnel from the `.8` runner to the
same dedicated `.6:11435` Ollama service after the direct application-port path
proved intermittent. The tunnel changed transport only; model digest, 131,072
context, temperature zero, tokenizer, task order, workspaces, and policy were
unchanged.

Key files:

- `campaign_state.json`: immutable retry, qualification, pairing, and outcome ledger;
- `baseline_conditional_summary.json`: reproducible paired and unconditional reduction;
- `taskNN_full_*`: compact exact-prefix FULL artifacts;
- `taskNN_recent_frontier`: compact treatment artifacts for eligible identities.
