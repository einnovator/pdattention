# Task 6 exact-prefix FULL qualification

This bundle records two FULL-history qualification attempts for
`scikit-learn__scikit-learn-13135` at episode 6 of the frozen persistent
session. Both use the same five-episode prefix (SHA-256
`c95cbae606a68e30f776849df0dc09c84caa09430479d45fda30b4cdf1442283`),
clean task workspace, model revision, tokenizer, scaffold, and decoding
parameters.

| FULL repeat | Official solve | Calls | Materialized tokens | M2/P1 executed |
|---|---:|---:|---:|---:|
| 1 | 0 | 5 | 142,805 | no |
| 2 | 0 | 5 | 142,821 | no |

The model-visible inputs and assistant actions match through the first two
requests. Response three differs, after which the final two assistant actions
reconverge; both executions remain unresolved. There are no upstream-error
calls. The candidate arm is withheld after two genuine FULL failures.

This is baseline/session-reliability evidence, not a policy failure and not a
zero-saving treatment row. It demonstrates the fail-closed admission rule: a
multi-task candidate can be scored only when FULL first shows that the model,
agent, prefix, and clean task workspace can solve the current issue.
