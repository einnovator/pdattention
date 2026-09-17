# Autonomous Task-5 M=2 paired-success pilot

This run forks the exact saved four-issue persistent-FULL prefix and evaluates
whole causal-group information-flow retirement with the last two user prompts
as the live frontier.

| Arm | Official result | Calls | Cumulative materialized input |
|---|---:|---:|---:|
| Historical persistent FULL | 1/1 | 12 | 264,204 |
| Frontier DAG M=2 | 1/1 | 12 | 137,055 |

The treatment saves **48.13% on the paired denominator**, loses no success, and
adds no calls. Relative to its own 265,563-token full-history counterfactual,
it saves 48.39%.

The treatment does not reproduce exact assistant text or commands: it diverges
at the first action and submits a smaller but officially resolving patch. That
is a legitimate policy-quality outcome, not 100% implementation parity. The
primary endpoint is official resolution and cumulative work; exact action
agreement remains a diagnostic.

This is one task identity and one historical FULL execution, so it is a
qualifying mechanism point rather than an accuracy estimate or default-profile
promotion. The next test is a repeated M=2 execution plus at least two more
paired FULL-success tasks.
