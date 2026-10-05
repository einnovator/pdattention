# Atomic-E2 three-identity repeat cohort

This bundle extends the post-sampler Atomic-E2 cohort with a held-out
`django__django-15368` cluster at episode depth six.  Every run uses the same
five-episode immutable prefix, Qwen3-Coder-30B Q4_K_M digest, tokenizer,
temperature-zero/top-k-20 sampler, mini-swe-agent 2.4.6 scaffold, ordinary
Bash capability, and whole-record selection.  The selector receives no
evaluator task ID or explicit task-boundary marker.

## New held-out cluster

| Repeat | FULL solve / calls / tokens | Atomic-E2 solve / calls | Candidate counterfactual / selected | Own saving | Paired saving |
| --- | --- | --- | ---: | ---: | ---: |
| r01 | yes / 7 / 183,502 | yes / 9 | 237,577 / 98,374 | 58.59% | 46.39% |
| r02 | yes / 11 / 292,227 | yes / 7 | 183,477 / 75,208 | 59.01% | 74.26% |
| cluster | 2/2 / 18 / 475,729 | 2/2 / 16 | 421,054 / 173,582 | 58.77% | 63.51% |

Both pairs pass the schema-2 strict pairing validator with no identity
mismatch.  Repeat-level call deltas are +2 and -4, demonstrating residual
trajectory variance at temperature zero; the task-cluster aggregate favors
Atomic-E2 by two calls.  Both Atomic-E2 repeats and both FULL controls resolve
officially.

## Combined promotion cohort

Combining this held-out cluster with `atomic_e2_two_identity_repeat_v2` gives:

| Task clusters | Paired executions | FULL / Atomic-E2 solves | Calls (F/E2) | Own saving | Paired saving |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 6 | 6/6 / 6/6 | 59 / 45 | 50.42% | 60.40% |

Own saving is the engine-portable logical omission coordinate: Atomic-E2
selects 540,694 of 1,090,645 tokens on its own six trajectories.  Paired
saving compares those selected tokens with 1,365,498 tokens consumed by the
matched FULL trajectories and therefore also includes trajectory contraction.
Only own saving may be used as the pre-rounding target for native K/V
realization in Paper 4.5.

Observed paired preservation is 100%, but three task identities do not support
a near-100% population-accuracy or production-default claim.  The task identity
is the uncertainty unit, not the six executions.

## FULL admission screen

Two other identities that had resolved in the earlier Easy-14 persistent FULL
campaign were screened on the same five-episode prefix before policy execution:

| Identity | Calls | Official result | Treatment run |
| --- | ---: | --- | --- |
| `sympy__sympy-23534` | 11 | fail: patch apply / empty effective patch | not run |
| `psf__requests-2317` | 10 | unresolved | not run |
| `django__django-15368` | 7, 11 | 2/2 resolved | Atomic-E2 paired twice |

These failures are unconditional FULL reliability evidence, not Atomic-E2
failures.  They also show that a task solved in one persistent sequence is not
automatically admitted under a different historical prefix.  Selective quality
must therefore be reported conditionally on contemporaneous matched FULL
success, while FULL admission reliability remains a separate outcome.

## Provenance and claim boundary

- Execution revision: `48158c1c62893387a962b984b97cafeedbbbed98`.
- Model digest: `4abd6222c3c1a34f94cc04542bfe08523213f915b59b4a6443059ef741e8d90f`.
- Prefix digest: `c95cbae606a68e30f776849df0dc09c84caa09430479d45fda30b4cdf1442283`.
- Remote immutable roots: `paper85-atomic-e2-heldout-django15368-v1`,
  `paper85-atomic-e2-heldout-sympy23534-v1`, and
  `paper85-atomic-e2-heldout-requests2317-v1` on the 48-GB M4 host.
- This is logical-policy evidence.  It is not a native-K/V, latency, or
  cross-engine autonomous result.

Top-level JSON and JSONL artifacts from every counted pair and both rejected
FULL screens are preserved in the subdirectories.  Model-visible request bodies
and Docker workspaces remain in the immutable remote roots because they are
large diagnostic artifacts rather than publication inputs.
