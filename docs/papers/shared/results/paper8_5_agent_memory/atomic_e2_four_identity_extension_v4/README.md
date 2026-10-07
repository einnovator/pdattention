# Atomic-E2 fourth-identity extension

This bundle extends the post-sampler Atomic-E2 cohort with
`django__django-12741` at episode depth six.  Every execution uses the same
five-episode immutable prefix, Qwen3-Coder-30B Q4_K_M digest, tokenizer,
temperature-zero/top-k-20 sampler, mini-swe-agent 2.4.6 scaffold, ordinary
Bash capability, whole-record selection, and official SWE-bench grading.
The selector receives no evaluator task ID or explicit task-boundary marker.

## Admitted pair

| Arm | Official | Calls | Own counterfactual | Selected | Own saving | Paired saving |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL r01 | yes | 12 | 314,664 | 314,664 | 0% | 0% |
| Atomic-E2 r01 | yes | 12 | 313,997 | 128,393 | 59.11% | 59.20% |

The schema-2 strict validator accepts all 34 frozen execution-identity fields
and the shared pair ID.  Atomic E2 preserves the official solve and call count,
excludes 185,604 logical tokens, and records no reacquisition event.

## Updated conditional-preservation cohort

Combining this pair with `atomic_e2_three_identity_repeat_v3` gives:

| Task identities | Paired executions | FULL / Atomic-E2 solves | Calls (F/E2) | Own saving | Paired saving |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 7 | 7/7 / 7/7 | 71 / 57 | 52.37% | 60.18% |

Atomic E2 selects 669,087 of 1,404,642 tokens on its own seven
trajectories.  The paired FULL trajectories consume 1,680,162 tokens.  Only
the 52.37% own coordinate is an engine-portable pre-rounding target; the
60.18% paired coordinate also includes trajectory contraction.

This is observed 100% *conditional preservation* over four admitted task
identities, not a population-accuracy or production-default estimate.  The
fourth identity has one admitted pair but is not repeat-qualified because its
second FULL control fails.

## FULL admission reliability

The contemporaneous screen also records three ineligible controls:

| Identity / repeat | Calls | Official result | Atomic-E2 treatment |
| --- | ---: | --- | --- |
| `scikit-learn__scikit-learn-13135` r01 | 6 | unresolved; terminal submission without the required mutation | not run |
| `sphinx-doc__sphinx-8721` r01 | 11 | unresolved submitted patch | not run |
| `django__django-12741` r02 | 9 | unresolved submitted patch | not run |

The extension therefore admits one of four FULL executions and one of three
screened task identities.  These failures are baseline reliability evidence,
not selection failures.  A historical Easy-14 plain-success label is not a
current admission guarantee under a different prefix and stochastic backend
trajectory, even at temperature zero.

The initial Scikit launch that failed before inference because a tokenizer
path alias was absent is quarantined on the remote host and is not represented
in this bundle.  The alias was restored to the unchanged tokenizer files
before any counted execution.

## Provenance and claim boundary

- Execution revision: `9bb5f61c`.
- Model digest: `4abd6222c3c1a34f94cc04542bfe08523213f915b59b4a6443059ef741e8d90f`.
- Prefix digest: `c95cbae606a68e30f776849df0dc09c84caa09430479d45fda30b4cdf1442283`.
- Remote immutable root: `paper85-atomic-e2-current-expansion-v1` on the
  48-GB M4 host.
- This is logical-policy evidence.  It is not native-K/V, latency, or
  cross-engine autonomous evidence.

Each preserved run contains its manifest, official result, autonomous metrics,
request-selection ledger, persistent-episode export, and auxiliary-workspace
status.  Model-visible request bodies and Docker workspaces remain under the
immutable remote root because they are large diagnostic artifacts.
