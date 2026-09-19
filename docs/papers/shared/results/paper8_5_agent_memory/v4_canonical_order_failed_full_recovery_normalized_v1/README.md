# Canonical-order cold-normalized failed-FULL recovery diagnostic

This is a separately declared explanatory diagnostic, not a
policy-preservation cohort. The frozen boundary-free Recent Frontier M2/P1
policy is applied to the six unresolved exact-prefix FULL controls from the
second, canonical-order eight-issue campaign. Before every cell, the runner
unloads the Ollama model with `keep_alive=0`, performs an OpenAI-compatible
warmup, requires three successful generation probes, and verifies an active
context capacity of at least 131,072 tokens. Every candidate reuses the exact
source prefix declared before execution.

The declaration binds source campaign-state SHA-256
`71e23de900be10189ced1b73991b2c17849305735e6727235a1c1c27c41df258`.
The completed diagnostic ledger has SHA-256
`87f6436348a79bc4e0dcaa07c360f7c9e74740e7221f69b43734fa4bfe76aeb3`.

| Episode | Identity | Failed FULL calls | M2/P1 calls / outcome | Candidate-own saving | Call delta | First action same |
|---:|---|---:|---|---:|---:|---|
| 2 | `scikit-learn__scikit-learn-13135` | 6 | 6 / fail | 0.00% | 0 | yes |
| 3 | `django__django-14089` | 31 | 40 / fail | 33.34% | +9 | yes |
| 4 | `sphinx-doc__sphinx-8721` | 8 | 11 / fail | 26.16% | +3 | no |
| 5 | `sympy__sympy-23534` | 12 | 14 / fail | 54.89% | +2 | yes |
| 6 | `psf__requests-2317` | 19 | 11 / fail | 59.31% | -8 | no |
| 7 | `sympy__sympy-16886` | 16 | 22 / fail | 62.82% | +6 | yes |

The candidate trajectories materialize 1,732,380 of 3,395,429
message-content tokens, a 48.98% ratio-of-sums saving on their own realized
trajectories. They use 104 calls versus 92 in the failed FULL controls. None of
the six failures is recovered. The Requests candidate terminates eight calls
earlier, but both arms fail, so that reduction is failure-termination economics
rather than a quality-preserving efficiency result.

Episode 2 is an abstention/repeatability control: M2 has no older prompt epoch
to retire, selected and FULL token workloads are identical, the first action is
identical, and the official failure repeats. More aggressive retirement becomes
available later in the sequence and reaches 62.82% candidate-own saving, but it
does not repair the baseline failures.

This order also reverses the earlier normalized Sphinx recovery. In the first
held-out order, M2/P1 recovered Sphinx while saving 36.57%; in this canonical
order, Sphinx remains unresolved while saving 26.16%. Recovery is therefore a
prefix/order-sensitive event, not a stable per-task property. Across both
orders, old-task interference is demonstrably possible, but M2/P1 is not a
general repair mechanism for unstable or capability-limited FULL trajectories.

These six cells remain outside the primary policy-preservation denominator.
The first episode of the source campaign is the separate qualified preservation
control: FULL and M2/P1 both resolve in 27 calls with exact action/input hashes
and 0% retirement. The eighth source episode ended in infrastructure failure
and is excluded pending an immutable retry.

Primary machine-readable evidence is in `diagnostic_state.json`. Each episode
directory contains the request-level selection ledger, exported typed
trajectory, official result, metrics, and run manifest.
