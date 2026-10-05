# Atomic-E2 two-identity repeat cohort

This bundle combines two post-sampler, schema-2-qualified mini-swe-agent task
clusters. Both use Qwen3-Coder-30B Q4_K_M at effective top-k 20, ordinary
Bash, whole-record deletion, no static workflow anchor, no transactional edit
tool, and no synthetic replacement receipt.

| Task / episode | Repeats | FULL / E2 solves | Calls (F/E2) | Own saving | Paired saving |
| --- | ---: | ---: | ---: | ---: | ---: |
| pytest-7982 / N=4 | 2 | 2/2 / 2/2 | 24 / 12 | 18.91% | 60.82% |
| scikit-14496 / N=6 | 2 | 2/2 / 2/2 | 17 / 17 | 57.09% | 56.77% |
| Aggregate | 4 pairs | 4/4 / 4/4 | 41 / 29 | 45.17% | 58.74% |

Atomic E2 keeps the active genuine-user instruction component and the two most
recent terminally closed components whole. Older closed components retire as
complete causal groups. The selector receives neither SWE-bench task IDs nor
explicit episode boundaries.

The N=6 replication is the key depth result: retiring three older closed
epochs omits 57.09% of the candidate trajectories' logical history while
preserving both official solves and tying aggregate calls. Repeat-level call
deltas are -1 and +1, so the honest result is no aggregate call increase, not
exact action-trajectory parity. Both candidates independently emit the same
officially resolving patch; the FULL repeats emit two different resolving
patches, confirming residual trajectory variance at temperature zero.

Across both task clusters, the predeclared 30--50% aggregate own-saving target
is observed at 45.17%, with no lost paired success and calls reduced 41 to 29.
The 58.74% paired coordinate additionally includes shorter trajectories and
must not be mapped directly to resident K/V savings. Paper 4.5 should freeze
the exact selected-record ledgers and expect the 45.17% logical ratio before
engine page rounding; copying, scratch memory, re-encoding, and latency remain
separate runtime outcomes.

The uncertainty unit is the task identity, and there are only two. This is a
promotion cohort, not a population-accuracy estimate or production default.
The next quality gate is held-out task clusters; the next systems gate is the
same frozen ledger realized from resident original-position K/V.
