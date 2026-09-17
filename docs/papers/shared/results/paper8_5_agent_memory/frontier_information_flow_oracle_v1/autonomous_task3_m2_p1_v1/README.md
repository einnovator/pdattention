# Task 3 recent-frontier DAG M=2 plus P1

P1 retains the latest harness-certified valid protocol-completion causal
bundle. It rejects a more recent malformed completion as an exemplar and acts
as a reachability barrier so task-specific predecessors remain retireable.

- instance: `scikit-learn__scikit-learn-12585`
- official outcome: resolved
- calls: 7, versus 6 for historical paired FULL
- cumulative materialized input: 92,621 message-content tokens
- historical paired FULL input: 88,510 tokens
- paired saving: -4.64%
- own-trajectory full counterfactual: 104,661 tokens
- own-trajectory pruning: 11.50%

The source mutation matches the two rejected M2 runs, but the submission is
now a valid unified git diff and passes official grading. This is a targeted
repair of protocol behavior, not a claim that P1 improves task reasoning or
that Task 3 is individually efficient.

The JSON metrics, official result, manifest, and trajectory are copied
verbatim from the immutable clean remote run directory. The earlier stopped
zero-exclusion implementation is quarantined and is not evidence.
