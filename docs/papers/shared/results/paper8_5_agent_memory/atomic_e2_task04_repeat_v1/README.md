# Repeat-qualified Atomic-E2 Task-4 result

This bundle records the first post-sampler, behaviorally matched repeat pair
for boundary-free atomic instruction-epoch retirement. The selector sees
ordinary user-prompt roots and information-flow metadata; it receives no
evaluator task ID or explicit task-boundary marker.

| Arm | Official resolution | Calls | Cumulative input tokens | Saving |
| --- | ---: | ---: | ---: | ---: |
| Full, repeat 1 | 1/1 | 12 | 216,344 | control |
| Atomic E2, repeat 1 | 1/1 | 6 | 84,760 selected; 104,530 own Full counterfactual | 18.91% own; 60.82% paired |
| Full, repeat 2 | 1/1 | 12 | 216,344 | control |
| Atomic E2, repeat 2 | 1/1 | 6 | 84,760 selected; 104,530 own Full counterfactual | 18.91% own; 60.82% paired |

Both pairs pass the schema-2 strict pairing validator added in commit
`36e18df6`. Full and Atomic E2 share the task, model digest, tokenizer,
sampler, completion ceiling, prefix digest, agent scaffold, protocol, and
ordinary shell capability. Neither arm contains a static workflow anchor,
transactional edit tool, or synthetic replacement receipt.

Atomic E2 keeps the active instruction epoch plus the two most recent closed
epochs and retires older terminally closed prompt components as whole causal
groups. Both selective repeats emit the same correct patch and solve in six
calls. Both Full controls emit the same correct patch and solve in twelve
calls. The first action differs, so this is outcome and cost preservation, not
exact trajectory preservation.

The two saving coordinates answer different questions. The 18.91% own saving
is the portable logical omission ratio on the candidate's six-call trajectory.
The 60.82% paired saving additionally reflects that the candidate terminates
in half as many calls. Paper 4.5 may map the frozen selected-record ledger to
resident K/V and report page rounding, copying, temporary memory, and latency;
it must not advertise 60.82% as a policy-only K/V saving.

This is repeated evidence on one SWE-bench identity, not a population accuracy
estimate or production-default qualification. The next gate is a second
repeat-qualified identity at episode index at least four, followed by a frozen
cross-engine realization.

One initial selective attempt stalled upstream before the first response. It
was quarantined and is not included in either denominator.
