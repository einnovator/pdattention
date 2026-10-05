# Task 7 matched transactional-workflow diagnostic

This bundle replaces the earlier unmatched interpretation of the Task 7
workflow-affordance experiment.  Both primary arms use the same six-episode
persistent prefix, Qwen3-Coder-30B model digest, mini-swe-agent scaffold,
the same requested sampler fields, 1,024-token completion ceiling, static
transactional workflow anchor, and installed `pra_replace` compatibility tool.
A later live-slot audit shows that the OpenAI-compatible endpoint ignored the
requested `top_k=1` and used effective `top_k=20` in both arms.  The pair is
still internally matched, but it is not an engine-qualified top-k-one control.
The only
intended difference is Full history versus the boundary-free P0/W0 frontier
selection plan.

| Arm | Official | Calls / executed actions | Cumulative input tokens | Saving |
| --- | ---: | ---: | ---: | ---: |
| Full-v5 | 1/1 | 6 / 6 | 213,298 | control |
| Selective-v5 | 1/1 | 8 / 7 | 30,647 selected; 297,431 own Full counterfactual | 89.70% own; 85.63% paired |

The selective run has no recorded reacquisition event.  Its transactional edit
receipt is a complete, single-resource write to
`django/utils/datastructures.py`, with no unknown-effect barrier.  It produces
a different but officially resolving patch.  One of its two additional model
calls is a length-terminated response rejected before execution; the remaining
action delta is one extra focused verification.  Therefore the capability and
saving gates pass on this identity, while the strict no-call-increase gate does
not.

The `max2048_failed` arm is a separately declared generation diagnostic, not a
policy result.  Raising the completion ceiling to 2,048 tokens causes 20 calls,
19 executed actions, and an official failure despite 84.62% own-trajectory
saving.  It demonstrates that a larger completion allowance does not repair
the remaining trajectory cost and must not be pooled with the matched pair.

No engine/KV result is claimed here.  Paper 4.5 may consume only the frozen
logical selection ledger after a multi-identity no-loss/no-call-increase gate.
