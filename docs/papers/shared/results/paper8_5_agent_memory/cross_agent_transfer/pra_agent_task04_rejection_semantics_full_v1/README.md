# PRA Agent Task-4 rejection-semantics FULL admission

This bundle records the first clean Easy-14 Task 4 FULL execution after PRA
Agent stopped labelling host-rejected actions as executed.  The frozen
Qwen3-Coder-30B 64K serving alias used temperature 0, top-p 1 and seed 0.

The model executed eight tools before the source-mutation floor, wrote an
untracked scratch test, and then proposed two read-only verification commands.
Both commands were rejected and durably represented as **not executed**.  The
model recovered, edited `django/utils/formats.py`, simplified its first edit,
inspected the result, and terminated normally.  The independent SWE-bench
grader resolved the 701-byte patch.

This is a FULL agent-scaffold qualification, not a memory-saving result.  It
supports the causal diagnosis that the historical empty-patch failure was an
agent-loop/protocol failure rather than insufficient base-model capability.
The run still motivates a source-only mutation guard: the scratch test was not
part of `model.patch`, but it consumed one tool action.

