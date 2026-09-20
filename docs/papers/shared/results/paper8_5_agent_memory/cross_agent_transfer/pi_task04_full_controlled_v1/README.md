# Pi Task-4 FULL controlled admission

Pi 0.75.3 ran `django__django-15741` with the frozen model and FULL native
history. The control is a primary failure: after seven model calls and six
tools, the final response reaches the 1,024-token completion ceiling while
explaining the correct `str(format_type)` direction but before issuing a write.
Pi treats the length-stopped reasoning-only response as terminal and exports an
empty patch.

This is not a memory-policy failure because FULL selected every record. It is
also not evidence that the model lacks the repair: the terminal response states
the correct diagnosis. It exposes a harness portability requirement: a coding
agent must distinguish semantic completion from provider `finish_reason=length`
and either continue the turn or declare an incomplete outcome.

The frozen 1,024-token run remains the primary plain-three result. A separately
declared 2,048-token diagnostic may test the completion-horizon explanation,
but cannot replace or be pooled with the frozen control.

