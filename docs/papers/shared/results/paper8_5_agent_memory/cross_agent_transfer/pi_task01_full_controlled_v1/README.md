# Pi Task-1 FULL controlled admission

Pi 0.75.3 ran the first locked Easy-14 identity,
`django__django-15277`, with the frozen `qwen3-coder:30b` revision,
temperature 0, top-p 1, seed 0, a 1,024-token completion ceiling, and FULL
native history.

The run officially resolves the task. It makes the minimal guarded
`MaxLengthValidator` change and passes 2/2 FAIL_TO_PASS plus 161/161
PASS_TO_PASS tests. Pi uses 23 model calls and 22 native tool calls, with
403,543 cumulative logical input tokens and 2,560 output tokens.

Four tool results are semantic errors. Most importantly, three native `edit`
attempts fail before the agent recovers with Bash. The action, failed result,
and recovery must remain one atomic causal unit; transport completion alone is
not a usable success label. The temporary untracked `charfield_fix.py` is not
part of the submitted tracked patch.

The first two grader attempts are quarantined infrastructure diagnostics: one
selected an ARM image and one used a duplicated namespace. The recorded report
comes from the clean x86-tagged retry `p85-pi-task01-full-v4`.

