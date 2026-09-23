# Same-model agent failure audit

Locked Easy-14 Task 4 (`django__django-15741`) provides a clean diagnostic of
why agents can differ even when configured for the same Qwen3-Coder-30B
endpoint at temperature zero. Pi and Kilo both found the correct
`str(format_type)` repair but treated a provider length stop as task
completion. The historical PRA Agent control found the same repair and then
terminated on future-tense prose without mutating the workspace. OpenHands
continued for 43 actions, implemented a 1,049-byte patch, and resolved the
official task.

The identifiable treatment is therefore the agent loop, not merely the base
model. Prompts, tool affordances, observation rendering, stop handling and
completion guards change the model's input and determine whether a correct
diagnosis becomes a submitted patch. OpenHands demonstrates robustness here,
but its 43-action path is substantially more expensive than the prematurely
terminated paths.

This audit motivated two generic PRA Agent repairs: reject apparent completion
until a tracked patch exists, and continue after provider length/max-token
termination. A fresh FULL rerun is required before those repairs may replace
the historical 2/3 admission result.

The historical OpenHands artifact names the same configured served model but
does not contain an observed digest. Therefore the strongest present claim is
same configured endpoint/model, not fully audited bit-identical model identity.
All new cross-agent admissions must validate and record the observed digest
before execution.
