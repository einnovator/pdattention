# PRA Agent Task 1 fixed-sampling FULL control

This bundle supersedes the earlier PRA Agent Task 1 admission as the
comparable `FULL` control. The earlier remote transport did not put
temperature, top-p, or seed on the OpenAI request; the Ollama server therefore
used its default temperature of 1.0 even though the campaign intended
temperature zero.

The transport now carries the frozen controls explicitly. Server-side logs
confirmed temperature 0.0 and top-p 1.0; the run manifest records seed 0 and
the observed `qwen3-coder:30b` digest. The run resolves
`django__django-15277` officially, produces the same minimal 689-byte patch
(`8512d6bc...`), and uses 15 model requests and 14 typed tool executions.
Provider accounting reports 148,225 cumulative prompt tokens, including
30,731 cached prompt tokens, and 2,245 completion tokens.

This is an admission/control result, not a selective-policy or native-K/V
result. A selective arm must use the same explicit generation controls and
must be paired against a fresh controlled FULL execution.
