# PRA Agent Task-2 FULL admission

This directory is the first controlled SWE-bench admission for the record-native
PRA Agent on locked Easy-14 Task 2, `django__django-15368`, using
`qwen3-coder:30b`.

The official grader resolves the issue: 1/1 FAIL_TO_PASS and 29/29 PASS_TO_PASS
tests pass. The agent makes 9 model requests and 8 typed tool calls, emits the
same minimal 671-byte patch as the Pi and Kilo controls, and reports 86,626
cumulative prompt tokens, 16,353 cached prompt tokens, and 1,356 completion
tokens. These totals are an agent-specific FULL baseline and must only be used
in paired PRA-Agent comparisons.

The admission also exposed and corrected four implementation defects before
the successful run:

- native OpenAI `tool_calls` with null text were previously treated as the
  literal answer `None`;
- replaying the transient `<tool_call>` control token as ordinary assistant
  history caused Qwen/Ollama follow-up failures;
- intermediate assistant actions and their arguments were not durable records;
- tool-handler errors terminated the process instead of becoming recoverable
  typed observations.

This is compatibility evidence, not a retention-policy estimate and not a
native-K/V result. `official_report.json` and `official_instance_report.json`
are authoritative for task resolution; `qualification.json` is the compact
machine-readable decision record.
