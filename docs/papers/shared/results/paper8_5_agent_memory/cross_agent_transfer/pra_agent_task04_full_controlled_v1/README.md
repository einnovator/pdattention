# PRA Agent Task-4 FULL admission

This directory contains the final corrected FULL admission for PRA Agent on
locked Easy-14 Task 4, `django__django-15741`, using `qwen3-coder:30b`.

The final control is unresolved. It makes six model requests and five typed
tool calls, consumes 39,491 cumulative prompt tokens (7,048 cached) and 1,584
completion tokens over 817.4 seconds, and exports an empty patch. Its final
response correctly identifies that the lazy `format_type` must be converted to
a string, then says it will implement the fix without issuing another tool
action. The official harness consequently has no submitted instance to grade.

Two implementation diagnostics are retained separately:

- `diagnostic_pre_qwen_function_parser/` stops before its first tool because
  the provider emits a terminal Qwen `<function=...>` block rather than an
  OpenAI `tool_calls` object.
- `diagnostic_pre_malformed_recovery/` continues after that parser repair and
  leaves an incomplete 816-byte patch. It then emits malformed action JSON.
  The official instance report applies the patch but fails both FAIL_TO_PASS
  tests while passing all 104 PASS_TO_PASS tests. The runtime now records a
  malformed explicit action as a rejection observation and asks for a fresh
  action; it never repairs and executes malformed arguments.

After both compatibility repairs, the frozen final control still fails without
a patch. PRA Agent therefore completes the plain-three admission at 2/3, and
only Tasks 1 and 2 may enter its within-agent policy denominator. This is agent
capability evidence under FULL history, not evidence against a retention
policy and not a native-K/V result.
