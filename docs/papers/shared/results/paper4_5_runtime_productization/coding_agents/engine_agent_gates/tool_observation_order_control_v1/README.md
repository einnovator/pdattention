# Tool-observation order prerequisite for agent parity

Paper 8.5 commit `3974b027` isolates a cross-runtime agent confound using
mini-swe-agent, Qwen2.5-Coder-14B revision `29efdbab55a1`, temperature zero,
and SWE-bench Verified `django__django-15277`.

The successful direct-MLX Paper 4.5 control and two failed ordinary-MLX FULL
controls have identical system text, task text, first assistant response,
first recursive-`grep` command, return code, and match set. The first tool
observation differs only in line order. The next assistant response diverges.
The two fresh failures are byte-identical over seven actions and submit the
same malformed patch. An opt-in lexicographic normalization removes this
ordering variance but does not restore task success.

Consequently, cross-engine exact-action parity requires equality of the full
ordered model-visible observation stream. If observation hashes differ first,
the fork is environmental/tool-level and cannot diagnose K/V consumption. If
observations match and the assistant response differs, numerical consumer or
backend sensitivity remains a candidate. Official resolution, call count, and
patch identity are reported independently.

The complete run manifests and request ledgers remain in Paper 8.5 under
`cross_agent_transfer/mini_swe_qwen25_14b_tool_order_control_v1` and
`mini_swe_qwen25_14b_additional_admission_v1`.
