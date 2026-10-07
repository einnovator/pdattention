# Current SGLang policy-transfer control

On `django__django-15277`, the standard mini-swe-agent scaffold resolves under
Full, PRA-100, and matched-token-tail at 90%. All three use six calls and submit
the same patch. PRA-100 matches all six assistant contents and commands. The
selective arm first changes reasoning text at request 4 but retains all six
commands.

The selective arm saves 4.45% of its own materialized message-content tokens
and 4.09% of paired input-plus-completion work against Full. This is a positive
quality point but a low-yield saving point, as expected for a six-call single
task. It is not evidence for the 30--50% persistent-session target.

The run also separates policy from scaffold effects. Additional workflow
anchors and transactional-edit restrictions caused baseline localization and
fixed-point failures before any selection. Those arms are diagnostic controls,
not PRA-policy failures, and are excluded from the quality--saving curve.

Paper 4.5 owns the corresponding K/V residence, copy, suffix-graft, temporary
memory, and latency accounting.
