# Kilo Task-1 controlled FULL admission

Kilo 7.7.5 ran locked Easy-14 Task 1, `django__django-15277`, against the
same frozen `qwen3-coder:30b` revision used by the other transfer agents.  The
auditing proxy passed all 17 model requests through exactly at FULL retention;
temperature, top-p, seed, and the 1,024-token completion ceiling were fixed.

Kilo produced the minimal guarded `MaxLengthValidator` patch in 17 model calls
and 16 native tool calls.  The official SWE-bench 4.1.0 grader resolves the
task: 2/2 FAIL_TO_PASS and 158/158 PASS_TO_PASS tests pass.

The trajectory reports 289,550 cumulative logical input tokens, including
Kilo's own system prompt and tool schemas, and 1,425 output tokens.  These
totals are valid only for within-Kilo pairing.  Four tool outputs contain
semantic failure signals even though every Kilo tool transport status is
`completed`; portable memory policy must therefore preserve structured command
outcomes rather than infer success from the transport label.

`official_report.json` is the aggregate official report and
`official_instance_report.json` contains the complete per-instance test
evidence.  `proxy_trace.jsonl` preserves the exact FULL request audit.
