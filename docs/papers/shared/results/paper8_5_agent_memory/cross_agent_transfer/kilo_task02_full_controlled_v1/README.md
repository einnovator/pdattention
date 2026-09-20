# Kilo Task-2 controlled FULL admission

Kilo 7.7.5 ran the same locked Task 2 and model control used for Pi. The
compatibility proxy audited 23 complete requests with temperature 0, top-p 1,
seed 0, a 1,024-token completion ceiling, FULL message retention, and native
tool-call/result identities intact.

Kilo produced the same minimal one-line Django fix in 23 model calls and 22
native tool calls. The official SWE-bench 4.1.0 per-instance report resolves
the task: 1/1 FAIL_TO_PASS and 29/29 PASS_TO_PASS tests passed. As in the Pi
run, a Docker stale-container race affected only aggregate-report cleanup after
the authoritative result was written.

The event stream exposes an agent-normalization issue important to Paper 8.5.
Kilo labels all 22 tool events `completed`, yet four outputs contain failure
signals, including missing pytest, a missing manage.py, and a traceback.
Therefore portable policy metadata cannot equate transport completion with
semantic success; it needs explicit exit status or failure evidence.

Kilo reports 391,091 cumulative logical input tokens including its system
prompt and tool schemas. These totals are paired only within Kilo. The raw
event stream is stored outside Git at
`D:/git/rd/p85-kilo-transfer-raw/kilo_task02_full_controlled_v1_events.jsonl`;
its digest is in `event_summary.json`.
