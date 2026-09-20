# OpenHands Task 1 live execution-receipt smoke

This immutable artifact is the first prospective OpenHands run through the
generic execution-receipt side channel. It uses OpenHands SDK 1.49.2,
`qwen3-coder:30b`, temperature zero, disabled native condensation, and FULL
logical history on `django__django-15277`.

All 27 model requests succeeded. The agent emitted and delivered one receipt
for each of 27 action/observation pairs. By the last model request, the proxy
had joined the preceding 26 receipts; the finalization receipt correctly has
no later generation to consume it. Nine FileEditor receipts carry exact file
versions. Sixteen generic Terminal operations remain unknown-effect barriers.

The run produced the same 689-byte patch (SHA-256
`8512d6bc5111e1339c1786cd5be241461e34fd8ee13a81a04afa5257ef244aae`)
as the prior independently graded OpenHands FULL control, which resolved the
issue with 2/2 FAIL_TO_PASS and 158/158 PASS_TO_PASS tests. This smoke was not
independently regraded, so patch identity is supporting evidence rather than a
second official resolution observation.

The run also found a bridge regression: the receipt held exact
`version_before`/`version_after` values, but canonical DAG metadata did not
populate the legacy pre/post version maps. Commit `d21c4008` fixes that mapping
and adds a regression test. FULL selection made the defect outcome-neutral in
this run; no selective policy may claim the version evidence until a post-fix
prospective execution passes.

`qualification.json` is the compact evidence ledger. `openhands_events.jsonl`
contains native events and receipts, while `proxy_trace.jsonl` contains the
ordinary-text request audit. The transparent TCP relay changed no payload; it
only bridged a macOS local-network permission difference between system Python
and the experiment venv.
