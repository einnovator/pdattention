# PRA Agent boundary-free persistent FULL control

This bundle records the first clean three-issue persistent FULL control after
three scaffold repairs: the PRA task graph is disabled in boundary-free mode,
rejected actions are no longer projected as executed, and exhausted read-only
tools are withdrawn from the disclosed palette.  One continuous PRA Agent
session uses a fresh official SWE-bench container for each issue and preserves
all prior conversation records.

The independent SWE-bench grader resolves all three issues.  The cumulative
prefix workload is 184,147, 393,487 and 708,506 exact-token history tokens at
N=1, 2 and 3.  Requests per episode are 15, 10 and 10; executed tool events are
10, 6 and 9.  This is an admission/control result, not a saving result.

The paired E0 execution is still pending.  A quarantined attempt produced the
same 689-byte Task-1 patch at 100% realized retention but crashed only while
exporting an unbound rejected action.  Commit `e1377d0d` fixes that evidence
export path; the interrupted and endpoint-load attempts are excluded from all
aggregates.

Key artifacts:

- `run_manifest.json`: complete N=3 execution and cumulative accounting;
- `selection_trace.json`: per-request FULL identity ledger;
- `official_grade/report.json`: independent 3/3 resolution report;
- `episodes/*/model.patch`: submitted source patches;
- `episodes/*/tool_events.jsonl`: executed/rejected runtime outcomes;
- `qualification.json`: compact admission decision.

