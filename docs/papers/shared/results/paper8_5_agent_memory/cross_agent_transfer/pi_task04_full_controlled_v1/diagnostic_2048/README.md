# Pi Task-4 2,048-token completion diagnostic

This diagnostic changes only the per-response completion ceiling from 1,024 to
2,048 tokens. It is not part of the frozen plain-three denominator.

The larger ceiling prevents the exact seven-call length-stop seen in the
primary control, but it does not recover the task. Pi takes 14 model calls and
10 tool calls, creates an untracked reproduction script, and submits no source
patch. Three automatic retries follow model responses for which the client
reports `Stream ended without finish_reason`; the final run contains four
`agent_end` events. The result therefore rejects the simple explanation that
raising the token ceiling alone repairs Task 4. Agent-loop continuation and
provider stream termination both require qualification.

