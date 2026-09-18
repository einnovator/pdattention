# Frozen eight-identity confirmation: first qualification result

This bundle preserves the first executed cell of
`paper85-v2-frozen-confirmation8-ctx131k`.  It is a baseline-reliability
result, not a Prompt Pinned treatment result.

The first six launch attempts stopped before task execution because the
uv-managed Python on the M5 host could not use the macOS local-network path
to the M4 endpoint.  Commit `78e3a42e` makes cold unload/warmup use the same
explicit curl transport as the later health probes.  Retry 7 then passed:

- cold unload and warmup, with an exact `OK` response;
- three of three temperature-zero health probes;
- resident `qwen3-coder:30b` qualification at 131,072 context tokens;
- the official mini-swe-agent and SWE-bench execution.

The same-prefix FULL control for
`scikit-learn__scikit-learn-13135` used 14 calls and 97,604 cumulative
message-content input tokens, but scored 0/1.  It sorted the completed bin
edges after computing midpoints from unsorted K-means centers; the official
tests reject that patch.  This is an ordinary model/trajectory failure under
FULL history, not a selection failure or a submission-only failure.

The fail-closed campaign therefore launched neither Prompt Pinned nor Recent
Frontier.  Its status is `stopped_unqualified_session_interference`, and this
identity contributes no paired policy row.  The next confirmation execution
must retain this failure and use a separately predeclared sequence; it must not
retry this task until it succeeds and then pair only the favorable repeat.

Files:

- `campaign_state.json`: frozen campaign and stop receipt;
- `task01_full_control/official_result.json`: primary official outcome;
- `task01_full_control/autonomous_metrics.json`: calls and token accounting;
- `task01_full_control/run_manifest.json`: immutable environment identity;
- `task01_full_control/preds.json`: submitted patch;
- `task01_full_control/persistent_episode_export.json`: auditable trajectory;
- `task01_full_control/request_selection.jsonl`: FULL request ledger.
