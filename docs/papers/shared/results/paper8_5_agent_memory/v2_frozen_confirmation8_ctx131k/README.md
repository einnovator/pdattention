# Frozen eight-identity confirmation: repeated first-identity qualification

This bundle preserves both predeclared policy cells attempted for the first
identity of `paper85-v2-frozen-confirmation8-ctx131k`.  They are
baseline-reliability results, not Prompt Pinned or Recent Frontier treatment
results.

The first six launch attempts stopped before task execution because the
uv-managed Python on the M5 host could not use the macOS local-network path
to the M4 endpoint.  Commit `78e3a42e` makes cold unload/warmup use the same
explicit curl transport as the later health probes.  Retry 7 then passed:

- cold unload and warmup, with an exact `OK` response;
- three of three temperature-zero health probes;
- resident `qwen3-coder:30b` qualification at 131,072 context tokens;
- the official mini-swe-agent and SWE-bench execution.

The first same-prefix FULL control for
`scikit-learn__scikit-learn-13135` used 14 calls and 97,604 cumulative
message-content input tokens, but scored 0/1.  It sorted the completed bin
edges after computing midpoints from unsorted K-means centers; the official
tests reject that patch.  This is an ordinary model/trajectory failure under
FULL history, not a selection failure or a submission-only failure.

The independently cold-started FULL control for the predeclared Recent
Frontier cell reproduced the result: 14 calls, 97,604 cumulative input tokens,
and 0/1 official resolution.  All 14 tuples of assistant-command,
assistant-content, and selected-message digests are identical between the two
clean controls.  Two earlier attempts in that cell are quarantined as
infrastructure failures: the reconstructed environment first lacked
mini-swe-agent and then lacked the SWE-bench grader.  The clean retry pins the
same mini-swe-agent 2.4.6 and SWE-bench 4.1.0 versions as the original
manifest.

The fail-closed campaign therefore launched neither Prompt Pinned nor Recent
Frontier.  Both cells have status `stopped_unqualified_session_interference`,
and this identity contributes no paired policy row.  The next confirmation
execution must retain these failures and use a separately predeclared
baseline-eligible sequence or missingness rule; it must not retry this task
until it succeeds and then pair only the favorable repeat.

Files:

- `campaign_state.json`: frozen campaign and stop receipt;
- `task01_full_control/official_result.json`: primary official outcome;
- `task01_full_control/autonomous_metrics.json`: calls and token accounting;
- `task01_full_control/run_manifest.json`: immutable environment identity;
- `task01_full_control/preds.json`: submitted patch;
- `task01_full_control/persistent_episode_export.json`: auditable trajectory;
- `task01_full_control/request_selection.jsonl`: FULL request ledger.
- `task01_recent_frontier_full_control/*`: the independent clean FULL repeat
  preceding the withheld Recent Frontier arm.
