# Recovered remote diagnostics

These artifacts were recovered on 2026-09-08 from the M5 worktree
`/Users/jorge.simao/git/rd/pdattention-paper4-easy50` at detached commit
`9306a19b`. They are retained outside the canonical campaign-cell directories
and are not imported as completed efficacy or transport-qualification cells.

| Recovered directory | Observed state | Result | Evidence disposition |
| --- | --- | ---: | --- |
| `gateway_passthrough_unqualified_target/` | complete, 50 official chunks | 17/50, two timeouts | Diagnostic only. The manifest preflighted an ephemeral local treatment proxy and did not bind or verify its upstream endpoint, so it cannot establish that the corrected product G00 route was exercised. Relative to the 14/50 baseline, the discordance was seven gains and four losses (exact paired McNemar p=0.549). |
| `pra_selected_50_invalid_unqualified_target/` | complete, 50 official chunks | 0/50 | Invalid as G10 efficacy evidence. The local selector emitted detached `pra.resources`, but the manifest did not bind or verify a G10 consumer. Selected-resource consumption therefore is not established. |
| `truncation_25_interrupted/` | interrupted after five official chunks | no aggregate | Partial diagnostic only. The stale scheduler launched 25% truncation before the preregistered 50% matched control. The process was stopped after the M4 inference endpoint became unavailable. |

The recovered copies exclude Hugging Face cache locks and nested official-grader
scratch logs whose POSIX names are not portable to Windows. Run manifests,
top-level logs, normalized results, treatment telemetry, per-chunk receipts, and
agent trajectories are retained. The original complete trees remain on the M5.
