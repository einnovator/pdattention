# Cold-cache held-out five-episode-prefix qualification

This bundle freezes five completed independent SWE-bench episodes and evaluates
`django__django-15741` as episode six with Qwen3-Coder-30B, temperature zero,
top-p one, seed zero, and a 1,024-token completion cap.

The important control is trial-start cache state. Two cold-reset FULL trials
both resolve in 13 calls and reproduce the complete assistant-action and
assistant-content sequences. A FULL trial started with prior live Ollama cache
state diverges at request four and fails in nine calls. A further cold FULL
trial matches the qualified trajectory through six requests but is quarantined
after a transport failure. Policy arms are therefore compared only after an
explicit model unload and the same two-token `Reply OK.` warmup.

| Arm | Official | Calls | Materialized input | Own-trajectory saving | Paired input saving | Failure-aware |
|---|---:|---:|---:|---:|---:|---:|
| Cold FULL (two exact repeats) | 2/2 | 13, 13 | 361,680 each | 0% | 0% | 0% |
| M2/P1 | 0/1 | 21 | 356,651 | 41.13% | 1.39% | 0% |
| Prompt-pinned E2+F1C | 2/2 | 8, 8 | 125,643 each | 40.39% | 65.26% | 65.26% |

M2/P1 demonstrates why per-request retention is insufficient: it removes
41.13% of its own counterfactual history, but eight extra calls erase almost
all paired saving and the submitted patch fails. The trajectory repeatedly
reads and line-edits the target and accidentally deletes core logic.

The corrected E2+F1C profile preserves every genuine user prompt, keeps the two
most recent completed instruction epochs whole, keeps the active epoch whole,
and represents each of three older completed epochs with a compact closure
receipt. On the first request it materializes 14,608 of 25,251 tokens (42.15%
saving); the three receipts cost 66 tokens and replace 1,190 tokens of original
finalization text. Both cold repeats solve in eight calls with identical
selected-message, assistant-action, assistant-content, and submitted-patch
sequences. Its 40.39% own-trajectory saving is inside the paper's 30--50%
target; its paired saving is larger because it uses five fewer calls than FULL.

This is one repeat-qualified held-out task, not a population accuracy estimate.
The next gate is the same cold-matched FULL/candidate pair on additional task
identities. Latency is not reported because the model host was under unrelated
indexing and endpoint-security load.

`summary.json` is the compact machine-readable reduction. Raw result directories
retain manifests, selection traces, mini-swe-agent trajectories, submitted
patches, official outcome summaries, and execution sidecars. The Windows copy
omits the nested official-grader directory whose model-name component contains
a colon; `official_result.json`, `grader.log`, and the submitted patch retain
the scored outcome needed by this analysis.
