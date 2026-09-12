# Quarantined replay diagnostics

`matched_token_tail_unseeded_repeatability_failure.json` is a completed strict
matched-token-tail replay whose decisions 1--20 received the full, untrimmed
history. Despite byte-identical logical input, the unseeded temperature-zero
endpoint diverged from the earlier FULL reference beginning at decision 3.
Only decisions 21--23 actually materialized fewer tokens.

The artifact is retained as evidence that temperature zero alone did not make
this endpoint trajectory-exact. It must not be interpreted as a selector-quality
result or included in the primary comparison. A fixed-seed FULL-A/FULL-B gate
precedes the replacement matched-tail and negative-heuristic replays.
