# OpenHands Task 2 FULL vs boundary-free M2/P1

This is an officially graded, same-agent, same-model repeat on
`django__django-15368` using OpenHands 1.49.2 and
`qwen3-coder:30b` revision
`06c1097efce0431c2045fe7b2e5108366e43bee1b4603a7aded8f21689e90bca`.
Both arms used temperature 0, top-p 1, seed 0, the exact Qwen tokenizer, no
agent-native condensation, and the same SWE-bench image.

Both arms officially resolved the task and produced the identical 673-byte
patch. FULL used 25 actions. The boundary-free Recent Frontier M2/P1 arm used
33 actions and selected every record: a single-task session contains only one
user-instruction epoch, so M2/P1 has no older epoch that it is allowed to
retire. Its own logical and materialized saving are therefore exactly 0%.

The first assistant-content divergence is response 9, before any exclusion
(none occurs anywhere in the run). The extra eight actions and 53.91% paired
cumulative-token cost are consequently repeat/backend trajectory variation,
not a selection effect. This run is a safety/parity control and must not be
reported as a single-task saving point. Within-task policies such as corrected
H2/T4 or matched recency measure single-task savings; M2/P1 measures retirement
across persistent user-instruction epochs.

`qualification.json` is the compact evidence ledger. The full immutable
request traces remain in the campaign run directory recorded there; compact
run manifests, patches, and official reports are included beside it.
