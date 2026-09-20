# OpenCode admission preflight

OpenCode 1.18.31 was built inside the locked Django-15277 SWE-bench image on
the Medium Mac (`192.168.1.8`). The container reports the pinned version, and
the hermetic configuration exposes exactly one model:
`pra/qwen3-coder:30b`. Both the primary and auxiliary `small_model` identifiers
resolve to that same controlled endpoint.

This preflight qualifies the runner image and configuration only. No model
request was made because the pinned `.6` endpoint was unhealthy. Consequently
the artifact contains no task success, token-saving, or policy-transfer claim.
The next admissible cell is `FULL` on the first locked task after endpoint
identity and health checks pass.
