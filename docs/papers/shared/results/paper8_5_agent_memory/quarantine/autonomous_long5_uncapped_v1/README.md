# Quarantined uncapped autonomous long-five launch

This launch is excluded from every Paper 8.5 accuracy--saving estimate. The
campaign configuration at repository commit `84e1697f` encoded
`max_completion_tokens: null`, whereas the qualified Qwen3-Coder controls used
a 1,024-token ceiling. The execution identities therefore do not match.

Two tasks produced four FULL-control outcomes before the defect activated:
three primary official resolves and one malformed-submission failure. These
outcomes are retained only as operational diagnostics and are not reused as
controls for a capped treatment.

On `django__django-15741`, request two failed to emit EOS and the Ollama
llama.cpp server reported more than 10,774 generated tokens by 18:19 WEST on
2026-09-13. The task had recorded only its first completed request. The
campaign and its task container were terminated, while the original remote
artifacts were preserved under
`/Users/admin.jorge.simao/git/rd/paper85-runs/autonomous-long5-v2` on the 48 GB
M4 host.

The replacement campaign changes its campaign ID, uses a new output root,
freezes `max_completion_tokens: 1024`, and rejects an absent or non-positive
ceiling before launch.
