# PRA Agent Task 4 completion-guard auxiliary result

This immutable ledger preserves the first post-fix PRA Agent run that crossed
the historical premature-completion failure on `django__django-15741`.

The agent diagnosed the lazy-string failure, edited
`django/utils/formats.py`, and produced a patch that the official SWE-bench
grader resolved.  The run is **not** a qualified FULL control: after 20
successful model requests, request 21 failed with HTTP 500 `EOF`.  The final
successful request reported 32,734 prompt tokens and 97 completion tokens
(32,831 total), while the loaded Ollama runner had a 32,768-token serving
window.  The model catalog's 262,144-token training capability therefore did
not describe the active serving allocation.

This is auxiliary capability evidence only.  It demonstrates that the causal
text rendering and completion guard repaired the earlier empty-patch stopping
mode, but it cannot establish end-to-end FULL reliability.  A clean retry uses
the same weights through the separately identified
`qwen3-coder:30b-ctx131k` serving alias and must finish without transport
failure before entering the primary accuracy table.

The old raw manifest reported `tool_event_count=0` because it only inspected a
returned terminal turn.  The durable exported session contains 20
`tool_response` records.  The runner was subsequently corrected to derive this
count from durable session state even when a later request fails.

Raw run location on the execution host:

`/Users/jorge.simao/git/rd/p85-refactor-revalidation-6ccf/.pra/pra-agent-task04-causal-full-v2/`

