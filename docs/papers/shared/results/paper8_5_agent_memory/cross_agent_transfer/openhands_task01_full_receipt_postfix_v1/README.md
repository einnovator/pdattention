# OpenHands Task 1 post-fix live receipt qualification

This immutable artifact is the post-fix prospective qualification of the
generic execution-receipt side channel. It uses OpenHands SDK 1.49.2,
`qwen3-coder:30b`, temperature zero, disabled native condensation, and FULL
logical history on `django__django-15277`.

All 33 model requests succeeded. The agent emitted and delivered one receipt
for each of 33 action/observation pairs. The last generation consumed all 32
receipts that precede it; the finish receipt has no later request. Thirteen
FileEditor observations expose exact versioned resources to canonical DAG
metadata. Eighteen generic Terminal actions correctly remain unknown-effect
barriers. Four semantic failures remain distinct from transport completion.

The trajectory exactly reproduces the earlier controlled OpenHands FULL totals:
33 actions, 641,509 provider prompt tokens, 5,747 completion tokens, and the
same 689-byte patch. The independent official SWE-bench evaluation resolves
the issue with 2/2 FAIL_TO_PASS and 158/158 PASS_TO_PASS tests.

This run qualifies live receipt joining and FileEditor resource-version
propagation at FULL retention. It does not qualify a selective policy, and it
does not certify arbitrary terminal commands. Such commands remain barriers
until execution middleware provides complete effect scope and resource-version
evidence.

`qualification.json` is the compact evidence ledger. `openhands_events.jsonl`
contains native events plus emitted receipts; `proxy_trace.jsonl` records the
ordinary-text request audit and canonical metadata coverage. The transparent
localhost TCP relay changed no payload and only bridged a macOS permission
difference between the system and experiment Python runtimes.
