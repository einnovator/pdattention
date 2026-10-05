# Ollama sampling-contract audit

This bundle records a fail-closed audit discovered while expanding the
Paper 8.5 M2/P1 cohort to Easy-14 Task 7 (`django__django-14089`). The
mini-swe-agent request and the selection proxy both contained `top_k=1`, but
the live llama-server slot behind Ollama's OpenAI-compatible
`/v1/chat/completions` endpoint reported `top_k=20`. Transport acceptance was
therefore not evidence that the sampler applied the requested value.

Commit `3b3cb891` adds an explicit `ollama_native` transport. It translates the
already-validated OpenAI request to `/api/chat`, maps generation controls into
Ollama `options`, converts the response back to the agent contract, captures
both raw and agent-visible responses, and fails closed for unsupported fields.
The focused and campaign regression suites pass 136/136 tests. Direct live
slot inspection then reported `top_k=1` for the corrected arm and `top_k=20`
for an explicit effective-legacy arm.

Both corrected Full controls fail official grading in seven calls. They emit
the same six shell commands, with action-sequence SHA-256
`bec8f345ff71ce8678561e753fe7c69774a30936595b52d5dc5a532337e1e961`.
Both apply an unbounded `sed` insertion after every `def __len__`, corrupting
both `OrderedSet` and `CaseInsensitiveMapping`, then submit without testing.
Thus the failure is not caused by the top-k difference at temperature zero.
It shows that the earlier one-off successful Task 7 Full control is not a
repeat-qualified admission denominator.

Consequences:

- Historical Ollama artifacts that only record requested `top_k=1` must not be
  described as engine-enforced top-k-one or greedy controls.
- Matched historical pairs remain internally paired because both arms used the
  same effective backend default, but their effective sampler is `top_k=20`.
- Task 7 is not added to the M2/P1 accuracy cohort, and no selective arm is run
  after these failed Full admissions.
- Future Ollama cohorts must use `--upstream-dialect ollama_native` and retain
  the effective-options trace; a manifest alone is insufficient.

The two counted diagnostics are under `topk1_full/` and `topk20_full/`. Earlier
launches that failed before inference or were stopped after detecting the
transport mismatch remain quarantined outside this bundle.
