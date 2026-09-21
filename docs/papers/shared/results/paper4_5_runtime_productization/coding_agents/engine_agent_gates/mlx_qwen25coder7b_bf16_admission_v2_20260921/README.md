# MLX Qwen2.5-Coder-7B BF16 mini-swe-agent admission, 2026-09-21

This diagnostic tests whether the exact cross-engine bridge model is capable of
producing a plain mini-swe-agent success before any PRA arm is allowed. The
endpoint loaded `Qwen/Qwen2.5-Coder-7B-Instruct` BF16 at source revision
`c03e6d358207e414f1eca0bb1891e29f1db0e242`. The corrected MLX endpoint
independently reported that revision, an 8,192-token context ceiling, and PRA
source revision `d1a3e6a4`.

No row below is an official benchmark score. Empty-patch execution failures
were rejected before grading; the Task 7 grader also encountered a Docker
cleanup race after mini-swe-agent had already emitted an empty patch.

| Task | Scaffold | Actions | Terminal result | Diagnostic |
|---|---:|---:|---|---|
| 5, `pytest-dev__pytest-7982` | stock, 1,024 completion | 29 | timeout, empty patch | alternated the same two unsuccessful `git log` searches; no mutation |
| 3, `scikit-learn__scikit-learn-13135` | stock, 1,024 completion | 5 | context rejection | 8,523 prompt + 1,024 decode > 8,192 after unavailable-editor and oversized-file actions |
| 4, `django__django-15741` | stock, 1,024 completion | 7 | context rejection | 7,427 prompt + 1,024 decode > 8,192; no mutation |
| 4, `django__django-15741` | noninteractive-2k, 1,024 completion | 24 | context rejection | scaffold enabled mutation, but an over-broad substitution left invalid Python; 7,376 + 1,024 > 8,192 |
| 7, `django__django-14089` | noninteractive-2k, 512 completion | 30 | `LimitsExceeded`, empty patch | found the correct one-line fix but inserted a literal `\n` and repeated an ineffective repair |

The noninteractive-2k scaffold is general and task-independent. It moves the
existing noninteractive-shell rule into the system prompt and reduces each
visible oversized tool observation from the stock 10,000-character head/tail
materialization to 2,000 characters. Its checksum is recorded in every run
manifest. It improved tool behavior and prevented early observation-driven
overflow, but did not produce a valid patch.

Conclusion: this exact 7B BF16 model/scaffold/task cohort is not admitted for
the plain -> PRA-100 -> PRA-90 bridge. Running selective arms would confound
engine correctness with a failed plain-model capability gate. A subsequent
bridge must use a model/task pair with a repeatable plain success, while keeping
the model, tokenizer, precision, scaffold, context ceiling, and selection
ledger identical across engines.

Remote immutable run directories on the `.8` execution host:

- `miniswe-task05-full-strict-capfix-b`
- `miniswe-task03-full-strict-capfix-c`
- `miniswe-task04-full-strict-capfix-a`
- `miniswe-task04-full-compact2k-a`
- `miniswe-task07-full-compact2k512-a`

They are rooted at
`/Users/jorge.simao/git/rd/paper45-runs/bridge-qwen25-7b-bf16-mlx/`.

