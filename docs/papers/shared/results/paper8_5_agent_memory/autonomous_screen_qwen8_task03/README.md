# Qwen3-8B Task-03 FULL qualification screen

This is a negative baseline-admission result, not a memory-policy arm.

- Task: `scikit-learn__scikit-learn-13135` (locked Easy-14 index 3).
- Model: `qwen3-8b-pra-32k:latest`, Q4_K_M, explicit 32,768-token context.
- Agent: mini-swe-agent, ordinary full-history text, temperature 0, seed 0.
- Outcome: 20 requests / 19 actions and an empty submitted patch.
- Selection: FULL throughout, 123,279 materialized of 123,279 cumulative
  message-content tokens.

The SWE-bench reporter subsequently failed with a stale-container 404 after
printing `No instances to run`. That grading fault is separately preserved in
`grader.log`, but it does not make the model/task pair ambiguous: `preds.json`
contains an empty `model_patch`, so the FULL pair cannot qualify as a plain
success. No compressed arm is admitted and this run is excluded from the
quality--saving curve.

The preceding Qwen3-14B attempt on the 16-GB M5 was also excluded: an explicit
32K context consumed about 11.6 GB of model/KV residence and did not complete
the first request under memory pressure. That was a hardware-capacity failure,
not a quality observation.
