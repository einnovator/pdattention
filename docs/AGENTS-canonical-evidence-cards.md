# Canonical Evidence and Agent Evaluation Work Plan

This file is the maintainer checklist for keeping Hugging Face cards, technical documentation, papers, agent reports, `pra report`, and Control Plane evidence views consistent.

## One comparison contract

Every public comparison is keyed by:

```text
task/dataset x hardware x engine x model revision x mode x profile
```

Within that exact key, report:

```text
No PRA
PRA - No Adaptor
PRA - Adaptor Bundle
```

For every measured metric, show the three absolute values, both signed deltas against No PRA, and the optional incremental adaptor delta. Never copy evidence across model revisions, quantizations, engines, modes, profiles, or hardware. Use `NOT_MEASURED`, `NOT_APPLICABLE`, or `BLOCKED` instead of zero.

## Required metric groups

Each flagship row should include the following when the engine exposes them:

| Group | Headline metrics |
| --- | --- |
| Quality | token F1, exact match, official task success, answer log-probability |
| Context | input/visible tokens, selected native K/V tokens, materialization avoidance |
| Serving | TTFT p50/p95/p99, ITL p50/p95/p99, decode output tokens/s, requests/s, completion latency |
| Resources | peak accelerator or unified memory, host RAM, active/retained detail bytes, transfer bytes |
| Cost | cost/request, cost/task, cost/successful task, GPU time/successful task |
| Routing | evidence recall, precision, MRR, AUC at a stated budget; never substitute these for answer quality |

`output_tokens_per_second` means decode-only generated-token rate excluding time to first token. Throughput rows must record concurrency. A lone unlabeled TTFT value is not publishable.

## Current model catalog

| Model | Primary engine | Current public evidence | Highest-value missing condition |
| --- | --- | --- | --- |
| Qwen3-32B 4-bit | MLX | matched natural-QA quality, context, TTFT/ITL, decode rate, completion, memory | immutable adaptor-bundle rerun |
| Qwen3-14B 4-bit | MLX | matched natural-QA quality, context, TTFT/ITL, decode rate, completion, memory | immutable adaptor-bundle rerun |
| Qwen3-8B 4-bit | MLX | matched natural-QA quality, context, TTFT/ITL, decode rate, completion, memory | immutable adaptor-bundle rerun |
| Qwen3-4B 4-bit | MLX | exact-identity routing diagnostics | three-condition end-task run |
| Llama-3.1-8B-Instruct 4-bit | MLX | structural and routing qualification | three-condition end-task run |
| Gemma-3-1B-IT 4-bit | MLX | mixed/sliding structural and routing qualification | three-condition end-task run |
| Qwen2.5-1.5B-Instruct | HF | exact-identity routing diagnostics | three-condition generation and serving run |
| Qwen2.5-Coder-1.5B-Instruct | HF | exact-identity routing diagnostics | admitted coding-agent baseline, then three conditions |
| Qwen3-0.6B | HF | research/reference mechanism evidence | larger matched end-task cohort |

Cards must expose profile and mode coverage even when the corresponding metric row is missing. The learned router remains opt-in because its QASPER improvement does not transfer uniformly to HotpotQA.

## Agent task dataset recommendation

Create `PRA-Coding-Tasks-v1` as a controlled, redistributable screening dataset before spending on full official campaigns. Target 24 to 40 repository tasks with deterministic tests and a no-PRA success band of 30-80% for the selected agent/model.

Include balanced task families:

1. Repair one failing unit test without changing the test.
2. Implement one typed function from an existing specification.
3. Fix a parser edge case with deterministic fixtures.
4. Add one CLI option and corresponding tests.
5. Repair a configuration precedence bug.
6. Make a small two-file refactor with behavioral parity.
7. Fix one lint or type-check failure that requires code understanding.
8. Locate and update a cross-file call site after an API rename.
9. Add validation and an error-path test.
10. Make a documentation-and-code example agree with the executable behavior.

Each task record should freeze repository commit, patch scope, test command, timeout, expected changed files, network policy, and checksum. Report official success, verifier checks, tool-use correctness, turns, tool calls, invalid calls, cumulative input/output tokens, TTFT/ITL samples, output tokens/s, model-call throughput, queue/inference time, wall time, memory, and cost.

Use the controlled set only for baseline screening and mechanism attribution. Promote successful configurations to public benchmarks in this order:

1. SWE-bench Lite easy strata and Commit0.
2. RepoQA/code-navigation tasks when testing selected-context discovery.
3. Terminal-Bench tasks with demonstrated model headroom.
4. SWE-bench Verified after the three-condition pilot is stable.

The current Qwen3-14B and Qwen3-Coder-30B/OpenCode cohorts remain floor controls at zero official success. The deterministic Stage-A fixture is a ceiling-level harness check, not model-quality evidence.

## Agent promotion gate

1. Run 10-25 frozen no-PRA tasks.
2. Continue only when official success is within 30-80% and at least three tasks are present.
3. Freeze model revision, engine, agent version, prompt/tool policy, task IDs, timeout, and cache state.
4. Run BALANCED under No PRA, PRA - No Adaptor, and PRA - Adaptor Bundle.
5. Compare official success first, then input tokens/successful task, cost/successful task, wall time/successful task, TTFT, decode rate, materialization avoidance, and memory.
6. Expand to QUALITY and ECONOMY only after BALANCED preserves task success.

## Generation and release order

1. Normalize raw artifacts with `CanonicalEvidenceRecord`.
2. Recompute signed and percentage deltas in code.
3. Regenerate every local bundle and model card.
4. Regenerate the site catalog, qualification matrix, and canonical evidence matrix.
5. Update Paper 4.5 and the relevant engine paper from the same normalized artifact.
6. Validate cards, strict-build documentation and papers, and run regression tests.
7. Publish immutable bundle revisions only after clean-pull checksum validation.

The source of truth is structured evidence, never a number copied from prose.
