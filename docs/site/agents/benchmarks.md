# Agent Benchmarks

The coding-agent campaign asks whether PRA lowers cumulative context-processing
cost while preserving official task success. End-task success is primary;
tokens, latency, resources, and cost are interpreted only beside that result.
External code or documentation RAG is outside this campaign.

## Staged campaign

1. **Stage A, compatibility:** 2-5 trivial tasks verify unattended execution,
   tool use, protocol behavior, isolation, and accounting. No quality claims.
2. **Stage B, pilot:** OpenCode, Pi, and OpenHands run 10-25 frozen tasks on
   representative engines under No PRA, Selected BALANCED, and genuine Native
   BALANCED where supported.
3. **Stage C, profiles:** stable pairs expand to QUALITY, BALANCED, and ECONOMY
   for Selected Context and Native Memory.
4. **Stage D/E, breadth:** commercial-native controls and larger official
   benchmark cohorts are added only after the pilot is stable.

Terminal-Bench 2.1 is primary. SWE-bench Lite is the development secondary and
SWE-bench Verified is reserved for promoted configurations. Their official
harnesses remain responsible for task setup and grading.

## Current Stage A evidence

The 2026-09-02 gateway cohort used the same two exact-output tasks twice for
each agent. Ollama/Qwen3-14B ran on an M5 Mac; the isolated agent workspaces and
gateway ran on Windows.

| Agent | Protocol | Runs | Success | Input tokens | Mean task time |
| --- | --- | ---: | ---: | ---: | ---: |
| Codex CLI 0.147.0 | Responses | 4 | 4/4 | 97,958 | 47.3 s |
| OpenCode 1.18.26 | Chat Completions | 4 | 4/4 | 53,170 | 107.7 s |
| Pi 0.73.1 | Chat Completions | 4 | 4/4 | 14,482 | 44.0 s |

These token totals include different agent system prompts and tool schemas, so
they are not a same-agent efficiency comparison. The fixture is intentionally
too small for latency conclusions. Raw and normalized evidence lives under
`docs/papers/shared/results/paper4_5_runtime_productization/coding_agents/`.

## Reproduce qualification

Audit installed agents and validate the deterministic fixture first:

```bash
python -m experiments.agents audit --output artifacts/agent-audit.json
python -m experiments.agents run \
  --manifest fixture_smoke.yaml \
  --output artifacts/fixture
```

Run an external agent only after pinning its executable and provider. The
runner supports repeated `--agent-arg` values for provider-specific flags and
never copies `--env` values into normalized artifacts.

Generate official-harness plans without pretending they have executed:

```bash
python -m experiments.agents plan \
  --manifest terminal_bench_pilot.yaml \
  --agent opencode --model MODEL --condition selected-balanced

python -m experiments.agents plan \
  --manifest swebench_lite_smoke.yaml \
  --agent pi --model MODEL --condition no-pra
```

## Required controls

Every compared condition uses identical task IDs and a fresh worktree. No PRA
preserves the agent's native compaction, cache, and tool policy. Native Memory
is `NOT_APPLICABLE` unless a real engine capability check succeeds. A selected
condition that saves tokens but solves fewer tasks is not better.
