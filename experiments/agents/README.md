# PRA Coding-Agent Evaluation

This package runs paired coding-agent experiments while keeping official task
success primary and context economy secondary. It is deliberately separate from
external code/document RAG.

```bash
python -m experiments.agents audit --output artifacts/agent-audit.json
python -m experiments.agents plan --manifest terminal_bench_pilot.yaml \
  --agent opencode --model Qwen/Qwen3-Coder --condition selected-balanced
python -m experiments.agents run --manifest fixture_smoke.yaml \
  --output artifacts/fixture/no-pra
python -m experiments.agents analyze artifacts/fixture/no-pra/runs.jsonl
```

An external CLI can receive provider-specific argv without changing its audited
base command. For example, Codex can use a PRA Gateway Responses endpoint with
repeated `--agent-arg` values. Pass leading dashes with `=` so `argparse` does
not interpret them as benchmark-runner options:

```bash
python -m experiments.agents smoke-agent --agent codex \
  --model qwen3-14b-pra --engine ollama --connection gateway \
  --protocol openai-responses --output artifacts/codex-gateway \
  --env OPENAI_API_KEY=local-benchmark \
  --agent-arg=-c --agent-arg='model_provider="pra_gateway"' \
  --agent-arg=-c --agent-arg='model_providers.pra_gateway.name="PRA Gateway"' \
  --agent-arg=-c --agent-arg='model_providers.pra_gateway.base_url="http://127.0.0.1:18081/v1"' \
  --agent-arg=-c --agent-arg='model_providers.pra_gateway.env_key="OPENAI_API_KEY"' \
  --agent-arg=-c --agent-arg='model_providers.pra_gateway.wire_api="responses"'
```

The fixture cohort tests isolation, normalization, artifact writing, and
analysis only. It is never coding-agent quality evidence. Terminal-Bench runs
must use Harbor; SWE-bench grading must use the official container harness.
Frozen manifests prevent task drift between PRA conditions.

On benchmark hosts where Harbor's default OpenCode `nvm` bootstrap cannot
clone public GitHub repositories, use the repository adapter below. It changes
only installation: Node 22 archives are checksum-pinned, while OpenCode
execution, trajectory capture, task isolation, and grading remain Harbor's.

```bash
harbor run -d terminal-bench/terminal-bench-2-1 \
  -a experiments.agents.harbor_agents:PinnedNodeOpenCode \
  -m openai/qwen3:14b \
  -i terminal-bench/filter-js-from-html \
  --agent-env OPENAI_API_KEY=pra-local \
  --agent-env OPENAI_BASE_URL=http://GATEWAY_HOST:18100/v1 \
  --allow-agent-host GATEWAY_HOST -n 1 -y
```

## Conditions

Stable combinations progress from No PRA to Selected Context and then Native
Memory. QUALITY, BALANCED, and ECONOMY are crossed only after the BALANCED pilot
is stable. Native Memory is `NOT_APPLICABLE` unless an engine capability check
and a real native path both succeed.

Every task receives a clean worktree/container, reset agent and PRA sessions,
declared cache state, and a unique run ID. Credentials remain in process
environment or engine configuration and are never written to normalized JSONL.
