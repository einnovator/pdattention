# DeepSeek Harness

DeepSeek Harness can keep ownership of its event-sourced session and tool loop
while a PRA bridge turns durable results into typed resources. The repository
provides the tested `DeepSeekHarnessPRAAdapter`; it is a Python SDK bridge, not a
published DeepSeek Harness bundle.

## What the PRA plugin does

The bridge recognizes normalized `tool/result`, `session/reference`, and
`attachment` events. It:

- assigns a stable `pra://agent/deepseek_harness/...` URI;
- preserves event identity, tool name, task, tenant, and session provenance;
- deduplicates replayed events by stable resource ID;
- applies resource-count and selected-token budgets;
- builds a tensor-free PRA request for a gateway or direct capable engine;
- requires an explicit error when Native Memory is mandatory but unavailable.

It does not execute tools, rewrite Harness authorization, or move model-native
K/V through the Harness process.

## Set up the PRA plugin bridge

Install PRA in the Python environment used by the bridge process:

```bash
python -m pip install -e .
```

Create one adapter per Harness session:

```python
from pra_hf import DeepSeekHarnessPRAAdapter, PRAAgentPluginConfig

bridge = DeepSeekHarnessPRAAdapter(
    PRAAgentPluginConfig(
        model="Qwen/Qwen3-0.6B",
        tenant_id="team-a",
        max_resources=8,
        max_selected_tokens=2048,
        allow_text_fallback=True,
    ),
    session_id="dsh-session-42",
    task_id="repair-build",
)

bridge.ingest_event({
    "type": "tool/result",
    "id": "tool-call-17",
    "toolName": "read_file",
    "result": {"content": [{"type": "text", "text": "exact tool output"}]},
})
```

A Harness host plugin should listen to the durable `session/event` stream,
normalize completed `tool/result` records to this shape, and hand them to the
Python bridge over the host's chosen RPC boundary. Submit the resulting request
with `bridge.request(messages).to_openai()`.

DeepSeek Harness plugins are installed into a profile with its plugin manager.
Once the host glue is packaged locally, install that bundle with:

```bash
dsh plugin --profile pra add file:../deepseek-pra-plugin
dsh --profile pra
```

There is no published `deepseek-pra-plugin` package in this repository yet; the
command shows where the host bundle belongs. See the [DeepSeek Harness plugin
reference](https://github.com/deepseek-ai/deepseek-harness/blob/master/apps/cli/reference/README.md)
for bundle packaging and profile installation.

## Through the PRA gateway

For an ordinary backend, use explicit Selected Context fallback:

```bash
pra gateway serve \
  --mode selected-context \
  --backend vllm \
  --backend-url http://127.0.0.1:8000/v1 \
  --sessions-dir .pra/gateway-sessions
```

Post `bridge.request(messages).to_openai()` to
`http://127.0.0.1:8080/v1/chat/completions`. The gateway budgets and renders the
typed resources, records the downgrade, and sends ordinary labeled context to
the backend.

Use `typed-transport` only when the immediate backend advertises typed records:

```bash
pra gateway serve \
  --mode typed-transport \
  --backend sglang \
  --backend-url http://127.0.0.1:30000
```

## Direct PRA engine

Post the same OpenAI payload directly to a PRA-aware engine endpoint. Set
`require_native_pra=True` and `allow_text_fallback=False` when the workload must
use Native Memory:

```python
config = PRAAgentPluginConfig(
    model="Qwen/Qwen3-0.6B",
    require_native_pra=True,
    allow_text_fallback=False,
)
```

The direct endpoint must parse the `pra` request extension and advertise the
required capabilities. Pointing Harness at an ordinary OpenAI endpoint skips
the typed bridge and is not direct PRA integration.

## Integration check

Verify a replayed tool result produces one resource, that two tenants cannot
share a session-bound record, and that an ordinary fallback response reports
`native_kv: false`. These are the same contracts covered by the repository's
DeepSeek bridge tests.
