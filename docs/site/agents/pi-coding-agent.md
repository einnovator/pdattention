# Pi Coding Agent

Pi exposes extension events at the exact boundaries PRA needs: completed tool
execution, durable tool-result messages, and the final provider request. The
repository provides the tested `PiCodingAgentPRAAdapter` Python bridge. A small
Pi TypeScript extension is still required to connect those events to the bridge
or to emit the PRA envelope directly.

## What the PRA plugin does

The Pi bridge recognizes `tool_execution_end` and tool-result `message_end`
events. It converts full results into stable typed records, deduplicates by tool
call ID, preserves error and tool metadata, and carries a bounded resource set
to the model endpoint. It does not replace Pi's tools, approval UI, session
store, or provider selection.

## Set up the PRA plugin bridge

Install PRA and instantiate one adapter per Pi session:

```python
from pra_hf import PiCodingAgentPRAAdapter, PRAAgentPluginConfig

bridge = PiCodingAgentPRAAdapter(
    PRAAgentPluginConfig("Qwen/Qwen3-0.6B"),
    session_id="pi-session-42",
    task_id="inspect-repository",
)

bridge.ingest_event({
    "type": "tool_execution_end",
    "toolCallId": "call-7",
    "toolName": "read",
    "result": {"content": [{"type": "text", "text": "exact file content"}]},
    "isError": False,
})
```

For a Pi-native extension, place `pra.ts` in `.pi/extensions/` or load it with
`pi -e /path/to/pra.ts`. The extension should:

1. listen to `tool_execution_end` and `message_end`;
2. retain exact results under stable tool-call IDs;
3. add their typed resource descriptors under the request's `pra.resources`;
4. replace the provider payload in `before_provider_request` without changing
   Pi's ordinary messages or tool schemas;
5. remove session-bound state when the Pi session closes.

Pi's [extension documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/extensions.md)
defines these events and extension locations. The repository does not yet ship
the TypeScript file as an installable Pi package, and no PRA Pi package has been
published.

## Through the PRA gateway

Configure a Pi custom provider in `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "pra-gateway": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [{"id": "Qwen/Qwen3-0.6B"}]
    }
  }
}
```

Start the gateway, launch Pi, select `pra-gateway/Qwen/Qwen3-0.6B`, and load the
PRA extension:

```bash
pra gateway serve --mode selected-context --backend vllm \
  --backend-url http://127.0.0.1:8000/v1
pi -e .pi/extensions/pra.ts
```

Without the extension, Pi can still call the model through the gateway, but no
typed tool-result records are sent and the route provides no PRA context gain.

Pi 0.73.1 completed 4/4 Stage A exact-output tasks over this ordinary gateway
path with Ollama/Qwen3-14B. The run qualifies streaming tool execution and
agent-event accounting only; it is not a coding-quality or PRA-memory result.

In the official Terminal-Bench `query-optimize` gate, the same pairing made
three model calls, read the original SQL, wrote `sol.sql`, and passed 4/6
verifier tests with 5,219 cumulative input tokens. It received task reward
zero because the query changed one result and was slower than the benchmark
threshold. This qualifies Pi's Harbor tool loop but does not clear the Stage-B
quality gate.

## Direct PRA engine

Change only `baseUrl` to the direct engine, for example
`http://127.0.0.1:8000/v1`, and keep the PRA extension loaded. The engine must
accept the `pra` extension envelope. Configure the plugin to require Native
Memory when fallback would invalidate the experiment.

Pi's official [custom-model documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/models.md)
describes the provider file and compatibility switches. An ordinary direct
provider without the PRA envelope remains a normal Pi model connection.

## Integration check

Run one tool twice with the same call ID and verify the request contains one
resource. Then run a new call and verify only one new resource delta appears.
Finally, disable the extension and confirm the capability trace no longer
claims typed records or Native Memory.
