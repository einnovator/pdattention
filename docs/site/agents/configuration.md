# Agent Configuration

PRA Agent accepts YAML, JSON, and TOML. Configuration precedence is deterministic:

```text
explicit constructor or CLI override
programmatic settings
configuration file
environment
defaults
```

Secrets are references, not inline values. `/config show` redacts custom header
values, and no command resolves a referenced token for display.

```yaml
agent:
  model: Qwen/Qwen3-4B
  provider: local
  user_id: researcher
  confirmations:
    high_impact_tools: true
    unknown_mcp_mutations: true

providers:
  local:
    type: openai
    base_url: http://127.0.0.1:8080
    model: Qwen/Qwen3-4B
    engine_instance: local-vllm
    runtime_model_id: qwen3-4b

mcp:
  servers:
    docs:
      transport: http
      url: http://127.0.0.1:9400/mcp
      auth:
        type: bearer
        token_env: PRA_MCP_TOKEN
      tool_allow: ["search_*", "read_*"]
      tool_deny: ["delete_*"]

control_plane:
  enabled: true
  url: http://127.0.0.1:9300
  auth:
    type: bearer
    token_env: PRA_CONTROL_TOKEN

tui:
  history_size: 1000
  history_file: ~/.local/share/pra/agent/history
  paste:
    block_threshold_chars: 2000
    block_threshold_lines: 20

session:
  path: .pra/sessions
  resume_last: true
```

SDK loading is explicit:

```python
from pra_hf import PRAAgent, PRAAgentSettings

settings = PRAAgentSettings.compose(
    config_file="pra-agent.yaml",
    config={"agent": {"context_records": 16}},
    overrides={"agent": {"model": "Qwen/Qwen3-4B"}},
)
agent = PRAAgent.from_config_file("pra-agent.yaml", config=settings)
```

Runtime edits made with `/config set` remain in memory. `/config save` or `pra
agent mcp ... --save` is required to modify a file.

## Typed schema

Stable sections use Pydantic models: `AgentRuntimeSettings`, `ProviderConfig`,
`MCPAgentConfig`, `MCPServerConfig`, `ControlPlaneClientConfig`, `TUIConfig`, and
`AgentSessionConfig`. Unknown keys fail validation instead of being ignored.
