# Gateway Deployment

The gateway lets an OpenAI-compatible application use typed PRA context without
changing the model endpoint protocol. It negotiates capabilities, preserves
resource identity, and owns explicit fallback.

## Selected Context gateway

Use this with an ordinary engine:

```bash
pra gateway serve \
  --mode selected-context \
  --backend vllm \
  --backend-url http://127.0.0.1:8000/v1
```

The agent can send typed resources to the gateway. The gateway selects and
renders authorized resources as labeled text before calling the downstream
engine.

## Typed transport gateway

Use this only when the downstream endpoint advertises typed resources:

```bash
pra gateway serve \
  --mode typed-transport \
  --backend sglang \
  --backend-url http://127.0.0.1:30000
```

The gateway preserves full or delta resource inventories, task/session metadata,
and policy. Native K/V still remains inside the engine process.

## Capability and health endpoints

- `GET /health`
- `GET /v1/pra/capabilities`
- `POST /v1/pra/generate`
- `POST /v1/chat/completions`

A reachable endpoint that lacks PRA capabilities can be treated as ordinary. An
unreachable endpoint is an error. Capability loss after reconnect invalidates
cached transport state and forces a full resynchronization.

## Security boundary

The gateway rejects cross-tenant resources and credential-like PRA metadata.
Credentials remain in transport headers or provider configuration. Relevance,
cache residency, and resource sharing never grant authorization.

See [Protocol](../protocol.md) for the JSON contract.
