# Prometheus

PRA can expose an owned, localhost-only `/metrics` endpoint independently from
OpenTelemetry:

```bash
pra gateway serve --backend vllm --backend-url http://localhost:8000 \
  --prometheus --prometheus-port 9464
```

The listener starts only when both global observability and Prometheus are
enabled. Configure a different path in YAML when required.

## Normalized metrics

| Domain | Examples |
| --- | --- |
| Agent | `pra_agent_turns_total`, `pra_agent_tool_calls_total`, turn/tool duration |
| Gateway | requests, duration, transport/message/resource/delta bytes, fallbacks, errors |
| Context | source, selected, newly materialized and visible-reuse tokens; route/select/realize time |
| Prefix/native | cached tokens, reuse status, native attaches, bytes and failures |
| Storage | HOT/WARM/COLD bytes, promotions, demotions, evictions, reloads, reconstructions |
| Engine | normalized request/TTFT/ITL/completion duration, successes and errors |

Useful derived quantities are:

```text
context reduction = 1 - selected_tokens / source_tokens
logical avoidance = visible_reuse_tokens / required_tokens
native reuse      = attached_native_resources / selected_resources
useful throughput = request_rate * success_probability
```

Task success is emitted only when a real outcome label exists.

## Engine-native scraping

Prometheus should scrape vLLM, SGLang, Triton/TensorRT-LLM, OVMS/OpenVINO, and
llama.cpp endpoints directly where their pinned versions expose metrics. PRA
adds normalized semantic context metrics but does not proxy-copy the complete
engine metric set. MLX, HF, AirLLM, Ollama, and FreeToken use wrapper metrics
for observations available through their APIs.

## Cardinality and security

Allowed labels are bounded deployment dimensions such as `engine`,
`model_family`, `profile`, `execution_mode`, `status`, `record_type`, and
`storage_tier`. Session, request, user, task, resource URI, prompt, and content
labels are rejected. Bind localhost by default and protect remote scrape paths
with network policy, authentication, and TLS.
