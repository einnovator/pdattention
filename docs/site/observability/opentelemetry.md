# OpenTelemetry

PRA uses W3C `traceparent` and `tracestate` propagation. When tracing is active,
an Agent turn can parent a Gateway request, context realization, engine call,
and engine-native spans. No headers are injected when tracing is off.

## Configuration

```yaml
observability:
  enabled: true
  otel:
    enabled: true
    endpoint: http://localhost:4317
    protocol: grpc
    traces: true
    sampler: parent_based_trace_id_ratio
    sample_rate: 0.01
  content:
    capture: none
```

Supported samplers are `always_off`, `parent_based`,
`parent_based_trace_id_ratio`, `trace_id_ratio`, and `always_on`. Use low-rate
parent-based sampling in production. Detailed/always-on tracing can affect
latency and must not replace telemetry-off qualification data.

CLI overrides are available on `pra serve`, `pra gateway serve`, and
`pra agent chat/run`:

```bash
pra agent chat --otel --otel-endpoint http://localhost:4317
```

`OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_SERVICE_NAME` are accepted below file
and CLI precedence.

## Span shape

```text
pra.agent.turn
└── pra.gateway.request
    ├── pra.gateway.session.resolve
    ├── pra.gateway.translate
    ├── pra.context.route / select / realize
    └── pra.engine.request
        └── engine-native spans, when explicitly enabled and supported
```

Service names remain distinct: `pra-agent`, `pra-gateway`, `pra-runtime`, and
the engine name. Useful attributes include token/record counts, realization
mode, prefix-reuse status, native attached bytes, storage activity, and engine
queue/prefill/decode time. Private content is omitted by default.

## Engine paths

- **Agent -> Gateway -> vLLM:** PRA propagates W3C context; compatible vLLM
  native tracing is enabled only by engine-native configuration.
- **Agent -> Gateway -> Ollama:** PRA wrapper spans cover mediation and the HTTP
  call; backend metrics come from the PRA-aware llama.cpp seam when available.
- **Embedded MLX/HF:** runtime wrapper spans remain in process and native timing
  APIs populate normalized attributes and metrics.

## Troubleshooting

No spans usually means one gate is off: global `enabled`, `otel.enabled`, or
`otel.traces`. Confirm the OTLP protocol/port pair (`4317` gRPC, `4318` HTTP),
then inspect collector logs. A working collector never turns tracing on by
itself.
