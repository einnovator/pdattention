# PRA gateway dashboard end-to-end check

This artifact records a controlled end-to-end gateway run on the shared Mac
observability host. The test launched an OpenAI-compatible backend and the real
PRA gateway, sent 12 successful completion requests through the gateway, and
then queried the same Prometheus and Tempo data sources used by Grafana.

## Result

- Status: `PASS`
- Tested commit: `e950967d67809abc371e241791791214a4494a07`
- Requests: `12/12`
- Gateway mode: `G10`
- Backend: `openai_generic` (`E0`)
- Positive Prometheus gateway rate: `0.04348954385964912` requests/s
- Grafana Prometheus result: `1` series with `19` positive points
- Tempo result: `16` matching `pra.gateway.request` traces
- Trace structure: `4` spans per request in service `pra-gateway`

The machine-readable evidence, including health capabilities, PromQL results,
TraceQL results, and Grafana data-source proxy checks, is in
`gateway_dashboard_e2e.json`.

## Dashboards

- Gateway metrics: `http://192.168.1.102:3000/d/pra-gateway/pra-gateway`
- Gateway traces: `http://192.168.1.102:3000/d/pra-gateway-otel/pra-2b-gateway3a-otel-traces`

Use a recent time range such as **Last 1 hour** when inspecting this run. The
test gateway exits after verification, so its scrape target is intentionally
down between test runs while the recorded time series remains queryable.

## Reproduction

```bash
python deploy/observability/run_gateway_dashboard_e2e.py \
  --metrics-port 9466 \
  --requests 12 \
  --otlp-endpoint http://127.0.0.1:4317 \
  --prometheus-url http://127.0.0.1:9090 \
  --tempo-url http://127.0.0.1:3200 \
  --grafana-url http://127.0.0.1:3000 \
  --output gateway_dashboard_e2e.json
```
