"""Generate the checked-in baseline Grafana dashboards deterministically."""

from __future__ import annotations

import json
from pathlib import Path


ENGINES = (
    "vllm", "sglang", "mlx", "openvino", "tensorrt-llm", "airllm",
    "llamacpp", "ollama", "freetoken", "hf",
)
VARIABLES = (
    "environment", "service", "engine", "model", "host", "instance", "profile",
    "execution_mode",
)


def variable(name: str) -> dict:
    metric_label = {"model": "model_family"}.get(name, name)
    return {
        "name": name,
        "label": name.replace("_", " ").title(),
        "type": "query",
        "datasource": {"type": "prometheus", "uid": "${datasource}"},
        "query": {"query": f"label_values({metric_label})", "refId": "variable"},
        "includeAll": True,
        "multi": True,
        "refresh": 1,
        "current": {"text": "All", "value": "$__all"},
    }


def panel(panel_id: int, title: str, expression: str, *, unit: str = "short") -> dict:
    return {
        "id": panel_id,
        "title": title,
        "type": "timeseries",
        "datasource": {"type": "prometheus", "uid": "${datasource}"},
        "gridPos": {
            "h": 8, "w": 12, "x": 0 if panel_id % 2 else 12,
            "y": ((panel_id - 1) // 2) * 8,
        },
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "options": {"legend": {"displayMode": "table", "placement": "bottom"}},
        "targets": [
            {"refId": "A", "expr": expression, "legendFormat": "{{engine}} {{execution_mode}}"}
        ],
    }


def dashboard(slug: str, title: str, panels: list[dict], *, engine: str | None = None) -> dict:
    variables = [
        {
            "name": "datasource", "label": "Datasource", "type": "datasource",
            "query": "prometheus",
            "current": {"text": "Prometheus", "value": "pra-prometheus"},
        },
        *(variable(name) for name in VARIABLES),
    ]
    return {
        "id": None,
        "uid": f"pra-{slug}",
        "title": title,
        "tags": ["pra", engine or "multi-engine"],
        "schemaVersion": 41,
        "version": 1,
        "refresh": "10s",
        "time": {"from": "now-1h", "to": "now"},
        "templating": {"list": variables},
        "annotations": {"list": []},
        "panels": panels,
    }


COMMON = [
    panel(1, "Selected / source token ratio", "sum(rate(pra_context_selected_tokens_total{engine=~\"$engine\"}[5m])) by (engine,execution_mode) / clamp_min(sum(rate(pra_context_source_tokens_total{engine=~\"$engine\"}[5m])) by (engine,execution_mode), 1)", unit="percentunit"),
    panel(2, "Visible and native reuse", "sum(rate(pra_context_visible_reuse_tokens_total{engine=~\"$engine\"}[5m])) by (engine,execution_mode)"),
    panel(3, "Gateway p95 latency", "histogram_quantile(0.95, sum(rate(pra_gateway_request_duration_seconds_bucket{engine=~\"$engine\"}[5m])) by (le,engine,execution_mode))", unit="s"),
    panel(4, "Successful request throughput", "sum(rate(pra_engine_successful_requests_total{engine=~\"$engine\"}[5m])) by (engine,execution_mode)", unit="reqps"),
    panel(5, "Native active bytes", "sum(pra_native_bytes{engine=~\"$engine\"}) by (engine,storage_tier)", unit="bytes"),
    panel(6, "Storage HOT bytes", "sum(pra_storage_hot_bytes{engine=~\"$engine\"}) by (engine)", unit="bytes"),
]


def main() -> None:
    target = Path(__file__).parent / "grafana" / "dashboards"
    target.mkdir(parents=True, exist_ok=True)
    values = {
        "pra-overview.json": dashboard("overview", "PRA Multi-engine Overview", COMMON),
        "pra-agent.json": dashboard("agent", "PRA Agent", [
            panel(1, "Agent turns", "sum(rate(pra_agent_turns_total[5m])) by (status)"),
            panel(2, "Agent p95 turn latency", "histogram_quantile(0.95, sum(rate(pra_agent_turn_duration_seconds_bucket[5m])) by (le,status))", unit="s"),
            panel(3, "Tool calls", "sum(rate(pra_agent_tool_calls_total[5m])) by (status)"),
            panel(4, "Active sessions", "sum(pra_agent_active_sessions)"),
        ]),
        "pra-gateway.json": dashboard("gateway", "PRA Gateway", [
            panel(1, "Request rate", "sum(rate(pra_gateway_requests_total[5m])) by (engine,execution_mode,status)"),
            panel(2, "p95 request latency", "histogram_quantile(0.95, sum(rate(pra_gateway_request_duration_seconds_bucket[5m])) by (le,engine,execution_mode))", unit="s"),
            panel(3, "Transport bytes", "sum(rate(pra_gateway_transport_bytes_total[5m])) by (engine)", unit="Bps"),
            panel(4, "Upstream errors", "sum(rate(pra_gateway_upstream_errors_total[5m])) by (engine)"),
        ]),
        "pra-runtime.json": dashboard("runtime", "PRA Runtime: Why PRA Helped", COMMON),
    }
    for engine in ENGINES:
        selector = engine.replace("-", "[_-]")
        values[f"pra-{engine}.json"] = dashboard(engine, f"PRA + {engine}", [
            panel(1, "Normalized request p95", f"histogram_quantile(0.95, sum(rate(pra_engine_request_duration_seconds_bucket{{engine=~\"{selector}\"}}[5m])) by (le,engine,execution_mode))", unit="s"),
            panel(2, "Context economy", f"sum(rate(pra_context_selected_tokens_total{{engine=~\"{selector}\"}}[5m])) / clamp_min(sum(rate(pra_context_source_tokens_total{{engine=~\"{selector}\"}}[5m])), 1)", unit="percentunit"),
            panel(3, "Native attaches", f"sum(rate(pra_native_attaches_total{{engine=~\"{selector}\"}}[5m])) by (status)"),
            panel(4, "Native bytes", f"sum(pra_native_bytes{{engine=~\"{selector}\"}}) by (storage_tier)", unit="bytes"),
        ], engine=engine)
    for name, value in values.items():
        (target / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
