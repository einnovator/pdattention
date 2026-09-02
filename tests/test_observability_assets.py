from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]
DEPLOY = ROOT / "deploy" / "observability"


def test_observability_compose_is_profile_gated_and_valid() -> None:
    compose = yaml.safe_load((DEPLOY / "docker-compose.yml").read_text(encoding="utf-8"))
    assert set(compose["services"]) == {
        "otel-collector", "tempo", "prometheus", "grafana"
    }
    assert all("observability" in service["profiles"] for service in compose["services"].values())
    assert compose["services"]["prometheus"]["ports"] == [
        "${OBSERVABILITY_BIND_ADDRESS:-127.0.0.1}:9090:9090"
    ]
    for path in DEPLOY.rglob("*.yaml"):
        assert yaml.safe_load(path.read_text(encoding="utf-8")) is not None


def test_every_required_dashboard_is_valid_and_portable() -> None:
    required = {
        "pra-overview.json", "pra-agent.json", "pra-gateway.json",
        "pra-gateway-otel.json", "pra-runtime.json",
        "pra-vllm.json", "pra-sglang.json", "pra-mlx.json", "pra-openvino.json",
        "pra-tensorrt-llm.json", "pra-airllm.json", "pra-llamacpp.json",
        "pra-ollama.json", "pra-freetoken.json", "pra-hf.json",
    }
    directory = DEPLOY / "grafana" / "dashboards"
    actual = {path.name for path in directory.glob("*.json")}
    assert required <= actual
    assert {
        f"pra-{engine}-otel.json"
        for engine in (
            "vllm", "sglang", "mlx", "openvino", "tensorrt-llm",
            "airllm", "llamacpp", "ollama", "freetoken", "hf",
        )
    } <= actual
    variables = {
        "datasource", "environment", "service", "engine", "model", "host",
        "instance", "profile", "execution_mode",
    }
    for path in directory.glob("*.json"):
        dashboard = json.loads(path.read_text(encoding="utf-8"))
        assert dashboard["uid"].startswith("pra-")
        assert dashboard["panels"]
        if not path.stem.endswith("-otel"):
            assert {row["name"] for row in dashboard["templating"]["list"]} == variables
        assert "localhost" not in path.read_text(encoding="utf-8")


def test_tempo_and_remote_discovery_are_wired() -> None:
    collector = yaml.safe_load((DEPLOY / "otel-collector.yaml").read_text(encoding="utf-8"))
    assert "otlphttp/tempo" in collector["service"]["pipelines"]["traces"]["exporters"]
    datasources = yaml.safe_load(
        (DEPLOY / "grafana/provisioning/datasources/prometheus.yaml").read_text(encoding="utf-8")
    )["datasources"]
    assert {row["uid"] for row in datasources} == {"pra-prometheus", "pra-tempo"}
    prometheus = yaml.safe_load((DEPLOY / "prometheus.yaml").read_text(encoding="utf-8"))
    assert prometheus["scrape_configs"][0]["file_sd_configs"]


def test_engine_examples_cover_container_and_hybrid_deployments() -> None:
    engines = yaml.safe_load(
        (DEPLOY / "docker-compose.engines.yml").read_text(encoding="utf-8")
    )["services"]
    assert {"vllm", "sglang", "openvino-model-server", "tensorrt-llm", "llama-cpp", "ollama"} <= set(engines)
    readme = (DEPLOY / "README.md").read_text(encoding="utf-8")
    for engine in ("MLX", "AirLLM", "FreeToken", "HF"):
        assert engine in readme
