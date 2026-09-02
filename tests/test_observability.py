from __future__ import annotations

import sys
import socket
import threading
import urllib.error
import urllib.request

import pytest

from pra_hf.observability import (
    Observability,
    ObservabilityConfig,
    OTelConfig,
    PrometheusConfig,
    load_observability_config,
)
from pra_hf.deployment import PRAEngineCapabilities, PRAWireRequest
from pra_hf.engine_profiles import EngineType
from pra_hf.gateway import PRAGateway


def test_absent_observability_is_a_true_noop() -> None:
    before_threads = {thread.ident for thread in threading.enumerate()}
    observed = []
    telemetry = Observability()

    with telemetry.span("never-created", lambda: observed.append("evaluated") or {}):
        pass
    telemetry.increment("not-even-looked-up", labels={"request_id": "unsafe"})

    assert observed == []
    assert telemetry.snapshot()["enabled"] is False
    assert telemetry.registry is None
    assert {thread.ident for thread in threading.enumerate()} == before_threads
    assert telemetry.start_metrics_server() is None


def test_config_precedence_and_independent_metric_gate(tmp_path) -> None:
    path = tmp_path / "observability.yaml"
    path.write_text(
        """observability:\n  enabled: true\n  prometheus:\n    enabled: true\n    port: 9911\n  otel:\n    enabled: false\n""",
        encoding="utf-8",
    )
    config = load_observability_config(
        path,
        environment={
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://environment:4317",
            "OTEL_SERVICE_NAME": "environment-service",
            "OTEL_RESOURCE_ATTRIBUTES": "deployment.region=eu-west,team=pra",
        },
        overrides={"prometheus": {"port": 9912}},
        service="gateway",
    )

    assert config.metrics_enabled is True
    assert config.tracing_enabled is False
    assert config.prometheus.port == 9912
    assert config.otel.endpoint == "http://environment:4317"
    assert config.service.name == "pra-gateway"
    assert config.service.attributes == {
        "deployment.region": "eu-west",
        "team": "pra",
    }


def test_prometheus_endpoint_and_bounded_labels() -> None:
    telemetry = Observability(
        ObservabilityConfig(
            enabled=True,
            prometheus=PrometheusConfig(enabled=True, port=0x2525),
        )
    )
    telemetry.increment(
        "pra_gateway_requests_total",
        labels={"engine": "hf", "execution_mode": "G11", "status": "success"},
    )
    telemetry.increment("pra_gateway_resyncs_total")
    telemetry.increment("pra_gateway_visible_reuse_tokens_total", 12)
    telemetry.increment("pra_gateway_new_materialized_tokens_total", 4)
    telemetry.increment("pra_gateway_capability_negotiations_total")
    text = telemetry.render_metrics().decode("utf-8")
    assert "pra_gateway_requests_total" in text
    assert "pra_gateway_capability_negotiations_total" in text
    assert "pra_gateway_visible_reuse_tokens_total" in text
    assert 'engine="hf"' in text
    with pytest.raises(ValueError, match="High-cardinality"):
        telemetry.increment(
            "pra_gateway_requests_total",
            labels={
                "engine": "hf",
                "execution_mode": "G11",
                "status": "success",
                "session_id": "private",
            },
        )


def test_native_reuse_metric_is_exported_with_bounded_context_labels() -> None:
    telemetry = Observability(
        ObservabilityConfig(
            enabled=True,
            prometheus=PrometheusConfig(enabled=True),
        )
    )
    labels = {"engine": "hf", "profile": "balanced", "execution_mode": "E2"}
    telemetry.increment("pra_context_native_reuse_tokens_total", 48, labels=labels)

    text = telemetry.render_metrics().decode("utf-8")

    assert (
        'pra_context_native_reuse_tokens_total{engine="hf",execution_mode="E2",profile="balanced"} 48.0'
        in text
    )


def test_gateway_metrics_treat_absent_transport_sizes_as_zero() -> None:
    class Storage:
        @staticmethod
        def usage():
            return {"hot_bytes": 128, "warm_bytes": 256, "cold_bytes": 512}

    class Adapter:
        storage = Storage()

        def capabilities(self) -> PRAEngineCapabilities:
            return PRAEngineCapabilities(
                adapter="controlled",
                engine_type=EngineType.OPENAI_GENERIC,
                integration_level="E0",
            )

    telemetry = Observability(
        ObservabilityConfig(
            enabled=True,
            prometheus=PrometheusConfig(enabled=True),
        )
    )
    gateway = PRAGateway(Adapter(), mode="G10", observability=telemetry)
    request = PRAWireRequest.from_dict(
        {"model": "controlled", "messages": [{"role": "user", "content": "test"}]}
    )

    gateway._record_metrics(  # noqa: SLF001 - optional engine diagnostics
        request,
        {
            "message_bytes_sent": None,
            "resource_bytes_sent": None,
            "session_delta_bytes": None,
            "prefix_cached_tokens": None,
        },
        0.01,
        status="success",
    )

    metrics = telemetry.render_metrics().decode("utf-8")
    assert 'pra_gateway_transport_bytes_total{engine="openai_generic"} 0.0' in metrics
    assert 'pra_native_bytes{engine="openai_generic",storage_tier="hot"} 128.0' in metrics
    assert 'pra_native_bytes{engine="openai_generic",storage_tier="warm"} 256.0' in metrics
    assert 'pra_native_bytes{engine="openai_generic",storage_tier="cold"} 512.0' in metrics


def test_prometheus_listener_is_explicit_and_owned(monkeypatch) -> None:
    def reject_reverse_dns(host: str) -> str:
        raise AssertionError(f"unexpected reverse DNS lookup for {host}")

    monkeypatch.setattr(socket, "getfqdn", reject_reverse_dns)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    telemetry = Observability(
        ObservabilityConfig(
            enabled=True,
            prometheus=PrometheusConfig(enabled=True, port=port),
        )
    )
    try:
        address = telemetry.start_metrics_server()
        assert address is not None
        payload = urllib.request.urlopen(
            f"http://127.0.0.1:{address[1]}/metrics", timeout=2
        ).read()
        assert b"pra_engine_requests_total" in payload
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(f"http://127.0.0.1:{address[1]}/wrong", timeout=2)
        assert error.value.code == 404
    finally:
        telemetry.close()


def test_otel_parent_child_w3c_and_privacy() -> None:
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    telemetry = Observability(
        ObservabilityConfig(
            enabled=True,
            otel=OTelConfig(enabled=True, sampler="always_on", sample_rate=1.0),
        ),
        span_exporter=exporter,
    )
    try:
        with telemetry.span(
            "pra.agent.turn",
            {"prompt": "secret", "pra.context.selected_tokens": 7},
        ) as root:
            carrier = telemetry.inject({})
            with telemetry.span(
                "pra.gateway.request",
                {"document.content": "secret", "pra.engine": "hf"},
                parent_headers=carrier,
            ):
                pass
        spans = {span.name: span for span in exporter.get_finished_spans()}
        assert spans["pra.gateway.request"].parent.span_id == root.get_span_context().span_id
        assert spans["pra.gateway.request"].context.trace_id == root.get_span_context().trace_id
        assert "traceparent" in carrier
        assert "prompt" not in spans["pra.agent.turn"].attributes
        assert "document.content" not in spans["pra.gateway.request"].attributes
        assert spans["pra.agent.turn"].attributes["pra.context.selected_tokens"] == 7
    finally:
        telemetry.close()


def test_disabled_import_path_does_not_load_telemetry_sdk() -> None:
    # The package may already be loaded by another test, so use the source-level
    # invariant that constructing the disabled object never adds SDK modules.
    before = set(sys.modules)
    Observability()
    after = set(sys.modules)
    assert not any(name.startswith("opentelemetry.sdk") for name in after - before)
