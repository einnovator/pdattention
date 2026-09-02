"""Default-off OpenTelemetry and Prometheus instrumentation for PRA.

The module deliberately imports telemetry libraries only after an explicit
enablement decision.  Callers may therefore keep an :class:`Observability`
instance on hot request paths without paying for attribute construction when
telemetry is disabled: span attributes are supplied as callables and are never
evaluated by the no-op path.
"""

from __future__ import annotations

import hashlib
import os
import socket
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, MutableMapping

import yaml

from .http_server import PRAThreadingHTTPServer


_SERVICE_NAMES = {
    "agent": "pra-agent",
    "gateway": "pra-gateway",
    "runtime": "pra-runtime",
}
_CONTENT_KEYS = {"prompt", "content", "document", "tool.arguments", "tool.result"}
_FORBIDDEN_LABELS = {
    "session_id",
    "request_id",
    "resource_uri",
    "task_id",
    "user_id",
    "prompt",
    "content",
}


@dataclass(frozen=True)
class ServiceConfig:
    name: str = "pra-runtime"
    environment: str = "development"
    version: str = "0.2.0rc1"
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "attributes", {str(key): str(value) for key, value in self.attributes.items()}
        )


@dataclass(frozen=True)
class PrometheusConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 9464
    path: str = "/metrics"

    def __post_init__(self) -> None:
        if not 0 < self.port < 65536:
            raise ValueError("Prometheus port must be between 1 and 65535.")
        if not self.path.startswith("/"):
            raise ValueError("Prometheus path must begin with '/'.")


@dataclass(frozen=True)
class OTelConfig:
    enabled: bool = False
    endpoint: str | None = None
    protocol: str = "grpc"
    traces: bool = True
    metrics: bool = False
    sampler: str = "parent_based_trace_id_ratio"
    sample_rate: float = 0.01

    def __post_init__(self) -> None:
        if self.protocol not in {"grpc", "http/protobuf"}:
            raise ValueError("OTel protocol must be 'grpc' or 'http/protobuf'.")
        if self.sampler not in {
            "always_off",
            "parent_based",
            "parent_based_trace_id_ratio",
            "trace_id_ratio",
            "always_on",
        }:
            raise ValueError(f"Unsupported OTel sampler: {self.sampler}")
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError("OTel sample_rate must be between 0 and 1.")


@dataclass(frozen=True)
class EngineNativeConfig:
    enable_tracing: bool = False
    enable_metrics: bool = False


@dataclass(frozen=True)
class ContentConfig:
    capture: str = "none"

    def __post_init__(self) -> None:
        if self.capture not in {"none", "metadata", "sampled-content", "full-content"}:
            raise ValueError(f"Unsupported observability content policy: {self.capture}")


@dataclass(frozen=True)
class ObservabilityConfig:
    """Shared telemetry policy; absence of configuration resolves to off."""

    enabled: bool = False
    service: ServiceConfig = field(default_factory=ServiceConfig)
    prometheus: PrometheusConfig = field(default_factory=PrometheusConfig)
    otel: OTelConfig = field(default_factory=OTelConfig)
    engine_native: EngineNativeConfig = field(default_factory=EngineNativeConfig)
    content: ContentConfig = field(default_factory=ContentConfig)
    grafana_url: str | None = None
    trace_url: str | None = None

    @property
    def metrics_enabled(self) -> bool:
        return self.enabled and self.prometheus.enabled

    @property
    def tracing_enabled(self) -> bool:
        return self.enabled and self.otel.enabled and self.otel.traces

    def for_component(self, component: str) -> "ObservabilityConfig":
        return replace(
            self,
            service=replace(self.service, name=_SERVICE_NAMES.get(component, component)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "ObservabilityConfig":
        raw = dict(value or {})
        service = dict(raw.get("service") or {})
        prometheus = dict(raw.get("prometheus") or {})
        otel = dict(raw.get("otel") or {})
        engine_native = dict(raw.get("engine_native") or {})
        content = dict(raw.get("content") or {})
        return cls(
            enabled=bool(raw.get("enabled", False)),
            service=ServiceConfig(**service),
            prometheus=PrometheusConfig(**prometheus),
            otel=OTelConfig(**otel),
            engine_native=EngineNativeConfig(**engine_native),
            content=ContentConfig(**content),
            grafana_url=raw.get("grafana_url"),
            trace_url=raw.get("trace_url"),
        )


def load_observability_config(
    path: str | Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
    service: str | None = None,
) -> ObservabilityConfig:
    """Resolve defaults, conventional OTel environment, file, then CLI overrides."""

    env = dict(os.environ if environment is None else environment)
    value: dict[str, Any] = {}
    endpoint = env.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    service_name = env.get("OTEL_SERVICE_NAME")
    resource_attributes = _parse_resource_attributes(
        env.get("OTEL_RESOURCE_ATTRIBUTES", "")
    )
    if endpoint or service_name or resource_attributes:
        value = {
            "service": {
                "name": service_name or "pra-runtime",
                "attributes": resource_attributes,
            },
            "otel": {"endpoint": endpoint},
        }
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, Mapping):
            raise ValueError("Observability configuration root must be a mapping.")
        section = loaded.get("observability", loaded)
        if not isinstance(section, Mapping):
            raise ValueError("The observability section must be a mapping.")
        value = _deep_merge(value, section)
    value = _deep_merge(value, overrides or {})
    config = ObservabilityConfig.from_dict(value)
    return config.for_component(service) if service else config


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, item in override.items():
        if isinstance(item, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], item)
        else:
            result[key] = item
    return result


def _parse_resource_attributes(value: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for item in value.split(","):
        if "=" not in item:
            continue
        key, content = item.split("=", 1)
        if key.strip():
            attributes[key.strip()] = content.strip()
    return attributes


_METRIC_SPECS: dict[str, tuple[str, tuple[str, ...], str]] = {
    "pra_agent_turns_total": ("counter", ("status",), "Agent turns."),
    "pra_agent_turn_duration_seconds": ("histogram", ("status",), "Agent turn latency."),
    "pra_agent_tool_calls_total": ("counter", ("status",), "Tool calls."),
    "pra_agent_tool_call_duration_seconds": ("histogram", ("status",), "Tool call latency."),
    "pra_agent_context_prepare_duration_seconds": ("histogram", (), "Context preparation latency."),
    "pra_agent_active_sessions": ("gauge", (), "Active agent sessions."),
    "pra_agent_active_tasks": ("gauge", ("status",), "Active agent tasks."),
    "pra_gateway_requests_total": ("counter", ("engine", "execution_mode", "status"), "Gateway requests."),
    "pra_gateway_request_duration_seconds": ("histogram", ("engine", "execution_mode", "status"), "Gateway request latency."),
    "pra_gateway_active_sessions": ("gauge", ("engine",), "Active gateway sessions."),
    "pra_gateway_transport_bytes_total": ("counter", ("engine",), "Gateway transport bytes."),
    "pra_gateway_message_bytes_total": ("counter", ("engine",), "Gateway message bytes."),
    "pra_gateway_resource_bytes_total": ("counter", ("engine",), "Gateway resource bytes."),
    "pra_gateway_delta_bytes_total": ("counter", ("engine",), "Gateway session delta bytes."),
    "pra_gateway_fallbacks_total": ("counter", ("engine", "status"), "Gateway fallbacks."),
    "pra_gateway_upstream_errors_total": ("counter", ("engine",), "Gateway upstream errors."),
    "pra_context_source_tokens_total": ("counter", ("engine", "profile", "execution_mode"), "Source tokens."),
    "pra_context_selected_tokens_total": ("counter", ("engine", "profile", "execution_mode"), "Selected tokens."),
    "pra_context_new_materialized_tokens_total": ("counter", ("engine", "profile", "execution_mode"), "Newly materialized tokens."),
    "pra_context_visible_reuse_tokens_total": ("counter", ("engine", "profile", "execution_mode"), "Visible reuse tokens."),
    "pra_context_selected_records_total": ("counter", ("engine", "profile", "execution_mode", "record_type"), "Selected records."),
    "pra_context_already_visible_records_total": ("counter", ("engine", "profile", "execution_mode", "record_type"), "Already visible records."),
    "pra_routing_duration_seconds": ("histogram", ("engine", "profile"), "Routing latency."),
    "pra_selection_duration_seconds": ("histogram", ("engine", "profile"), "Selection latency."),
    "pra_realization_duration_seconds": ("histogram", ("engine", "profile", "execution_mode"), "Realization latency."),
    "pra_serialization_duration_seconds": ("histogram", ("engine",), "Serialization latency."),
    "pra_prefix_cached_tokens_total": ("counter", ("engine",), "Prefix-cached tokens."),
    "pra_prefix_observations_total": ("counter", ("engine", "status"), "Prefix reuse observations."),
    "pra_native_attaches_total": ("counter", ("engine", "status"), "Native memory attachments."),
    "pra_native_attach_duration_seconds": ("histogram", ("engine",), "Native attach latency."),
    "pra_native_bytes": ("gauge", ("engine", "storage_tier"), "Active native bytes."),
    "pra_native_hot_resources": ("gauge", ("engine",), "Hot native resources."),
    "pra_native_warm_resources": ("gauge", ("engine",), "Warm native resources."),
    "pra_native_attach_failures_total": ("counter", ("engine",), "Native attach failures."),
    "pra_storage_hot_bytes": ("gauge", ("engine",), "HOT tier bytes."),
    "pra_storage_warm_bytes": ("gauge", ("engine",), "WARM tier bytes."),
    "pra_storage_cold_bytes": ("gauge", ("engine",), "COLD tier bytes."),
    "pra_storage_promotions_total": ("counter", ("engine", "storage_tier"), "Storage promotions."),
    "pra_storage_promotion_duration_seconds": ("histogram", ("engine", "storage_tier"), "Storage promotion latency."),
    "pra_storage_demotions_total": ("counter", ("engine", "storage_tier"), "Storage demotions."),
    "pra_storage_evictions_total": ("counter", ("engine", "storage_tier"), "Storage evictions."),
    "pra_storage_reloads_total": ("counter", ("engine", "storage_tier"), "Storage reloads."),
    "pra_storage_reconstructions_total": ("counter", ("engine",), "Storage reconstructions."),
    "pra_storage_source_reads_total": ("counter", ("engine",), "Storage source reads."),
    "pra_engine_requests_total": ("counter", ("engine", "model_family", "profile", "execution_mode", "status"), "Normalized engine requests."),
    "pra_engine_request_duration_seconds": ("histogram", ("engine", "model_family", "profile", "execution_mode", "status"), "Engine request latency."),
    "pra_engine_ttft_seconds": ("histogram", ("engine", "model_family", "profile", "execution_mode"), "Time to first token."),
    "pra_engine_itl_seconds": ("histogram", ("engine", "model_family", "profile", "execution_mode"), "Inter-token latency."),
    "pra_engine_completion_duration_seconds": ("histogram", ("engine", "model_family", "profile", "execution_mode"), "Completion latency."),
    "pra_engine_successful_requests_total": ("counter", ("engine", "model_family", "profile", "execution_mode"), "Successful requests."),
    "pra_engine_errors_total": ("counter", ("engine", "model_family", "profile", "execution_mode"), "Engine errors."),
}


class Observability:
    """Own optional telemetry backends and provide a cheap disabled path."""

    def __init__(
        self,
        config: ObservabilityConfig | Mapping[str, Any] | None = None,
        *,
        span_exporter: Any | None = None,
        start_server: bool = False,
    ) -> None:
        self.config = (
            config if isinstance(config, ObservabilityConfig)
            else ObservabilityConfig.from_dict(config)
        )
        self._tracer = None
        self._provider = None
        self._registry = None
        self._metrics: dict[str, Any] = {}
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        if self.config.tracing_enabled:
            self._initialize_otel(span_exporter)
        if self.config.metrics_enabled:
            self._initialize_metrics()
            if start_server:
                self.start_metrics_server()

    @property
    def enabled(self) -> bool:
        return self.config.tracing_enabled or self.config.metrics_enabled

    @property
    def tracing_enabled(self) -> bool:
        return self.config.tracing_enabled

    @property
    def metrics_enabled(self) -> bool:
        return self.config.metrics_enabled

    @property
    def registry(self) -> Any | None:
        return self._registry

    def _initialize_otel(self, span_exporter: Any | None) -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.sampling import (
                ALWAYS_OFF,
                ALWAYS_ON,
                ParentBased,
                TraceIdRatioBased,
            )
        except ImportError as error:
            raise RuntimeError("Install the 'observability' extra to enable OTel.") from error
        sampling = self.config.otel.sampler
        ratio = TraceIdRatioBased(self.config.otel.sample_rate)
        samplers = {
            "always_off": ALWAYS_OFF,
            "always_on": ALWAYS_ON,
            "trace_id_ratio": ratio,
            "parent_based": ParentBased(ALWAYS_ON),
            "parent_based_trace_id_ratio": ParentBased(ratio),
        }
        resource = Resource.create(
            {
                "service.name": self.config.service.name,
                "service.version": self.config.service.version,
                "deployment.environment": self.config.service.environment,
                "host.name": socket.gethostname(),
                "pra.version": self.config.service.version,
                **dict(self.config.service.attributes),
            }
        )
        self._provider = TracerProvider(resource=resource, sampler=samplers[sampling])
        exporter = span_exporter or self._otlp_exporter()
        if exporter is not None:
            self._provider.add_span_processor(SimpleSpanProcessor(exporter))
        self._tracer = self._provider.get_tracer("pra_hf.observability")

    def _otlp_exporter(self) -> Any | None:
        endpoint = self.config.otel.endpoint
        if not endpoint:
            return None
        try:
            if self.config.otel.protocol == "grpc":
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            else:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError as error:
            raise RuntimeError("Install the 'observability' extra to export OTel spans.") from error
        return OTLPSpanExporter(endpoint=endpoint)

    def _initialize_metrics(self) -> None:
        try:
            from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
        except ImportError as error:
            raise RuntimeError("Install the 'observability' extra to enable Prometheus.") from error
        self._registry = CollectorRegistry(auto_describe=True)
        constructors = {"counter": Counter, "gauge": Gauge, "histogram": Histogram}
        for name, (kind, labels, help_text) in _METRIC_SPECS.items():
            self._metrics[name] = constructors[kind](name, help_text, labels, registry=self._registry)

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None,
        *,
        parent_headers: Mapping[str, str] | None = None,
    ) -> Iterator[Any | None]:
        """Create one span, evaluating attributes only on the enabled path."""

        if not self.tracing_enabled:
            yield None
            return
        from opentelemetry import propagate

        context = propagate.extract(parent_headers or {}) if parent_headers else None
        with self._tracer.start_as_current_span(name, context=context) as current:
            values = attributes() if callable(attributes) else attributes
            for key, value in self._safe_attributes(values or {}).items():
                current.set_attribute(key, value)
            try:
                yield current
            except BaseException as error:
                current.record_exception(error)
                current.set_attribute("error.type", type(error).__name__)
                raise

    def inject(self, headers: MutableMapping[str, str] | None = None) -> MutableMapping[str, str]:
        """Inject W3C context only while tracing is explicitly active."""

        carrier: MutableMapping[str, str] = headers if headers is not None else {}
        if self.tracing_enabled:
            from opentelemetry import propagate

            propagate.inject(carrier)
        return carrier

    def hash_id(self, value: str | None) -> str | None:
        if value is None or not self.enabled:
            return None
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    def _safe_attributes(self, values: Mapping[str, Any]) -> dict[str, Any]:
        capture = self.config.content.capture
        result: dict[str, Any] = {}
        for key, value in values.items():
            lowered = key.lower()
            if capture == "none" and any(token in lowered for token in _CONTENT_KEYS):
                continue
            if value is None:
                continue
            if isinstance(value, (str, bool, int, float)):
                result[key] = value
            elif isinstance(value, (list, tuple)) and all(
                isinstance(item, (str, bool, int, float)) for item in value
            ):
                result[key] = list(value)
            else:
                result[key] = str(value)
        return result

    def _metric(self, name: str, labels: Mapping[str, str] | None) -> Any | None:
        metric = self._metrics.get(name)
        if metric is None:
            if self.metrics_enabled:
                raise KeyError(f"Unknown PRA metric: {name}")
            return None
        expected = _METRIC_SPECS[name][1]
        supplied = dict(labels or {})
        forbidden = _FORBIDDEN_LABELS.intersection(supplied)
        if forbidden:
            raise ValueError(f"High-cardinality Prometheus labels are forbidden: {sorted(forbidden)}")
        bounded = {key: str(supplied.get(key, "unknown")) for key in expected}
        return metric.labels(**bounded) if expected else metric

    def increment(self, name: str, amount: float = 1.0, *, labels: Mapping[str, str] | None = None) -> None:
        if not self.metrics_enabled:
            return
        self._metric(name, labels).inc(amount)

    def observe(self, name: str, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        if not self.metrics_enabled:
            return
        self._metric(name, labels).observe(float(value))

    def set_gauge(self, name: str, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        if not self.metrics_enabled:
            return
        self._metric(name, labels).set(float(value))

    def render_metrics(self) -> bytes:
        if not self.metrics_enabled:
            return b""
        from prometheus_client import generate_latest

        return generate_latest(self._registry)

    def start_metrics_server(self) -> tuple[str, int] | None:
        """Start the owned localhost metrics endpoint only when enabled."""

        if not self.metrics_enabled:
            return None
        if self._httpd is not None:
            return self._httpd.server_address
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path.split("?", 1)[0] != owner.config.prometheus.path:
                    self.send_error(404)
                    return
                payload = owner.render_metrics()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:
                return None

        self._httpd = PRAThreadingHTTPServer(
            (self.config.prometheus.host, self.config.prometheus.port), Handler
        )
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name=f"{self.config.service.name}-prometheus",
            daemon=True,
        )
        self._thread.start()
        return self._httpd.server_address

    def close(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._provider is not None:
            self._provider.shutdown()
            self._provider = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "tracing_enabled": self.tracing_enabled,
            "metrics_enabled": self.metrics_enabled,
            "service": self.config.service.name,
            "prometheus": (
                f"http://{self.config.prometheus.host}:{self.config.prometheus.port}{self.config.prometheus.path}"
                if self.metrics_enabled else None
            ),
            "trace_url": self.config.trace_url,
            "grafana_url": self.config.grafana_url,
            "content_capture": self.config.content.capture,
        }


DISABLED_OBSERVABILITY = Observability()


def engine_observability_capabilities(engine: str) -> dict[str, Any]:
    """Return conservative, version-independent telemetry integration metadata."""

    key = str(engine).replace("-", "_").lower()
    native_prometheus = {
        "vllm",
        "sglang",
        "openvino",
        "tensorrt_llm",
        "llama_cpp",
    }
    native_otel = {"vllm"}
    return {
        "otel": {
            "tracing": "native" if key in native_otel else "wrapped",
            "propagation": "w3c",
            "metrics": "native" if key in native_otel else "wrapped",
        },
        "prometheus": {
            "endpoint": "native" if key in native_prometheus else "wrapped",
            "path": "/metrics",
        },
        "native_metrics": {
            "vllm": ["scheduler", "kv_cache", "prefix_cache"],
            "sglang": ["scheduler", "radix_cache", "hicache"],
            "openvino": ["genai_performance", "ovms"],
            "tensorrt_llm": ["triton", "kv_cache"],
            "llama_cpp": ["prompt", "decode", "cache"],
            "mlx": ["native_timing", "metal_memory"],
        }.get(key, []),
        "explicit_enablement_required": True,
    }
