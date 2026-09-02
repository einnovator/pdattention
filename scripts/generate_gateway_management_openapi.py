"""Generate the checked-in PRA Gateway Management OpenAPI contract."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from pra_hf.deployment import PRAEngineCapabilities, PRAEngineResult
from pra_hf.gateway import PRAGateway
from pra_hf.gateway_management import (
    GatewayManagementAPIConfig,
    GatewayManagementProvider,
    GatewayMetricRecorder,
    GatewayUpstreamRouter,
    UpstreamCreate,
    create_gateway_management_app,
)


class _Observability:
    tracing_enabled = False
    metrics_enabled = False
    config = None

    def increment(self, *_args, **_kwargs): pass
    def observe(self, *_args, **_kwargs): pass
    def set_gauge(self, *_args, **_kwargs): pass

    @contextmanager
    def span(self, *_args, **_kwargs):
        yield None

    @staticmethod
    def hash_id(value):
        return "none" if value is None else str(value)


class _Adapter:
    def capabilities(self) -> PRAEngineCapabilities:
        return PRAEngineCapabilities(adapter="openapi", text_fallback=True)

    def prepare_session(self, request):
        return None

    def generate(self, request):
        return PRAEngineResult(text="")

    def stream(self, request):
        return iter(())

    def close_session(self, session_id):
        return None


def main() -> None:
    adapter = _Adapter()
    settings = GatewayManagementAPIConfig(enabled=True)
    telemetry = _Observability()
    gateway = PRAGateway(adapter, mode="G00", observability=telemetry)
    router = GatewayUpstreamRouter(
        UpstreamCreate(upstream_id="default", name="openapi", base_url="http://127.0.0.1:8000"),
        adapter,
    )
    provider = GatewayManagementProvider(
        gateway, router, settings, GatewayMetricRecorder(telemetry),
    )
    destination = Path("docs/site/api/openapi/pra-gateway-management-v1.json")
    destination.write_text(
        json.dumps(create_gateway_management_app(provider, settings).openapi(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(destination)


if __name__ == "__main__":
    main()
