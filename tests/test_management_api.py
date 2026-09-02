from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from pra_hf.management import (
    MANAGEMENT_PROTOCOL,
    AuthMode,
    CapabilityStatus,
    LoadedModel,
    ManagementAPIConfig,
    ManagementAuthConfig,
    ManagementProvider,
    PRAProfileSummary,
    create_management_app,
    start_management_api,
)


@dataclass
class _Entry:
    record_type: str = "document"
    resource_version: str = "v1"
    detail_bytes: int = 128
    current_tier: str = "hot"
    request_pin_count: int = 0
    last_access_ns: int = 1_000_000_000
    task_id: str | None = "task-secret"
    session_id: str | None = "session-secret"
    security_scope: str | None = "tenant:secret"
    source_sha256: str = "abc123"


class _Storage:
    def __init__(self) -> None:
        self.entries = {"raw-secret-resource-key": _Entry()}
        self.calls: list[tuple[str, str]] = []
        self._maintenance_thread = object()

    def inspect(self):
        return {
            "usage": {
                "hot_bytes": 128, "warm_bytes": 0, "cold_bytes": 0,
                "total_native_bytes": 128, "source_only_objects": 0,
            },
            "objects": {"hot": 1, "warm": 0, "cold": 0, "source": 0},
            "metrics": {"evictions": 0, "reloads": 1, "promotions": 1, "hits": {"source": 0}},
            "policy": {
                "profile": "balanced",
                "hot": {"max_bytes": 1024},
                "warm": {"max_bytes": 2048},
                "cold": {"max_bytes": 4096},
            },
        }

    def promote(self, key, **_kwargs):
        self.calls.append(("promote", key))

    def demote_hot(self, key):
        self.calls.append(("demote", key))
        self.entries[key].current_tier = "warm"

    def run_maintenance(self):
        self.calls.append(("maintenance", ""))


class _Sessions:
    def inspect_all(self):
        return ({
            "session_id": "raw-session-secret",
            "prefix_message_count": 3,
            "known_resources": {"raw-resource": "v1"},
            "visible_materializations": [{
                "resource_id": "raw-resource", "token_count": 12,
            }],
            "turns": 2,
            "canonical_messages": [{"role": "user", "content": "do not expose"}],
            "engine_session_present": True,
            "prefix_cache_handle_present": True,
        },)


def _provider(**kwargs) -> ManagementProvider:
    return ManagementProvider(
        engine="hf",
        engine_version="1.2.3",
        capabilities={
            "integration_level": "E2", "native_kv": True,
            "typed_records": True, "session_state": True,
            "prefix_cache_mode": "session_state", "text_fallback": True,
        },
        models=[LoadedModel(model_id="org/model", revision="abc", profile="BALANCED")],
        profiles=[PRAProfileSummary(
            name="BALANCED", qualification_status=CapabilityStatus.VALIDATED,
        )],
        effective_config={"profile": "BALANCED", "api_key": "must-not-leak"},
        storage_manager=_Storage(),
        session_source=_Sessions(),
        observability={
            "otel": {"enabled": True}, "prometheus": {"enabled": True},
        },
        config_patch_handler=lambda values: values,
        **kwargs,
    )


def _client(provider=None, settings=None) -> TestClient:
    return TestClient(create_management_app(
        provider or _provider(),
        settings or ManagementAPIConfig(enabled=True),
    ))


def test_management_is_disabled_by_default_and_starts_no_server(monkeypatch) -> None:
    assert ManagementAPIConfig().enabled is False
    imported = False

    def fail_import(_name):
        nonlocal imported
        imported = True
        raise AssertionError("disabled management must not import/start Uvicorn")

    monkeypatch.setattr("builtins.__import__", fail_import)
    assert start_management_api(_provider(), ManagementAPIConfig()) is None
    assert imported is False


def test_unauthenticated_non_loopback_binding_is_rejected() -> None:
    with pytest.raises(ValueError, match="loopback"):
        create_management_app(
            _provider(), ManagementAPIConfig(enabled=True, host="0.0.0.0")
        )


def test_mtls_requires_complete_server_credentials() -> None:
    with pytest.raises(ValueError, match="tls_certfile"):
        create_management_app(
            _provider(),
            ManagementAPIConfig(
                enabled=True,
                auth=ManagementAuthConfig(mode=AuthMode.MTLS),
            ),
        )


def test_health_info_and_capabilities_use_versioned_contract() -> None:
    client = _client()
    assert client.get("/v1/pra/health").json()["protocol"] == MANAGEMENT_PROTOCOL
    assert client.get("/v1/pra/info").json()["engine"] == "hf"
    capabilities = client.get("/v1/pra/capabilities").json()
    assert capabilities["native_memory"]["status"] == "AVAILABLE"
    assert capabilities["management_api_version"] == MANAGEMENT_PROTOCOL


def test_openapi_swagger_and_redoc_are_available_only_for_enabled_app() -> None:
    client = _client()
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    assert schema.json()["info"]["title"] == "PRA Engine Management API"
    assert "/v1/pra/actions/maintenance" in schema.json()["paths"]
    promote = schema.json()["paths"]["/v1/pra/actions/promote"]["post"]
    assert promote.get("parameters", []) == []
    assert promote["security"] == [{"HTTPBearer": []}]
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200


def test_checked_in_openapi_schema_matches_runtime_contract() -> None:
    generated = create_management_app(
        ManagementProvider(engine="engine-adapter", capabilities={"text_fallback": True}),
        ManagementAPIConfig(enabled=True),
    ).openapi()
    checked_in = json.loads(Path(
        "docs/site/api/openapi/pra-management-v1.json"
    ).read_text(encoding="utf-8"))
    assert checked_in == generated


def test_models_profiles_and_pagination() -> None:
    client = _client()
    models = client.get("/v1/pra/models?offset=0&limit=1").json()
    assert models["total"] == 1
    assert models["items"][0]["model_id"] == "org/model"
    assert client.get("/v1/pra/models/org/model").status_code == 200
    assert client.get("/v1/pra/profiles/BALANCED").json()["qualification_status"] == "VALIDATED"


def test_resources_and_sessions_never_expose_raw_content_or_identifiers() -> None:
    client = _client()
    resources = client.get("/v1/pra/resources").json()
    sessions = client.get("/v1/pra/sessions").json()
    encoded = json.dumps({"resources": resources, "sessions": sessions})
    assert resources["items"][0]["authorization_scope"] == "restricted"
    assert sessions["items"][0]["selected_token_total"] == 12
    for secret in (
        "raw-secret-resource-key", "raw-session-secret", "raw-resource",
        "do not expose", "task-secret", "session-secret", "tenant:secret",
    ):
        assert secret not in encoded


def test_config_redacts_secrets_and_distinguishes_restart_fields() -> None:
    client = _client()
    assert client.get("/v1/pra/config").json()["effective"]["api_key"] == "[REDACTED]"
    changed = client.patch("/v1/pra/config", json={"profile": "ECONOMY"})
    assert changed.status_code == 200
    assert changed.json()["observed_revision"] == 2
    restart = client.patch("/v1/pra/config", json={"device": "cuda"})
    assert restart.status_code == 409
    assert restart.json()["error"]["code"] == "RESTART_REQUIRED"
    assert restart.json()["error"]["restart_fields"] == ["device"]


def test_config_patch_without_engine_handler_is_not_a_successful_noop() -> None:
    client = _client(ManagementProvider(engine="openvino"))
    response = client.patch("/v1/pra/config", json={"profile": "ECONOMY"})
    assert response.status_code == 501
    assert response.json()["error"]["code"] == "CONFIG_PATCH_NOT_SUPPORTED"


def test_storage_actions_are_real_idempotent_and_audited() -> None:
    provider = _provider()
    client = _client(provider)
    resource_id = client.get("/v1/pra/resources").json()["items"][0]["resource_id"]
    body = {"resource_id": resource_id, "idempotency_key": "same-action"}
    first = client.post("/v1/pra/actions/promote", json=body)
    second = client.post("/v1/pra/actions/promote", json=body)
    assert first.status_code == second.status_code == 200
    assert second.json()["idempotent_replay"] is True
    assert provider.storage_manager.calls == [("promote", "raw-secret-resource-key")]
    conflict = client.post(
        "/v1/pra/actions/promote",
        json={"resource_id": resource_id, "tenant_id": "other", "idempotency_key": "same-action"},
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"
    events = client.get("/v1/pra/audit").json()["items"]
    assert events[0]["event"] == "RESOURCE_PROMOTED"


def test_unsupported_actions_are_not_noops() -> None:
    client = _client()
    response = client.post("/v1/pra/actions/reload-bundle", json={"bundle": "org/bundle"})
    assert response.status_code == 501
    assert response.json()["error"]["code"] == "ACTION_NOT_SUPPORTED"


def test_static_bearer_auth_enforces_scopes() -> None:
    settings = ManagementAPIConfig(
        enabled=True,
        auth=ManagementAuthConfig(
            mode=AuthMode.STATIC_BEARER,
            token="secret-token",
            scopes=("pra:read",),
        ),
    )
    client = _client(settings=settings)
    assert client.get("/v1/pra/health").status_code == 401
    headers = {"Authorization": "Bearer secret-token"}
    assert client.get("/v1/pra/health", headers=headers).status_code == 200
    assert client.get("/v1/pra/models", headers=headers).status_code == 403
    assert client.patch(
        "/v1/pra/config", headers=headers, json={"profile": "ECONOMY"}
    ).status_code == 403


@pytest.mark.parametrize(
    "engine",
    [
        "hf", "vllm", "sglang", "mlx", "openvino", "tensorrt_llm",
        "airllm", "llama_cpp", "ollama", "freetoken",
    ],
)
def test_all_engine_sidecars_share_the_core_contract(engine: str) -> None:
    client = _client(ManagementProvider(engine=engine, capabilities={"text_fallback": True}))
    for path in (
        "health", "info", "capabilities", "config", "state", "models",
        "profiles", "resources", "sessions", "storage", "observability",
        "metrics-link", "trace-link", "audit",
    ):
        assert client.get(f"/v1/pra/{path}").status_code == 200, (engine, path)
