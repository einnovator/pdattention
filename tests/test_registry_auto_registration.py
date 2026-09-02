from __future__ import annotations

import json
import socket
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pra_control.config import ControlPlaneConfig, FleetConfig, ServiceLinkConfig
from pra_control.fleet import FleetService
from pra_hf.registry_registration import (
    RegistryRegistrationClient,
    RuntimeInstanceIdentity,
    RuntimeRegistryConfig,
    resolve_instance_id,
)
from pra_hf.management import (
    ManagementAPIConfig, ManagementProvider, start_management_api, stop_management_api,
)
from pra_registry.api import create_registry_app
from pra_registry.config import (
    InstanceRegistrationPolicy, RegistryAuthConfig, RegistryConfig,
    RegistryObservabilityConfig,
)
from pra_registry.database import ManagedInstanceRecord, RegistryDatabase


def registration(instance_id: str = "engine-1") -> dict:
    return {
        "instance_id": instance_id,
        "instance_type": "ENGINE",
        "name": "vllm-west-1",
        "environment": "production",
        "region": "west",
        "cluster": "gpu",
        "namespace": "inference",
        "host": "worker-1",
        "management_url": "https://worker-1:9101",
        "inference_url": "https://worker-1:8000",
        "pra_version": "1.0",
        "component_version": "1.0",
        "engine_kind": "vllm",
        "engine_version": "0.10",
        "health": "healthy",
        "started_at": time.time(),
        "capabilities": {"native_memory": "AVAILABLE"},
        "models": [{"model_id": "Qwen/Qwen3-4B", "profile": "BALANCED"}],
        "runtime_summary": {"mode": "E2"},
        "observability": {"metrics_url": "https://worker-1:9464/metrics"},
        "observed_revision": 1,
        "in_sync": True,
        "labels": {"team": "platform"},
    }


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@pytest.fixture
def registry() -> TestClient:
    config = RegistryConfig(
        database_url="sqlite:///:memory:",
        instance_registration=InstanceRegistrationPolicy(offline_after_seconds=15),
    )
    database = RegistryDatabase("sqlite:///:memory:", create_schema=True)
    return TestClient(create_registry_app(config, database))


def test_instance_registration_lifecycle_is_idempotent_and_audited(registry: TestClient) -> None:
    first = registry.post("/v1/instances/register", json=registration())
    second = registry.post("/v1/instances/register", json={
        **registration(), "runtime_summary": {"mode": "E2", "requests": 4},
    })
    assert first.status_code == second.status_code == 200
    assert registry.get("/v1/instances").json()["total"] == 1
    heartbeat = registry.post("/v1/instances/engine-1/heartbeat", json={
        "health": "degraded", "uptime_seconds": 12, "observed_revision": 2,
        "runtime_summary": {"mode": "E0"}, "timestamp": time.time(),
    }).json()
    assert heartbeat["status"] == "DEGRADED"
    observed = registry.patch("/v1/instances/engine-1/observed", json={
        "observed_revision": 3, "in_sync": False, "drift_fields": ["profile"],
    }).json()
    assert observed["drift_fields"] == ["profile"]
    stopped = registry.post("/v1/instances/engine-1/deregister").json()
    assert stopped["status"] == "OFFLINE"
    actions = [row["action"] for row in registry.get("/v1/audit?resource_id=engine-1").json()["items"]]
    assert actions == [
        "INSTANCE_REGISTERED", "INSTANCE_REGISTERED", "INSTANCE_HEARTBEAT",
        "INSTANCE_UPDATED", "INSTANCE_DEREGISTERED",
    ]


def test_instance_identity_conflict_and_secret_rejection(registry: TestClient) -> None:
    assert registry.post("/v1/instances/register", json=registration()).status_code == 200
    conflict = registry.post("/v1/instances/register", json={
        **registration(), "instance_type": "GATEWAY", "name": "other",
        "engine_kind": None,
    })
    assert conflict.status_code == 409
    unsafe = registry.post("/v1/instances/register", json={
        **registration("unsafe"), "metadata": {"token": "do-not-store"},
    })
    assert unsafe.status_code == 422
    assert "do-not-store" not in json.dumps(registry.get("/v1/audit").json())


def test_instance_desired_state_is_separate_from_observed(registry: TestClient) -> None:
    registry.post("/v1/instances/register", json=registration())
    registry.post("/v1/deployments", json={
        "id": "prod-vllm", "environment": "production", "cluster": "gpu",
        "engine_instance_selector": {"labels": {"team": "platform"}},
        "desired_model_id": "Qwen/Qwen3-4B", "desired_profile_id": "QUALITY_MAX",
        "desired_mode": "E2",
    })
    desired = registry.get("/v1/instances/engine-1/desired").json()
    assert desired["desired_revision"] == 1
    assert desired["desired"]["desired_profile_id"] == "QUALITY_MAX"
    assert registry.get("/v1/instances/engine-1").json()["models"][0]["profile"] == "BALANCED"


def test_registration_admission_policy_rejects_wrong_environment() -> None:
    config = RegistryConfig(
        database_url="sqlite:///:memory:",
        instance_registration=InstanceRegistrationPolicy(allowed_environments=["production"]),
    )
    client = TestClient(create_registry_app(
        config, RegistryDatabase("sqlite:///:memory:", create_schema=True),
    ))
    assert client.post("/v1/instances/register", json={
        **registration(), "environment": "development",
    }).status_code == 403


def test_heartbeat_timeout_marks_instance_offline_without_clean_shutdown() -> None:
    database = RegistryDatabase("sqlite:///:memory:", create_schema=True)
    client = TestClient(create_registry_app(
        RegistryConfig(
            database_url="sqlite:///:memory:",
            instance_registration=InstanceRegistrationPolicy(offline_after_seconds=15),
        ), database,
    ))
    client.post("/v1/instances/register", json=registration("expired"))
    with database.session_factory() as session:
        row = session.get(ManagedInstanceRecord, "expired")
        assert row is not None
        row.last_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=16)
        session.commit()
    assert client.get("/v1/instances/expired").json()["status"] == "OFFLINE"
    assert "INSTANCE_OFFLINE" in {
        row["action"] for row in client.get("/v1/audit?resource_id=expired").json()["items"]
    }


def test_registry_auth_failure_does_not_create_an_instance() -> None:
    client = TestClient(create_registry_app(
        RegistryConfig(
            database_url="sqlite:///:memory:",
            auth=RegistryAuthConfig(
                mode="service_credentials", service_credentials={"engine": "correct-token"},
            ),
        ),
        RegistryDatabase("sqlite:///:memory:", create_schema=True),
    ))
    response = client.post(
        "/v1/instances/register", json=registration(),
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == 401


def test_registry_instance_prometheus_metrics() -> None:
    client = TestClient(create_registry_app(
        RegistryConfig(
            database_url="sqlite:///:memory:",
            observability=RegistryObservabilityConfig(enabled=True, prometheus_enabled=True),
        ),
        RegistryDatabase("sqlite:///:memory:", create_schema=True),
    ))
    client.post("/v1/instances/register", json=registration())
    client.get("/v1/instances")
    metrics = client.get("/metrics").text
    assert "pra_registry_instances_online 1" in metrics
    assert 'pra_registry_registration_total{result="success"} 1' in metrics
    assert 'pra_registry_heartbeat_age_seconds{instance_id="engine-1"}' in metrics


def test_instance_id_persists_and_explicit_identity_wins(tmp_path: Path) -> None:
    path = tmp_path / "engine.json"
    first = resolve_instance_id("ENGINE", identity_file=path)
    assert resolve_instance_id("ENGINE", identity_file=path) == first
    assert resolve_instance_id("ENGINE", "configured", path) == "configured"


def test_resilient_and_required_registration_modes(tmp_path: Path) -> None:
    payload = registration()
    resilient = RegistryRegistrationClient(
        RuntimeRegistryConfig(
            enabled=True, url="http://registry", heartbeat_seconds=0.1,
            retry_initial_seconds=0.05, retry_max_seconds=0.1,
            instance=RuntimeInstanceIdentity(id="resilient", identity_file=str(tmp_path / "r.json")),
        ),
        "ENGINE", lambda instance_id: {**payload, "instance_id": instance_id},
    )
    attempts = []

    def flaky(method, path, body=None):
        attempts.append(path)
        if len(attempts) == 1:
            raise OSError("registry unavailable")
        return {}

    resilient._request = flaky
    resilient.start()
    time.sleep(0.25)
    resilient.stop()
    assert resilient.status()["registration_failure_total"] >= 1
    assert resilient.status()["registration_success_total"] >= 1

    required = RegistryRegistrationClient(
        RuntimeRegistryConfig(
            enabled=True, required=True, url="http://registry",
            instance=RuntimeInstanceIdentity(id="required"),
        ),
        "ENGINE", lambda instance_id: {**payload, "instance_id": instance_id},
    )
    required._request = lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("down"))
    with pytest.raises(OSError, match="down"):
        required.start()


def test_no_registry_url_creates_no_worker() -> None:
    client = RegistryRegistrationClient(
        RuntimeRegistryConfig(), "ENGINE", lambda _instance_id: registration(),
    )
    client.start()
    assert client.thread is None
    assert client.status()["status"] == "disabled"


def test_control_plane_registry_discovery_uses_live_instances(tmp_path: Path) -> None:
    class Registry:
        async def instances(self, *, instance_type=None):
            assert instance_type == "ENGINE"
            return [registration() | {"status": "ONLINE"}]

        async def deployments(self):
            return []

        async def close(self):
            pass

    config = ControlPlaneConfig(
        database_url=f"sqlite:///{tmp_path / 'control.db'}",
        registry=ServiceLinkConfig(url=None),
        fleet=FleetConfig(discovery_mode="registry"),
    )
    from pra_control.persistence import ControlStore
    import asyncio
    store = ControlStore(config.database_url)
    service = FleetService(config, store, registry_client=Registry())
    targets = asyncio.run(service.targets())
    assert [(row.name, row.management_url) for row in targets] == [
        ("vllm-west-1", "https://worker-1:9101"),
    ]


def test_engine_startup_registers_and_shutdown_deregisters_over_http(tmp_path: Path) -> None:
    import uvicorn

    registry_port = free_port()
    management_port = free_port()
    app = create_registry_app(
        RegistryConfig(database_url="sqlite:///:memory:"),
        RegistryDatabase("sqlite:///:memory:", create_schema=True),
    )
    registry_server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=registry_port, log_level="warning",
    ))
    registry_thread = threading.Thread(target=registry_server.run, daemon=True)
    registry_thread.start()
    deadline = time.monotonic() + 5
    while not registry_server.started and registry_thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)

    provider = ManagementProvider(
        engine="test", capabilities={"text_fallback": True},
        effective_config={"inference_url": "http://127.0.0.1:8000"},
    )
    settings = ManagementAPIConfig(
        enabled=True, port=management_port,
        registry=RuntimeRegistryConfig(
            url=f"http://127.0.0.1:{registry_port}", required=True,
            heartbeat_seconds=0.1,
            instance=RuntimeInstanceIdentity(
                id="http-engine", name="http-engine",
                identity_file=str(tmp_path / "engine.json"),
            ),
        ),
    )
    management_server = start_management_api(provider, settings)
    try:
        registered = TestClient(app).get("/v1/instances/http-engine").json()
        assert registered["status"] == "ONLINE"
        assert registered["instance_type"] == "ENGINE"
        assert provider.registry_state()["metrics"]["pra_registry_connected"] == 1
    finally:
        stop_management_api(management_server)
    assert TestClient(app).get("/v1/instances/http-engine").json()["status"] == "OFFLINE"
    registry_server.should_exit = True
    registry_thread.join(timeout=5)
