from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

from pra_registry.api import create_registry_app
from pra_registry.config import RegistryAuthConfig, RegistryConfig, RegistryObservabilityConfig
from pra_registry.database import RegistryDatabase


MODEL = {
    "id": "model-qwen", "provider": "Qwen", "repo": "Qwen/Qwen3-0.6B",
    "revision": "base-sha", "architecture": "Qwen3ForCausalLM",
}
BUNDLE = {
    "id": "bundle-qwen", "immutable_revision": "bundle-sha",
    "base_model_id": "model-qwen", "base_model_revision": "base-sha",
    "trust": "einnovator-qualified",
    "artifact_sources": [{
        "id": "source-qwen", "source_type": "huggingface",
        "locator": "EInnovator/pra-qwen3-0.6b", "immutable_revision": "bundle-sha",
        "credential_reference": "vault/hf/registry",
    }],
}


@pytest.fixture()
def client() -> TestClient:
    database = RegistryDatabase("sqlite:///:memory:", create_schema=True)
    return TestClient(create_registry_app(RegistryConfig(database_url="sqlite:///:memory:"), database))


def seed(client: TestClient) -> None:
    assert client.post("/v1/models", json=MODEL).status_code == 201
    assert client.post("/v1/bundles", json=BUNDLE).status_code == 201


def test_health_openapi_swagger_and_redoc(client: TestClient) -> None:
    assert client.get("/health").json()["protocol"] == "pra-registry/1"
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "PRA Registry API"
    assert "/v1/resolve/bundle" in schema["paths"]
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200


def test_model_bundle_crud_soft_delete_and_pagination(client: TestClient) -> None:
    seed(client)
    page = client.get("/v1/models?limit=1&offset=0").json()
    assert page["total"] == 1 and page["items"][0]["repo"] == MODEL["repo"]
    assert client.patch("/v1/models/model-qwen", json={"parameter_class": "sub-1B"}).json()["parameter_class"] == "sub-1B"
    deleted = client.delete("/v1/models/model-qwen").json()
    assert deleted["deleted"] is True and deleted["approval_state"] == "DEPRECATED"
    source = client.get("/v1/bundles/bundle-qwen").json()["artifact_sources"][0]
    assert source["credential_reference"] == "vault/hf/registry"


def test_bundle_revision_and_qualification_are_immutable(client: TestClient) -> None:
    seed(client)
    assert client.patch("/v1/bundles/bundle-qwen", json={"immutable_revision": "changed"}).status_code == 422
    qualification = {
        "id": "qual-1", "model_id": "model-qwen", "model_revision": "base-sha",
        "bundle_id": "bundle-qwen", "bundle_revision": "bundle-sha",
        "engine": "hf", "workload": "qasper", "mode": "E2", "cohort_size": 50,
        "metrics": {"f1": 0.72},
    }
    assert client.post("/v1/qualifications", json=qualification).status_code == 201
    assert client.get("/v1/qualifications/qual-1").json()["metrics"]["f1"] == 0.72
    assert client.patch("/v1/qualifications/qual-1", json={"metrics": {}}).status_code == 405


def test_approval_workflow_is_append_only_and_audited(client: TestClient) -> None:
    seed(client)
    assert client.post("/v1/bundles/bundle-qwen/approve?reason=qualified").status_code == 200
    assert client.get("/v1/bundles/bundle-qwen").json()["approval_state"] == "APPROVED"
    approvals = client.get("/v1/approvals").json()
    assert approvals["total"] == 1 and approvals["items"][0]["reason"] == "qualified"
    audit = client.get("/v1/audit?resource_id=bundle-qwen").json()
    assert [row["action"] for row in audit["items"]] == ["create", "approval"]
    assert client.delete("/v1/audit/1").status_code in {404, 405}


def test_desired_state_revision_increments_and_resolves(client: TestClient) -> None:
    seed(client)
    deployment = {
        "id": "prod-qwen", "environment": "production", "cluster": "gpu-east",
        "desired_model_id": "model-qwen", "desired_bundle_id": "bundle-qwen",
        "desired_profile_id": "balanced", "desired_mode": "E2",
    }
    assert client.post("/v1/deployments", json=deployment).json()["desired_revision"] == 1
    assert client.patch("/v1/deployments/prod-qwen", json={"desired_mode": "E0"}).json()["desired_revision"] == 2
    desired = client.get("/v1/deployments/prod-qwen/desired").json()
    assert desired["desired_revision"] == 2 and desired["desired"]["desired_mode"] == "E0"
    resolved = client.post("/v1/resolve/deployment", json={"environment": "production", "cluster": "gpu-east"}).json()
    assert resolved["desired"]["id"] == "prod-qwen"


def test_bundle_resolver_is_deterministic_and_returns_evidence(client: TestClient) -> None:
    seed(client)
    second = {**BUNDLE, "id": "bundle-z", "immutable_revision": "z-sha", "artifact_sources": []}
    assert client.post("/v1/bundles", json=second).status_code == 201
    assert client.post("/v1/bundles/bundle-qwen/approve").status_code == 200
    compatibility = {
        "id": "compat-1", "engine": "hf", "engine_version_range": ">=4.45,<5",
        "bundle_id": "bundle-qwen",
        "model_id": "model-qwen", "execution_mode": "E2",
        "recommendation_status": "VALIDATED", "limitations": ["bounded cohort"],
    }
    assert client.post("/v1/compatibility", json=compatibility).status_code == 201
    body = {"model": MODEL["repo"], "model_revision": "base-sha", "engine": "hf", "engine_version": "4.52"}
    ids = [client.post("/v1/resolve/bundle", json=body).json()["selected_bundle"]["id"] for _ in range(3)]
    assert ids == ["bundle-qwen"] * 3
    assert client.post("/v1/resolve/bundle", json=body).json()["limitations"] == ["bounded cohort"]
    matching = client.get("/v1/compatibility/resolve?engine=hf&engine_version=4.52").json()
    excluded = client.get("/v1/compatibility/resolve?engine=hf&engine_version=5.1").json()
    assert [row["id"] for row in matching["items"]] == ["compat-1"]
    assert excluded["items"] == []


def test_secret_payloads_are_rejected_and_audit_does_not_contain_secret(client: TestClient) -> None:
    seed(client)
    unsafe = {**BUNDLE, "id": "unsafe", "artifact_sources": [{
        "id": "unsafe-source", "source_type": "s3", "locator": "s3://bucket/key",
        "immutable_revision": "sha", "credential_reference": "Bearer raw-secret",
    }]}
    assert client.post("/v1/bundles", json=unsafe).status_code == 422
    assert "raw-secret" not in json.dumps(client.get("/v1/audit").json())


def test_static_token_and_service_scope_boundaries() -> None:
    settings = RegistryConfig(
        database_url="sqlite:///:memory:",
        auth=RegistryAuthConfig(mode="static_token", static_token="admin-secret"),
    )
    client = TestClient(create_registry_app(settings, RegistryDatabase("sqlite:///:memory:", create_schema=True)))
    assert client.get("/v1/models").status_code == 401
    headers = {"Authorization": "Bearer admin-secret"}
    assert client.post("/v1/models", json=MODEL, headers=headers).status_code == 201
    assert client.get("/v1/models", headers=headers).status_code == 200


def test_service_credentials_cannot_approve(client: TestClient) -> None:
    settings = RegistryConfig(
        database_url="sqlite:///:memory:",
        auth=RegistryAuthConfig(mode="service_credentials", service_credentials={"ci": "ci-secret"}),
    )
    scoped = TestClient(create_registry_app(settings, RegistryDatabase("sqlite:///:memory:", create_schema=True)))
    headers = {"Authorization": "Bearer ci-secret"}
    assert scoped.post("/v1/models", json=MODEL, headers=headers).status_code == 201
    assert scoped.post("/v1/approvals", json={
        "resource_type": "model", "resource_id": "model-qwen", "version": "base-sha",
        "state": "APPROVED", "approver": "ignored", "reason": "ci should not approve",
    }, headers=headers).status_code == 403


def test_hf_import_endpoint_is_server_side_and_idempotent(client: TestClient, monkeypatch) -> None:
    from pra_registry.connectors import HuggingFaceConnector
    from pra_registry.contracts import BundleCreate, ModelCreate
    monkeypatch.setattr(HuggingFaceConnector, "inspect", lambda self, repo_id, revision=None: (
        ModelCreate(**MODEL), BundleCreate(**BUNDLE),
    ))
    first = client.post("/v1/import/huggingface", json={"repo_id": "EInnovator/pra-qwen"})
    second = client.post("/v1/import/huggingface", json={"repo_id": "EInnovator/pra-qwen"})
    assert first.status_code == second.status_code == 200
    assert second.json()["bundle"]["immutable_revision"] == "bundle-sha"


def test_unauthenticated_non_loopback_binding_is_rejected() -> None:
    with pytest.raises(ValueError, match="loopback"):
        create_registry_app(RegistryConfig(host="0.0.0.0", database_url="sqlite:///:memory:"))


def test_oidc_cannot_start_without_verification_metadata() -> None:
    with pytest.raises(ValueError, match="issuer, audience, and JWKS"):
        create_registry_app(RegistryConfig(
            database_url="sqlite:///:memory:",
            auth=RegistryAuthConfig(mode="oidc", oidc_issuer="https://id.example"),
        ))


def test_prometheus_is_default_off_and_reports_registry_operations_when_enabled() -> None:
    disabled = TestClient(create_registry_app(
        RegistryConfig(database_url="sqlite:///:memory:"),
        RegistryDatabase("sqlite:///:memory:", create_schema=True),
    ))
    assert disabled.get("/metrics").status_code == 404
    enabled = TestClient(create_registry_app(
        RegistryConfig(
            database_url="sqlite:///:memory:",
            observability=RegistryObservabilityConfig(enabled=True, prometheus_enabled=True),
        ),
        RegistryDatabase("sqlite:///:memory:", create_schema=True),
    ))
    enabled.get("/v1/models")
    metrics = enabled.get("/metrics").text
    assert 'pra_registry_operations_total{operation="db"} 1' in metrics
    assert "pra_registry_requests_total" in metrics


def test_alembic_upgrade_creates_registry_schema(tmp_path: Path) -> None:
    config = Config(str(Path(__file__).parents[1] / "alembic.ini"))
    database = tmp_path / "registry.db"
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database.as_posix()}")
    command.upgrade(config, "head")
    engine = RegistryDatabase(f"sqlite:///{database.as_posix()}").engine
    from sqlalchemy import inspect
    names = set(inspect(engine).get_table_names())
    assert {
        "registry_models", "registry_bundles", "registry_audit_events",
        "registry_managed_instances",
    } <= names


def test_checked_in_openapi_matches_runtime_contract() -> None:
    checked = Path(__file__).parents[1] / "docs/site/api/openapi/pra-registry-v1.json"
    generated = create_registry_app(
        RegistryConfig(database_url="sqlite:///:memory:"),
        RegistryDatabase("sqlite:///:memory:", create_schema=True),
    ).openapi()
    published = json.loads(checked.read_text(encoding="utf-8"))
    assert published["info"]["title"] == generated["info"]["title"]
    assert set(published["paths"]) == set(generated["paths"])
    for path in published["paths"]:
        assert set(published["paths"][path]) == set(generated["paths"][path])
    required = {
        "ModelCreate", "BundleCreate", "ProfileCreate", "CompatibilityCreate",
        "QualificationCreate", "DeploymentCreate", "PolicyCreate", "ApprovalCreate",
        "BundleResolveRequest", "ProfileResolveRequest", "DeploymentResolveRequest",
        "ManagedInstanceRegister", "ManagedInstanceHeartbeat", "ManagedInstanceObservedPatch",
    }
    assert required <= set(published["components"]["schemas"])
    assert required <= set(generated["components"]["schemas"])
    for name in required:
        assert set(published["components"]["schemas"][name].get("properties", {})) == set(
            generated["components"]["schemas"][name].get("properties", {})
        )
    assert published["components"]["securitySchemes"]["HTTPBearer"] == {
        "type": "http", "scheme": "bearer",
    }
