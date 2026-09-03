from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
from fastapi.testclient import TestClient
from click.testing import CliRunner

from pra_registry.api import create_registry_app
from pra_registry.config import RegistryConfig
from pra_registry.database import RegistryDatabase
from pra_router.adapters import (
    AtomicFileTransport,
    AgentGatewayAdapter,
    BifrostRouterAdapter,
    KubernetesGAIEAdapter,
    LiteLLMRouterAdapter,
    MemoryRouterTransport,
    ReferenceRouterAdapter,
    stable_backend_name,
)
from pra_router.controller import ReconcilePlan, RouterController, RouterDesiredState
from pra_router.cli import router_cli
from pra_router.reference import ReferenceRouter, create_reference_router_app
from pra_router.registry import InMemoryRouterSource
from pra_control.operations import OPERATION_CATALOG, TOOL_CATALOG


def desired_state(kind: str = "pra-reference") -> dict:
    backend = {
        "id": "engine-a:qwen", "pool_ids": ["qwen-eu"], "engine_instance_id": "engine-a",
        "runtime_model_id": "qwen", "inference_url": "http://engine-a:8000",
        "engine": "vllm", "engine_version": "0.11", "model_id": "Qwen/Qwen3-32B",
        "model_revision": "base-sha", "model_fingerprint": "fp", "bundle_id": "bundle-qwen",
        "bundle_revision": "bundle-sha", "profile": "BALANCED", "modes": ["selected-context", "native-memory"],
        "qualification_tier": "ENGINE_QUALIFIED", "approval_state": "APPROVED", "region": "eu",
        "cluster": "gpu", "health": "READY", "maintenance": False, "weight": 2.0,
        "cost": 0.2, "labels": {"tier": "fast"}, "metadata": {"hardware": "H100"},
    }
    return {
        "router": {"id": "router-eu", "kind": kind, "management_url": "http://router-admin", "observed_revision": 0},
        "desired_revision": 3,
        "routes": [{
            "id": "qwen32", "public_model": "pra/qwen3-32b", "route_kind": "llm",
            "policy": {"id": "balanced", "strategy": "weighted", "constraints": {}, "preferences": {}, "fallback": []},
            "pools": [{
                "id": "qwen-eu", "model_id": "Qwen/Qwen3-32B", "selectors": {"region": "eu"},
                "metadata": {"port": 8000}, "fallback": False, "backends": [backend], "excluded": [],
            }],
        }],
        "in_sync": False,
    }


def test_registry_compiles_qualification_aware_router_state() -> None:
    database = RegistryDatabase("sqlite:///:memory:", create_schema=True)
    client = TestClient(create_registry_app(RegistryConfig(database_url="sqlite:///:memory:"), database))
    assert client.post("/v1/routing-policies", json={
        "id": "qualified", "strategy": "weighted",
        "constraints": {"health": "READY", "approved_only": True, "minimum_evidence": "ENGINE_QUALIFIED", "required_modes": ["selected-context"]},
        "preferences": {"qualified_first": True},
    }).status_code == 201
    assert client.post("/v1/model-pools", json={
        "id": "qwen-eu", "model_id": "Qwen/Qwen3-32B", "model_revision": "base-sha",
        "selectors": {"region": "eu", "engine": ["vllm", "sglang"]},
    }).status_code == 201
    common = {
        "pool_ids": ["qwen-eu"], "runtime_model_id": "qwen", "engine": "vllm",
        "model_id": "Qwen/Qwen3-32B", "model_revision": "base-sha", "modes": ["selected-context"],
        "qualification_tier": "ENGINE_QUALIFIED", "approval_state": "APPROVED", "region": "eu",
        "cluster": "gpu", "health": "READY",
    }
    assert client.post("/v1/backend-endpoints", json={
        **common, "id": "ready", "inference_url": "http://ready:8000",
    }).status_code == 201
    assert client.post("/v1/backend-endpoints", json={
        **common, "id": "maintenance", "inference_url": "http://maintenance:8000", "maintenance": True,
    }).status_code == 201
    assert client.post("/v1/routers", json={
        "id": "router-eu", "kind": "litellm", "management_url": "http://router-admin",
        "credential_reference": "LITELLM_ADMIN_TOKEN", "region": "eu",
    }).status_code == 201
    assert client.post("/v1/routes", json={
        "id": "route-qwen", "public_model": "pra/qwen3-32b", "policy_id": "qualified", "pool_ids": ["qwen-eu"],
    }).status_code == 201
    assert client.post("/v1/route-bindings", json={
        "id": "bind-qwen-eu", "route_id": "route-qwen", "router_id": "router-eu",
    }).status_code == 201

    desired = client.get("/v1/routers/router-eu/desired").json()
    pool = desired["routes"][0]["pools"][0]
    assert [row["id"] for row in pool["backends"]] == ["ready"]
    assert pool["excluded"] == [{"id": "maintenance", "reason": "maintenance"}]
    assert desired["router"]["credential_reference"] == "LITELLM_ADMIN_TOKEN"
    assert client.patch("/v1/routes/route-qwen", json={"enabled": False}).json()["desired_revision"] == 2
    assert client.patch("/v1/routes/route-qwen", json={"pool_ids": ["missing"]}).status_code == 409
    assert client.patch("/v1/routers/router-eu", json={"management_url": "ftp://unsafe"}).status_code == 422
    assert client.patch("/v1/routers/router-eu", json={"metadata": {"api_key": "secret"}}).status_code == 422


def test_all_adapters_compile_the_same_route_graph() -> None:
    classes = (
        LiteLLMRouterAdapter, AgentGatewayAdapter, KubernetesGAIEAdapter,
        ReferenceRouterAdapter, BifrostRouterAdapter,
    )
    compiled = {cls.kind: cls(MemoryRouterTransport()).compile(RouterDesiredState.model_validate(desired_state(cls.kind))) for cls in classes}
    assert compiled["litellm"]["model_list"][0]["model_name"] == "pra/qwen3-32b"
    assert compiled["agentgateway"]["routes"][0]["name"] == "qwen32"
    assert isinstance(compiled["agentgateway"]["backends"], list)
    matcher = compiled["agentgateway"]["routes"][0]["matches"][0]["headers"][0]
    assert matcher["value"] == {"exact": "pra/qwen3-32b"}
    resources = compiled["kubernetes-gaie"]["items"]
    assert compiled["kubernetes-gaie"]["kind"] == "List"
    assert {row["kind"] for row in resources} == {"InferencePool", "HTTPRoute"}
    assert resources[0]["apiVersion"] == "inference.networking.k8s.io/v1"
    assert compiled["pra-reference"]["routes"][0]["backends"][0]["model"] == "qwen"
    targets = compiled["bifrost"]["governance"]["routing_rules"][0]["targets"]
    assert sum(row["weight"] for row in targets) == pytest.approx(1.0)
    assert stable_backend_name("router-eu", desired_state()["routes"][0]["pools"][0]["backends"][0]).startswith("pra-")


def test_controller_preview_apply_verify_and_last_good_failure() -> None:
    state = desired_state()
    source = InMemoryRouterSource({"router-eu": state})
    transport = MemoryRouterTransport()
    controller = RouterController(source, {"pra-reference": ReferenceRouterAdapter(transport)})

    preview = asyncio.run(controller.preview("router-eu"))
    assert preview.drifted and preview.serving_continues
    applied = asyncio.run(controller.reconcile("router-eu"))
    assert applied.status == "IN_SYNC" and applied.verified
    assert asyncio.run(controller.preview("router-eu")).operations == []
    assert source.reports[-1]["health"] == "READY"

    class FailingTransport(MemoryRouterTransport):
        async def apply(self, router, config, plan):
            raise RuntimeError("apply failed")

    failing = FailingTransport()
    failing.states["router-eu"] = dict(transport.states["router-eu"])
    changed = desired_state()
    changed["desired_revision"] = 4
    changed["routes"][0]["public_model"] = "pra/qwen3-32b-v2"
    failed_source = InMemoryRouterSource({"router-eu": changed})
    failed_controller = RouterController(
        failed_source, {"pra-reference": ReferenceRouterAdapter(failing)},
    )
    failed = asyncio.run(failed_controller.reconcile("router-eu"))
    assert failed.status == "FAILED" and not failed.verified
    assert failing.states["router-eu"] == transport.states["router-eu"]
    assert failed_source.reports[-1]["health"] == "DEGRADED"


class FakeBackendClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, dict]] = []
        self.fail: set[str] = set()

    async def request(self, backend, payload, headers, timeout):
        self.calls.append((backend.id, payload, headers))
        if backend.id in self.fail:
            raise RuntimeError("offline")
        return {"id": "response", "choices": [{"message": {"content": backend.id}}]}

    async def stream(self, backend, payload, headers, timeout):
        self.calls.append((backend.id, payload, headers))
        yield b"data: first\n\n"
        yield b"data: [DONE]\n\n"


def reference_config() -> dict:
    return {
        "revision": 4, "max_attempts": 2,
        "routes": [{
            "id": "coder", "public_model": "pra/coder", "strategy": "round-robin",
            "backends": [
                {"id": "a", "url": "http://a", "model": "coder", "weight": 1},
                {"id": "b", "url": "http://b", "model": "coder", "weight": 1},
            ],
        }],
    }


def test_reference_router_openai_failover_stream_and_header_allowlist() -> None:
    backend = FakeBackendClient()
    backend.fail.add("a")
    router = ReferenceRouter(reference_config(), backend)
    result = asyncio.run(router.complete({"model": "pra/coder", "messages": []}, {
        "traceparent": "00-abc", "authorization": "Bearer must-not-pass", "x-pra-session-id": "session",
    }))
    assert result["choices"][0]["message"]["content"] == "b"
    assert [row[0] for row in backend.calls] == ["a", "b"]
    assert "authorization" not in backend.calls[0][2]
    assert backend.calls[0][2]["traceparent"] == "00-abc"
    assert router.metric_snapshot()["counters"]["coder:retries"] == 1

    app = create_reference_router_app(router)
    client = TestClient(app)
    assert client.get("/health").json()["revision"] == 4
    response = client.post("/v1/chat/completions", json={"model": "pra/coder", "messages": [], "stream": True})
    assert response.status_code == 200 and b"[DONE]" in response.content
    assert client.get("/v1/router/routes").json()["items"][0]["public_model"] == "pra/coder"


def test_reference_router_enforces_tenants_before_streaming_and_uses_fallback_last() -> None:
    config = reference_config()
    config["max_attempts"] = 3
    config["routes"][0]["tenant_ids"] = ["tenant-a"]
    config["routes"][0]["backends"].append({
        "id": "fallback", "url": "http://fallback", "model": "coder",
        "weight": 1, "fallback": True,
    })
    backend = FakeBackendClient()
    backend.fail.update({"a", "b"})
    router = ReferenceRouter(config, backend)
    client = TestClient(create_reference_router_app(router))

    denied = client.post(
        "/v1/chat/completions",
        headers={"x-pra-tenant-id": "tenant-b"},
        json={"model": "pra/coder", "messages": [], "stream": True},
    )
    assert denied.status_code == 403
    result = asyncio.run(router.complete(
        {"model": "pra/coder", "messages": []}, {"x-pra-tenant-id": "tenant-a"},
    ))
    assert result["choices"][0]["message"]["content"] == "fallback"
    assert [call[0] for call in backend.calls] == ["a", "b", "fallback"]
    metrics = router.metric_snapshot()
    assert metrics["route_decision_ms"]["coder"]["count"] == 1


def test_atomic_file_transport_keeps_revision_outside_native_config(tmp_path: Path) -> None:
    path = tmp_path / "agentgateway.yaml"
    transport = AtomicFileTransport(path)
    plan = ReconcilePlan(
        router_id="router-eu", router_kind="agentgateway", desired_revision=7,
        observed_revision=0, desired_digest="desired", observed_digest="observed",
    )
    config = {"gateways": {"pra": {"port": 3000}}, "backends": [], "routes": []}
    asyncio.run(transport.apply({"id": "router-eu"}, config, plan))

    assert "praDesiredRevision" not in path.read_text(encoding="utf-8")
    assert asyncio.run(transport.read({"id": "router-eu"})) == {"revision": 7, "config": config}


def test_router_compilers_never_embed_credentials() -> None:
    state = RouterDesiredState.model_validate(desired_state("litellm"))
    output = json.dumps(LiteLLMRouterAdapter(MemoryRouterTransport()).compile(state)).casefold()
    assert "authorization" not in output
    assert "api_key" not in output


def test_bifrost_uses_keyless_custom_provider_and_valid_cel_model_variable() -> None:
    state = RouterDesiredState.model_validate(desired_state("bifrost"))
    output = BifrostRouterAdapter(MemoryRouterTransport()).compile(state)
    provider = next(iter(output["providers"].values()))
    assert provider["custom_provider_config"]["base_provider_type"] == "openai"
    assert provider["custom_provider_config"]["is_key_less"] is True
    assert output["governance"]["routing_rules"][0]["cel_expression"] == 'model == "pra/qwen3-32b"'


def test_router_control_is_exposed_semantically_and_visible_in_ui() -> None:
    assert {item.id for item in OPERATION_CATALOG}.issuperset({
        "router.list", "route.list", "route.plan", "route.apply",
    })
    assert {item.name for item in TOOL_CATALOG}.issuperset({"pra_router", "pra_route"})
    static = Path(__file__).parents[1] / "src/pra_control/static"
    assert 'data-view="routers"' in (static / "index.html").read_text(encoding="utf-8")
    assert "Reconciliation plan" in (static / "app.js").read_text(encoding="utf-8")


def test_router_cli_requires_explicit_confirmation_before_apply() -> None:
    runner = CliRunner()
    help_result = runner.invoke(router_cli, ["--help"])
    assert help_result.exit_code == 0
    assert {"serve", "inspect", "routes", "preview", "reconcile", "controller"}.issubset(
        set(help_result.output.split())
    )
    result = runner.invoke(router_cli, ["reconcile", "router-eu"])
    assert result.exit_code == 2
    assert "pass --confirm after reviewing preview" in result.output
