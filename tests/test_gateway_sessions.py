from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from pra_hf.deployment import (
    OpenAICompatibleEngineAdapter,
    PRAEngineCapabilities,
    PRAEngineResult,
    PRAWireRequest,
    PRAWireResource,
)
from pra_hf.engine_profiles import EngineProfileRegistry, EngineType, PrefixCacheMode
from pra_hf.gateway import PRAGateway, create_gateway_server
from pra_hf.gateway_session import HistoryMode, ResourceOperation
from pra_hf.session_service import InMemorySessionService


class SessionAdapter:
    def __init__(self, *, pra=False, streaming=False, incremental=True):
        self.requests = []
        self.prepared = []
        self.closed = []
        self._capabilities = PRAEngineCapabilities(
            adapter="session-fixture",
            engine_type="custom",
            integration_level="E1" if pra else "E0",
            prefix_cache_mode="session_state",
            automatic_prefix_cache=not pra,
            session_state=True,
            incremental_messages=incremental,
            resource_delta=pra,
            cache_affinity=True,
            logical_refs=pra,
            native_kv=False,
            text_fallback=True,
            streaming=streaming,
        )

    def capabilities(self):
        return self._capabilities

    def prepare_session(self, request):
        self.prepared.append(request)
        return f"engine:{request.tenant_id}:{request.session_id}:{request.model}"

    def generate(self, request):
        self.requests.append(request)
        return PRAEngineResult("answer", {"prefix_cache_hit": None})

    def stream(self, request):
        self.requests.append(request)
        yield {"type": "delta", "text": "an"}
        yield {"type": "delta", "text": "swer"}
        yield {"type": "done"}

    def close_session(self, session_id):
        self.closed.append(session_id)


def _resource(text="alpha", version="v1"):
    return PRAWireResource(
        "facts", "pra://tenant-a/facts", text=text,
        metadata={"tenant_id": "tenant-a", "version": version},
        authorization_scope="tenant-a",
    )


def _request(*, messages=None, resources=(), history_mode="AUTO", model="offline/model", tenant="tenant-a", session="session-a", metadata=None):
    return PRAWireRequest(
        model=model,
        messages=tuple(messages or ({"role": "user", "content": "question"},)),
        tenant_id=tenant,
        session_id=session,
        resources=tuple(resources),
        history_mode=history_mode,
        metadata=metadata or {},
        allow_text_fallback=True,
    )


def test_engine_type_changes_resolved_prefix_capabilities():
    registry = EngineProfileRegistry.default()
    assert registry.resolve(EngineType.VLLM).default_prefix_cache_mode == PrefixCacheMode.AUTOMATIC_PREFIX_CACHE

    generic = OpenAICompatibleEngineAdapter("http://unused", engine_type="openai_generic").capabilities()
    vllm = OpenAICompatibleEngineAdapter("http://unused", engine_type="vllm").capabilities()
    ollama = OpenAICompatibleEngineAdapter("http://unused", engine_type="ollama").capabilities()
    assert generic.prefix_cache_mode == PrefixCacheMode.UNKNOWN
    assert not generic.session_state
    assert vllm.automatic_prefix_cache
    assert ollama.prefix_cache_mode == PrefixCacheMode.STATELESS


def test_configured_e1_remote_payload_carries_typed_session_and_resource_delta():
    adapter = OpenAICompatibleEngineAdapter(
        "http://unused",
        engine_type="custom",
        pra_level="E1",
        prefix_cache_mode="session_state",
        session_state=True,
        incremental_messages=True,
        resource_delta=True,
        cache_affinity=True,
    )
    gateway = PRAGateway(adapter, mode="G11")
    request = _request(
        messages=({"role": "user", "content": "delta"},),
        resources=(_resource(),),
        history_mode="DELTA",
    )
    resolved, _, _, _, _, _, _ = gateway._resolve_request(request)
    payload = adapter._payload(resolved)

    assert payload["pra"]["history_mode"] == "DELTA"
    assert payload["pra"]["engine_session_id"] == "session-a"
    assert payload["pra"]["cache_affinity_key"].startswith("pra-affinity:")
    assert payload["pra"]["resource_ops"][0]["operation"] == "ADD"
    assert payload["pra"]["resources"][0]["resource_id"] == "facts"


def test_prepare_is_idempotent_reuse_is_scoped_and_close_is_explicit():
    adapter = SessionAdapter()
    durable = InMemorySessionService()
    gateway = PRAGateway(adapter, mode="G00", session_service=durable)

    first = gateway.generate(_request())
    second = gateway.generate(_request(
        messages=(
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "follow up"},
        )
    ))

    assert len(adapter.prepared) == 1
    assert adapter.requests[1].history_mode == HistoryMode.DELTA
    assert adapter.requests[1].messages == ({"role": "user", "content": "follow up"},)
    assert second.trace[2]["gateway_prefix_stable"] is True
    assert second.trace[2]["engine_session_reuse"] is True
    assert second.trace[2]["engine_prefix_cache_hit"] is None
    assert durable.get_session("gateway", "session-a").tenant_id == "tenant-a"

    assert gateway.close_session("tenant-a", "session-a", "offline/model")
    assert adapter.closed == ["engine:tenant-a:session-a:offline/model"]
    assert not gateway.close_session("tenant-a", "session-a", "offline/model")


def test_full_delta_auto_and_prefix_preserving_g10_keep_prior_messages_identical():
    adapter = SessionAdapter()
    gateway = PRAGateway(adapter, mode="G10")
    gateway.generate(_request(resources=(_resource(),)))
    history = (
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "follow up"},
    )
    result = gateway.generate(_request(messages=history, resources=(_resource("beta", "v2"),)))

    transformed = adapter.requests[-1]
    assert transformed.history_mode == HistoryMode.DELTA
    assert len(transformed.messages) == 1
    assert "PRA text fallback context" in transformed.messages[0]["content"]
    assert transformed.messages[0]["content"].endswith("follow up")
    assert result.trace[2]["gateway_prefix_stable"] is True
    assert result.trace[2]["prefix_reuse_fraction"] == pytest.approx(0.5)

    delta = gateway.generate(_request(
        messages=({"role": "user", "content": "delta turn"},),
        resources=(_resource(),),
        history_mode="DELTA",
    ))
    assert delta.trace[2]["history_mode"] == "DELTA"


def test_full_transport_replays_the_exact_prior_g10_serialization():
    adapter = SessionAdapter(incremental=False)
    gateway = PRAGateway(adapter, mode="G10")
    gateway.generate(_request(resources=(_resource(),)))
    prior = adapter.requests[-1].messages + ({"role": "assistant", "content": "answer"},)
    gateway.generate(_request(
        messages=(
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "next"},
        ),
        resources=(_resource("new evidence", "v2"),),
    ))
    current = adapter.requests[-1]
    assert current.history_mode == HistoryMode.FULL
    assert current.messages[: len(prior)] == prior


def test_pra_session_derives_add_unchanged_update_and_remove_without_prefix_invalidation():
    adapter = SessionAdapter(pra=True)
    gateway = PRAGateway(adapter, mode="G11")
    gateway.generate(_request(resources=(_resource(),)))
    assert [row.operation for row in adapter.requests[-1].resource_ops] == [ResourceOperation.ADD]

    gateway.generate(_request(messages=({"role": "user", "content": "two"},), resources=(_resource(),), history_mode="DELTA"))
    assert [row.operation for row in adapter.requests[-1].resource_ops] == [ResourceOperation.UNCHANGED]
    assert adapter.requests[-1].resources == ()

    update = gateway.generate(_request(messages=({"role": "user", "content": "three"},), resources=(_resource("beta", "v2"),), history_mode="DELTA"))
    assert [row.operation for row in adapter.requests[-1].resource_ops] == [ResourceOperation.UPDATE]
    assert update.trace[2]["prefix_invalidations"] == 0

    gateway.generate(_request(messages=({"role": "user", "content": "four"},), resources=(), history_mode="DELTA"))
    assert [row.operation for row in adapter.requests[-1].resource_ops] == [ResourceOperation.REMOVE]


def test_rewrite_and_template_change_invalidate_but_tenant_model_keys_never_cross():
    adapter = SessionAdapter()
    gateway = PRAGateway(adapter, mode="G00")
    first = gateway.generate(_request(metadata={"chat_template_digest": "qwen-v1"}))
    affinity = first.trace[2]["cache_affinity_key"]

    rewritten = gateway.generate(_request(
        messages=({"role": "system", "content": "rewritten"}, {"role": "user", "content": "q"}),
        metadata={"chat_template_digest": "qwen-v2"},
    ))
    assert rewritten.trace[2]["prefix_invalidations"] == 1
    assert adapter.closed
    assert len(adapter.prepared) == 2

    other_tenant = gateway.generate(_request(tenant="tenant-b", session="session-a"))
    other_model = gateway.generate(_request(model="other/model"))
    assert other_tenant.trace[2]["cache_affinity_key"] != affinity
    assert other_model.trace[2]["cache_affinity_key"] != affinity


@pytest.mark.parametrize(
    "family,messages",
    (
        ("qwen", ({"role": "system", "content": "policy"}, {"role": "user", "content": "q"})),
        ("llama", ({"role": "system", "content": "policy"}, {"role": "user", "content": "q"})),
        ("gemma", ({"role": "user", "content": "q"},)),
        ("openai", ({"role": "developer", "content": "policy"}, {"role": "user", "content": "q"})),
    ),
)
def test_default_fallback_preserves_template_role_sequence(family, messages):
    adapter = SessionAdapter()
    gateway = PRAGateway(adapter, mode="G10")
    gateway.generate(_request(messages=messages, resources=(_resource(),)))
    transformed = adapter.requests[-1].messages
    assert [row["role"] for row in transformed] == [row["role"] for row in messages]
    assert transformed[-1]["role"] == "user"
    assert transformed[-1]["content"].endswith("q")


def test_stream_completion_commits_session_and_cancellation_does_not_close_it():
    adapter = SessionAdapter(streaming=True)
    gateway = PRAGateway(adapter, mode="G00")
    assert list(gateway.stream(_request()))[-1]["trace"]["engine_session_present"]
    iterator = gateway.stream(_request(messages=({"role": "user", "content": "next"},), history_mode="DELTA"))
    next(iterator)
    iterator.close()
    assert adapter.closed == []
    assert gateway.inspect_session("tenant-a", "session-a", "offline/model") is not None


def test_capability_and_session_debug_http_endpoints_are_non_sensitive():
    adapter = SessionAdapter()
    gateway = PRAGateway(adapter, mode="G00")
    gateway.generate(_request(resources=(_resource(),)))
    server = create_gateway_server(gateway, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urllib.request.urlopen(f"{base}/v1/pra/capabilities") as response:
            caps = json.loads(response.read())
        assert caps["engine"]["engine_type"] == "custom"
        assert caps["engine"]["session_state"] is True
        url = f"{base}/v1/pra/sessions/session-a?tenant_id=tenant-a&model=offline%2Fmodel"
        with urllib.request.urlopen(url) as response:
            state = json.loads(response.read())
        assert state["engine_type"] == "custom"
        assert state["known_resources"] == {"facts": "v1"}
        assert "alpha" not in json.dumps(state)
        request = urllib.request.Request(url, method="DELETE")
        with urllib.request.urlopen(request) as response:
            assert json.loads(response.read())["closed"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
