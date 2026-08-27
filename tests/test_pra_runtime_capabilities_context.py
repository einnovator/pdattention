from __future__ import annotations

import pytest

from pra_hf import (
    AgentConfig,
    CapabilitySDK,
    ContextPolicy,
    CursorAction,
    ExecutionAuthorization,
    HuggingFaceBackend,
    PRARuntime,
    PRARuntimeConfig,
    RecordCapabilities,
    RecordType,
    RecordViewName,
    SafeToolExecutor,
    Skill,
    TypeContextPolicy,
)


def lookup_incidents(service: str) -> dict[str, object]:
    """Return recent incidents for one service."""

    return {"service": service}


class _Backend:
    name = "test"

    def add_reference(self, reference, *, text=None, uri=None):
        return {"reference": reference, "text": text, "uri": uri}

    def generate(self, prompt, **kwargs):
        return prompt

    def inspect(self):
        return {"backend": self.name}


def _sdk() -> CapabilitySDK:
    return CapabilitySDK(AgentConfig(
        tools=(lookup_incidents,),
        skills=(Skill(
            name="incident_triage",
            description="Prioritize operational incidents.",
            when_to_use="Use when service health degrades.",
            instructions="Inspect evidence, assess impact, and assign the next safe action.",
            namespace="runtime-test",
            tenant_id="tenant-a",
        ),),
        namespace="runtime-test",
        tenant_id="tenant-a",
        max_candidates=4,
        selection_view_token_budget=256,
    ))


def _runtime(tmp_path) -> PRARuntime:
    sdk = _sdk()
    tool_resource = sdk.tools[0].to_agent_resource()
    executor = SafeToolExecutor(
        (tool_resource,),
        {
            tool_resource.uri: lambda arguments, _observations: {
                "columns": ["service", "status", "latency_ms"],
                "rows": [
                    {"service": arguments["service"], "status": "ok", "latency_ms": 12},
                    {"service": arguments["service"], "status": "failed", "latency_ms": 950},
                    {"service": arguments["service"], "status": "ok", "latency_ms": 18},
                ],
            }
        },
    )
    return PRARuntime(
        config=PRARuntimeConfig(),
        backend=_Backend(),
        capability_sdk=sdk,
        executor=executor,
        context_policy=ContextPolicy(
            local_store=tmp_path,
            persistent_store=False,
            record_policies={
                RecordType.TOOL_RESPONSE: TypeContextPolicy(unit_limit=2),
            },
        ),
    )


def test_runtime_unifies_lazy_capabilities_and_compact_tool_results(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    session = runtime.open_session(
        session_id="session-a", user_id="user-a", tenant_id="tenant-a"
    )
    resources = runtime.capability_resources()
    record_ids = tuple(resource.uri for resource in resources)

    assert {resource.kind for resource in resources} == {"tool", "skill"}
    assert all(resource.metadata["discovery_view"] == "selection" for resource in resources)
    assert "Inspect evidence" not in "\n".join(resource.description for resource in resources)

    palette = runtime.activate_capability_candidates(record_ids)
    skill_id = next(resource.uri for resource in resources if resource.kind == "skill")
    selected_skill = runtime.activate_capability(skill_id)
    assert set(palette.admitted_record_ids) == set(record_ids)
    assert selected_skill.record_id == skill_id
    assert selected_skill.semantic_rediscovery_calls == 0

    tool = next(resource for resource in resources if resource.kind == "tool")
    outcome = runtime.execute_tool_and_record(
        '<tool_call>{"name":"lookup_incidents","arguments":{"service":"billing"}}</tool_call>',
        session=session,
        selected_uris=(tool.uri,),
        authorization=ExecutionAuthorization(frozenset((tool.uri,))),
        call_id="call-1",
        capabilities=RecordCapabilities(
            searchable=True,
            partial_selectors=("rows", "fields"),
        ),
    )

    assert outcome.execution.executed
    assert outcome.record is not None
    compact = runtime.compact_result(session, outcome.record.record_id)
    assert compact["row_count"] == 3
    assert len(compact["representative_rows"]) == 2
    assert runtime.search_results(session, "failed latency")[0].record_id == outcome.record.record_id

    selected = runtime.materialize_result(
        session,
        outcome.record.record_id,
        level=RecordViewName.SELECTED,
        selector={"rows": [1, 2]},
    )
    assert selected.payload["rows"][0]["status"] == "failed"

    cursor = runtime.open_result_cursor(
        session, outcome.record.record_id, collection="rows"
    )
    page = runtime.execute_result_cursor(session, CursorAction(cursor.cursor_id, "next"))
    assert page.success
    assert len(page.payload.items) == 3

    snapshot = runtime.inspect()
    assert snapshot["capabilities"]["tools"] == 1
    assert snapshot["capabilities"]["skills"] == 1
    assert snapshot["result_contexts"][session.session_id]["accounting"]["records"] == 1


def test_result_records_are_session_scoped_and_ephemeral_on_close(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    first = runtime.open_session(
        session_id="first", user_id="user-a", tenant_id="tenant-a"
    )
    record = runtime.ingest_result(
        first,
        {"secret": "first-session-only"},
        record_type=RecordType.API_RESULT,
    )
    second = runtime.open_session(
        session_id="second", user_id="user-a", tenant_id="tenant-a"
    )

    with pytest.raises(KeyError):
        runtime.materialize_result(second, record.record_id)
    with pytest.raises(RuntimeError, match="disabled"):
        runtime.register_result_backing(first, record.record_id)

    runtime.close_session(first)

    assert first.closed
    assert "first" not in runtime.result_contexts
    assert runtime.context_for(second).runtime.store.stats().records == 0
    with pytest.raises(ValueError, match="open runtime session"):
        runtime.context_for(first)
    runtime.close_session(second)


def test_native_result_routing_rejects_non_huggingface_backend() -> None:
    with pytest.raises(ValueError, match="Hugging Face"):
        PRARuntime(
            config=PRARuntimeConfig(),
            backend=_Backend(),
            native_result_routing=True,
        )


def test_native_result_routing_is_explicit_and_cleaned_up(tmp_path) -> None:
    class _Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return type("Encoded", (), {"input_ids": list(range(len(text.split())))})()

        def decode(self, values, skip_special_tokens=True):
            return " ".join(f"token-{value}" for value in values)

    class _PRA:
        def __init__(self):
            self.tokenizer = _Tokenizer()
            self.references = {}

        def add_reference(self, reference, *, text):
            self.references[reference] = text
            return type("Handle", (), {
                "id": reference,
                "uri": reference,
                "tokens": len(text.split()),
                "chunks": 1,
            })()

        def remove_reference(self, handle):
            self.references.pop(handle.uri, None)

        def stats(self):
            return {"routing_index_bytes": 64, "resident_detail_kv_bytes": 256}

        def route(self, _query):
            uri = next(uri for uri in self.references if uri.endswith("/views/backing"))
            return type("Routing", (), {
                "selected": ({
                    "reference_uri": uri,
                    "chunk_id": f"{uri}#0",
                    "logical_start": 0,
                    "logical_end": 3,
                },),
                "stats": {"selection_policy": "top_k"},
            })()

    model = _PRA()
    runtime = PRARuntime(
        config=PRARuntimeConfig(),
        backend=HuggingFaceBackend(model),
        context_policy=ContextPolicy(local_store=tmp_path),
        native_result_routing=True,
    )
    session = runtime.open_session(
        session_id="native", user_id="user-a", tenant_id="tenant-a"
    )
    record = runtime.ingest_result(
        session,
        "alpha beta exact evidence omega",
        record_type=RecordType.GENERIC_TEXT,
    )

    assert not model.references
    handle = runtime.register_result_backing(session, record.record_id)
    selection = runtime.route_result_backing(session, "exact evidence")
    materialized = runtime.materialize_routed_result(session, selection)

    assert handle.uri.endswith("/views/backing")
    assert selection.record_ids == (record.record_id,)
    assert materialized.success
    runtime.close_session(session)
    assert not model.references
