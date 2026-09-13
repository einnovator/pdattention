from __future__ import annotations

import pytest

from pra_hf.agent_history import AgentRecordRole, OpenAIRecordizer
from pra_hf.agent_memory_planner import PortableStateAuthorityPlanner
from pra_hf.deployment import PRAEngineCapabilities, PRAEngineResult, PRAWireRequest
from pra_hf.mediated_gateway import MediatedPRAGateway
from pra_hf.mediation import (
    MediationConflictError,
    PRAMediationConfig,
    RecordInferenceError,
    RequestMediator,
    WireAgentMemoryPlan,
)


def _logical_capabilities() -> PRAEngineCapabilities:
    return PRAEngineCapabilities(
        adapter="test-logical",
        integration_level="E1",
        logical_refs=True,
        typed_records=True,
    )


def _ordinary_capabilities() -> PRAEngineCapabilities:
    return PRAEngineCapabilities(adapter="test-ordinary", integration_level="E0")


def _request(messages, *, metadata=None, resources=()):
    return PRAWireRequest(
        model="model",
        messages=tuple(messages),
        session_id="session-1",
        metadata=metadata or {},
        resources=resources,
    )


def test_openai_recordizer_is_exact_for_standard_tool_call_traffic():
    result = OpenAIRecordizer().recordize([
        {"role": "system", "content": "help"},
        {"role": "user", "content": "inspect the project"},
        {
            "role": "assistant",
            "content": "I will inspect it.",
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "source"},
    ], request_metadata={"session_id": "session-1"})

    assert result.exact is True
    assert result.source == "openai_standard"
    assert len(result.history.turns) == 1
    assert result.history.turns[0].complete is True
    assert result.history.records[-1].primary_role == AgentRecordRole.TOOL_OBSERVATION


def test_standard_tool_schema_can_declare_portable_semantics_without_agent_adapter():
    messages = [
        {"role": "system", "content": "help"},
        {"role": "user", "content": "inspect the project"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call-1", "type": "function",
            "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
        }]},
        {"role": "tool", "tool_call_id": "call-1", "content": "source"},
    ]
    request = PRAWireRequest(
        model="model",
        messages=tuple(messages),
        tools=({
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {"type": "object"},
                "x-pra-semantics": {
                    "category": "filesystem",
                    "operation_kind": "read",
                },
            },
        },),
    )

    prepared = RequestMediator({"location": "embedded"}).prepare(
        request, _logical_capabilities()
    )
    action = prepared.recordization.history.records[2]

    assert action.metadata["operation_kind"] == "read"
    assert action.metadata["tool_category"] == "filesystem"
    assert action.metadata["tool_calls"][0]["name"] == "read_file"


def test_openai_recordizer_does_not_guess_nonstandard_user_observation():
    result = OpenAIRecordizer().recordize([
        {"role": "system", "content": "help"},
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "```custom\nread a.py\n```"},
        {"role": "user", "content": "source"},
    ])

    assert result.exact is False
    assert "assistant_without_typed_tool_call:2" in result.ambiguity_reasons
    assert "untyped_user_after_assistant:3" in result.ambiguity_reasons


def test_explicit_records_make_nonstandard_transport_generic():
    messages = [
        {
            "role": "system",
            "content": "help",
            "metadata": {"pra_record": {
                "record_id": "sys", "turn_id": "system",
                "causal_group_id": "system", "primary_role": "system",
            }},
        },
        {
            "role": "user",
            "content": "fix it",
            "metadata": {"pra_record": {
                "record_id": "task", "turn_id": "task",
                "causal_group_id": "task", "primary_role": "task",
            }},
        },
        {
            "role": "assistant",
            "content": "custom action",
            "metadata": {"pra_record": {
                "record_id": "a1", "turn_id": "t1",
                "causal_group_id": "g1", "primary_role": "assistant_action",
            }},
        },
        {
            "role": "user",
            "content": "custom result",
            "metadata": {"pra_record": {
                "record_id": "o1", "turn_id": "t1",
                "causal_group_id": "g1", "primary_role": "tool_observation",
                "resource_ids": ["repo:a.py"],
            }},
        },
    ]
    result = OpenAIRecordizer().recordize(messages)

    assert result.exact is True
    assert result.source == "typed"
    assert result.history.turns[0].complete is True
    assert result.history.records[-1].resource_ids == ("repo:a.py",)


def test_compact_config_proposal_normalizes_to_explicit_contract():
    config = PRAMediationConfig.from_mapping({
        "location": "embedded",
        "mode": "auto",
        "strict": True,
        "infer_records": True,
        "history_selection": "task-aware-progress-spine-v4",
        "result_compaction": "shadow",
        "tool_disclosure": "shadow",
        "prevent_double_mediation": True,
    })

    assert config.location.value == "embedded"
    assert config.strict_typed_contract is True
    assert config.record_inference.enabled is True
    assert config.history_selection.mode.value == "active"
    assert config.history_selection.policy == "task-aware-progress-spine-v4"
    assert config.result_compaction.mode.value == "shadow"
    assert config.tool_disclosure.mode.value == "shadow"


def test_auto_uses_g01_for_exact_standard_records_and_g00_for_ambiguity():
    mediator = RequestMediator({"location": "embedded", "mode": "auto"})
    exact = _request([
        {"role": "system", "content": "help"},
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "read", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ])
    ambiguous = _request([
        {"role": "system", "content": "help"},
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "custom action"},
        {"role": "user", "content": "custom result"},
    ])

    assert mediator.prepare(exact, _logical_capabilities()).resolved_mode.value == "G01"
    fallback = mediator.prepare(ambiguous, _logical_capabilities())
    assert fallback.resolved_mode.value == "G00"
    assert fallback.trace["recordization_exact"] is False


def test_mediation_stamp_skips_identical_and_rejects_conflicting_policy():
    request = _request([
        {"role": "system", "content": "help"},
        {"role": "user", "content": "do it"},
    ])
    mediator = RequestMediator({"location": "embedded", "mode": "auto"})
    first = mediator.prepare(request, _ordinary_capabilities())
    second = mediator.prepare(first.request, _ordinary_capabilities())

    assert second.skipped_existing is True
    assert second.request == first.request
    conflicting = RequestMediator({
        "location": "external", "mode": "auto",
        "history_selection": "task-aware-progress-spine-v4",
    })
    with pytest.raises(MediationConflictError):
        conflicting.prepare(first.request, _ordinary_capabilities())


class _CaptureAdapter:
    def __init__(self):
        self.requests = []

    def capabilities(self):
        return _logical_capabilities()

    def prepare_session(self, request):
        return request.session_id

    def generate(self, request):
        self.requests.append(request)
        return PRAEngineResult("ok")

    def stream(self, request):
        self.requests.append(request)
        yield {"type": "done", "request_id": request.request_id}

    def close_session(self, session_id):
        return None


def _typed_message(role, content, record_id, turn_id, group_id, primary_role):
    return {
        "role": role,
        "content": content,
        "metadata": {"pra_record": {
            "record_id": record_id,
            "turn_id": turn_id,
            "causal_group_id": group_id,
            "primary_role": primary_role,
        }},
    }


def test_mediated_gateway_realizes_one_frozen_plan_without_agent_logic():
    messages = [
        _typed_message("system", "system", "s", "system", "system", "system"),
        _typed_message("user", "task", "t", "task", "task", "task"),
        _typed_message("assistant", "old action", "a1", "one", "g1", "assistant_action"),
        _typed_message("tool", "large old output", "o1", "one", "g1", "tool_observation"),
        _typed_message("assistant", "current action", "a2", "two", "g2", "assistant_action"),
        _typed_message("tool", "current output", "o2", "two", "g2", "tool_observation"),
    ]
    plan = WireAgentMemoryPlan(
        schema_version=1,
        policy="task-aware-progress-spine-v4",
        selected_record_ids=("s", "t", "a1", "o1", "a2", "o2"),
        record_replacements={"o1": "[PRA memory] prior output superseded"},
    )
    request = _request(messages, metadata={"agent_memory_plan": plan.to_dict()})
    adapter = _CaptureAdapter()
    gateway = MediatedPRAGateway(
        adapter,
        mediation={
            "location": "embedded",
            "mode": "auto",
            "history_selection": "task-aware-progress-spine-v4",
            "result_compaction": "active",
        },
    )

    result = gateway.generate(request)

    assert result.text == "ok"
    assert len(adapter.requests) == 1
    forwarded = adapter.requests[0]
    assert forwarded.messages[3]["content"] == "[PRA memory] prior output superseded"
    assert forwarded.metadata["mediation_stamp"]["plan_digest"] == plan.digest
    assert result.trace[0]["stage"] == "request_mediation"
    assert result.trace[0]["resolved_mode"] == "G11"


def test_shadow_mediation_does_not_change_model_visible_input():
    messages = [
        _typed_message("system", "system", "s", "system", "system", "system"),
        _typed_message("user", "task", "t", "task", "task", "task"),
    ]
    plan = WireAgentMemoryPlan(
        schema_version=1,
        policy="candidate",
        selected_record_ids=("s", "t"),
        record_replacements={"t": "changed only if active"},
    )
    request = _request(messages, metadata={"agent_memory_plan": plan.to_dict()})
    adapter = _CaptureAdapter()
    gateway = MediatedPRAGateway(
        adapter,
        mediation={
            "location": "embedded",
            "mode": "auto",
            "history_selection": {"mode": "shadow", "policy": "candidate"},
            "result_compaction": "shadow",
        },
    )

    gateway.generate(request)

    assert adapter.requests[0].messages == request.messages


def test_external_and_embedded_locations_realize_identical_frozen_plan():
    messages = [
        _typed_message("system", "system", "s", "system", "system", "system"),
        _typed_message("user", "task", "t", "task", "task", "task"),
        _typed_message("assistant", "old action", "a1", "one", "g1", "assistant_action"),
        _typed_message("tool", "old output", "o1", "one", "g1", "tool_observation"),
        _typed_message("assistant", "current action", "a2", "two", "g2", "assistant_action"),
        _typed_message("tool", "current output", "o2", "two", "g2", "tool_observation"),
    ]
    plan = WireAgentMemoryPlan(
        schema_version=1,
        policy="portable-policy",
        selected_record_ids=("s", "t", "a2", "o2"),
    )
    request = _request(messages, metadata={"agent_memory_plan": plan.to_dict()})
    outputs = []
    for location in ("external", "embedded"):
        prepared = RequestMediator({
            "location": location,
            "mode": "auto",
            "history_selection": "portable-policy",
        }).prepare(request, _logical_capabilities())
        outputs.append(prepared.request.messages)

    assert outputs[0] == outputs[1]


def test_stale_frozen_plan_is_rejected_before_model_inference():
    messages = [
        _typed_message("system", "system", "s", "system", "system", "system"),
        _typed_message("user", "task", "t", "task", "task", "task"),
    ]
    plan = WireAgentMemoryPlan(
        schema_version=1,
        policy="portable-policy",
        selected_record_ids=("s", "t"),
        source_history_digest="not-this-history",
    )
    request = _request(messages, metadata={"agent_memory_plan": plan.to_dict()})
    mediator = RequestMediator({
        "location": "embedded",
        "history_selection": "portable-policy",
    })

    with pytest.raises(RecordInferenceError, match="different history"):
        mediator.prepare(request, _logical_capabilities())


def test_portable_planner_uses_declared_metadata_without_tool_syntax():
    def typed(role, content, rid, turn, group, primary, metadata=None):
        message = _typed_message(role, content, rid, turn, group, primary)
        message["metadata"]["pra_record"]["metadata"] = metadata or {}
        return message

    access = [{
        "resource_id": "repo:a.py", "version": "v1", "span_kind": "whole",
    }]
    messages = [
        typed("system", "system", "s", "system", "system", "system"),
        typed("user", "task", "t", "task", "task", "task"),
        typed("assistant", "custom-native-read", "a1", "one", "g1", "assistant_action", {
            "operation_kind": "read", "resource_accesses": access,
        }),
        typed(
            "tool",
            "old evidence " * 40,
            "o1", "one", "g1", "tool_observation",
            {"operation_kind": "read", "resource_accesses": access, "output_complete": True},
        ),
        typed("assistant", "different-native-read", "a2", "two", "g2", "assistant_action", {
            "operation_kind": "read", "resource_accesses": access,
        }),
        typed(
            "tool", "current evidence", "o2", "two", "g2", "tool_observation",
            {"operation_kind": "read", "resource_accesses": access, "output_complete": True},
        ),
    ]
    request = _request(messages)
    planner = PortableStateAuthorityPlanner(lambda text: len(text.split()))
    mediator = RequestMediator(
        {
            "location": "embedded",
            "history_selection": {
                "mode": "active",
                "policy": "task-aware-progress-spine-v4",
                "head_turns": 0,
                "tail_turns": 1,
            },
            "result_compaction": "active",
        },
        plan_builder=planner,
    )

    prepared = mediator.prepare(request, _logical_capabilities())

    assert prepared.trace["agent_memory_plan_source"] == "generated"
    assert prepared.trace["record_replacement_count"] == 1
    assert prepared.request.messages[3]["content"] == (
        "<returncode>unknown</returncode>\n<output></output>"
    )
    assert prepared.request.metadata["agent_memory_plan"]["decision_metadata"][
        "receipt_decisions"
    ][0]["rule"] == "H3"
    assert prepared.request.metadata["agent_memory_plan"]["plan_digest"]
    assert mediator.prepare(prepared.request, _logical_capabilities()).skipped_existing
