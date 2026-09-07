"""Paper 9 lineage, permeability, validity, and native-reuse contracts."""

from __future__ import annotations

from dataclasses import replace

from pra_hf.context_records import ContextRecord, RecordType, serialize_record
from pra_hf.session_service import LocalSessionService
from pra_hf.subagent_context import (
    AgentContextGraph,
    ConsistencyMode,
    ContextVisibilityPolicy,
    CrossAgentReuseRuntime,
    EffectType,
    NativeKVHandle,
    RecordVisibility,
    ResourceIdentity,
    ReuseDecision,
    ReuseMissReason,
    ToolEffectDescriptor,
)
from pra_hf.subagent_harness import DeclarativeTool, SubagentHarness, SubagentSpec


def _read_effect(path: str = "/repo/shared.py", **kwargs) -> ToolEffectDescriptor:
    return ToolEffectDescriptor(
        EffectType.READ,
        (ResourceIdentity("os.FILE", path),),
        reuse_enabled=True,
        **kwargs,
    )


def test_context_record_agent_identity_serializes_and_persists(tmp_path) -> None:
    record = ContextRecord(
        "r1",
        RecordType.GENERIC_TEXT,
        "evidence",
        session_uuid="session-1",
        agent_uuid="agent-1",
        task_uuid="task-1",
        created_at=10.0,
        logical_clock=3,
    )
    assert '"agent_uuid": "agent-1"' in serialize_record(record)

    service = LocalSessionService(tmp_path)
    service.create_session("user", "session-1")
    service.append_record("user", "session-1", record)
    restored = LocalSessionService(tmp_path).get_session("user", "session-1").records[0]
    assert restored.agent_uuid == "agent-1"
    assert restored.logical_clock == 3


def test_agent_graph_tracks_lifecycle_lineage_and_selected_visibility() -> None:
    graph = AgentContextGraph("session")
    root = graph.start_root("root")
    root_record = graph.append_record(
        root.agent_uuid, ContextRecord("shared", RecordType.FILE_READ, "shared definition")
    )
    selected = graph.spawn(
        "root",
        agent_uuid="selected-child",
        context_policy=ContextVisibilityPolicy(
            ancestor=RecordVisibility.SELECTED,
            selected_record_ids=frozenset({root_record.record_id}),
        ),
    )
    hidden = graph.spawn("root", agent_uuid="hidden-child")

    assert graph.lineage(selected.agent_uuid) == ("selected-child", "root")
    assert root_record in graph.visible_records(selected.agent_uuid)
    assert root_record not in graph.visible_records(hidden.agent_uuid)
    assert graph.records("root")[0].record_type == RecordType.AGENT_START


def test_agent_graph_rehydrates_lifecycle_and_record_streams() -> None:
    graph = AgentContextGraph("session")
    graph.start_root("root")
    graph.spawn(
        "root",
        agent_uuid="child",
        context_policy=ContextVisibilityPolicy(ancestor="routable"),
    )
    graph.append_record("child", ContextRecord("child:result", RecordType.GENERIC_TEXT, "done"))
    graph.stop("child", result_record_uuids=("child:result",))

    restored = AgentContextGraph.from_records("session", graph.all_records)
    assert restored.descriptor("child").status.value == "stopped"
    assert restored.descriptor("child").result_record_uuids == ("child:result",)
    assert restored.lineage("child") == ("child", "root")
    assert restored.all_records == graph.all_records


def test_exact_ancestor_result_reuse_avoids_tool_execution() -> None:
    calls = []
    harness = SubagentHarness("session", root_agent_uuid="root")
    tool = DeclarativeTool(
        "tool://read",
        lambda arguments: calls.append(arguments["path"]) or "content",
        lambda arguments: _read_effect(str(arguments["path"])),
    )
    parent = harness.execute_tool("root", tool, {"path": "/repo/shared.py"})
    child = harness.spawn_subagent(
        "root",
        SubagentSpec(context_policy=ContextVisibilityPolicy(ancestor="routable")),
        agent_uuid="child",
    )
    reused = harness.execute_tool(child.agent_uuid, tool, {"path": "/repo/shared.py"})

    assert parent.executed
    assert not reused.executed
    assert reused.output == "content"
    assert reused.reuse.decision == ReuseDecision.HIT
    assert reused.reuse.payload_reused
    assert calls == ["/repo/shared.py"]


def test_unknown_effect_and_hidden_sibling_fail_closed() -> None:
    graph = AgentContextGraph("session")
    graph.start_root("root")
    graph.spawn("root", agent_uuid="left", context_policy=ContextVisibilityPolicy(ancestor="routable"))
    graph.spawn("root", agent_uuid="right", context_policy=ContextVisibilityPolicy(ancestor="routable"))
    runtime = CrossAgentReuseRuntime(graph)
    source = graph.append_record("left", ContextRecord("left:r", RecordType.TOOL_RESPONSE, {}))
    effect = _read_effect()
    runtime.register_result(
        tool_uri="tool://read", arguments={}, source_record=source, effect=effect, payload="x"
    )
    entry, hidden = runtime.lookup(
        request_agent_uuid="right",
        tool_uri="tool://read",
        arguments={},
        requested_effect=effect,
    )
    unknown = ToolEffectDescriptor(EffectType.UNKNOWN)
    _, rejected = runtime.lookup(
        request_agent_uuid="right",
        tool_uri="tool://unknown",
        arguments={},
        requested_effect=unknown,
    )

    assert entry is None and hidden.reason == ReuseMissReason.NOT_VISIBLE
    assert rejected.reason in {ReuseMissReason.UNKNOWN_EFFECT, ReuseMissReason.POLICY_DISABLED}


def test_selective_invalidation_rejects_relevant_write_but_preserves_other_resource() -> None:
    harness = SubagentHarness("session", root_agent_uuid="root")
    read_a = DeclarativeTool("tool://read-a", lambda _: "A0", lambda _: _read_effect("/repo/a.py"))
    read_b = DeclarativeTool("tool://read-b", lambda _: "B0", lambda _: _read_effect("/repo/b.py"))
    harness.execute_tool("root", read_a, {})
    harness.execute_tool("root", read_b, {})
    writer = harness.spawn_subagent(
        "root",
        SubagentSpec(context_policy=ContextVisibilityPolicy(ancestor="routable")),
        agent_uuid="writer",
    )
    write_a = DeclarativeTool(
        "tool://write-a",
        lambda _: "written",
        lambda _: ToolEffectDescriptor(
            EffectType.WRITE, (ResourceIdentity("os.FILE", "/repo/a.py"),)
        ),
    )
    harness.execute_tool(writer.agent_uuid, write_a, {})
    reader = harness.spawn_subagent(
        "root",
        SubagentSpec(context_policy=ContextVisibilityPolicy(ancestor="routable")),
        agent_uuid="reader",
    )
    a = harness.execute_tool(reader.agent_uuid, read_a, {})
    b = harness.execute_tool(reader.agent_uuid, read_b, {})

    assert a.executed and a.reuse.reason == ReuseMissReason.WRITE_INVALIDATION
    assert not b.executed and b.reuse.decision == ReuseDecision.HIT


def test_ancestry_consistency_ignores_sibling_worktree_write() -> None:
    graph = AgentContextGraph("session")
    graph.start_root("root", workspace_id="base")
    graph.spawn("root", agent_uuid="left", workspace_id="left", context_policy=ContextVisibilityPolicy(ancestor="routable"))
    graph.spawn("root", agent_uuid="right", workspace_id="right", context_policy=ContextVisibilityPolicy(ancestor="routable"))
    runtime = CrossAgentReuseRuntime(graph)
    effect = _read_effect(consistency=ConsistencyMode.ANCESTRY)
    source = graph.append_record("root", ContextRecord("root:r", RecordType.FILE_READ, "base"))
    runtime.register_result(tool_uri="tool://read", arguments={}, source_record=source, effect=effect, payload="base")
    runtime.register_write(
        "left",
        ToolEffectDescriptor(
            EffectType.WRITE,
            (ResourceIdentity("os.FILE", "/repo/shared.py", workspace_id="left"),),
        ),
    )
    entry, decision = runtime.lookup(
        request_agent_uuid="right", tool_uri="tool://read", arguments={}, requested_effect=effect
    )
    assert entry is not None and decision.decision == ReuseDecision.HIT


def test_external_resources_require_successful_validation() -> None:
    graph = AgentContextGraph("session")
    graph.start_root("root")
    graph.spawn("root", agent_uuid="child", context_policy=ContextVisibilityPolicy(ancestor="routable"))
    runtime = CrossAgentReuseRuntime(graph)
    effect = _read_effect(consistency=ConsistencyMode.EXTERNAL, version_token="etag-1")
    source = graph.append_record("root", ContextRecord("root:r", RecordType.API_RESULT, "value"))
    runtime.register_result(tool_uri="tool://http", arguments={}, source_record=source, effect=effect, payload="value")

    _, failed = runtime.lookup(
        request_agent_uuid="child", tool_uri="tool://http", arguments={}, requested_effect=effect
    )
    entry, passed = runtime.lookup(
        request_agent_uuid="child",
        tool_uri="tool://http",
        arguments={},
        requested_effect=effect,
        external_validator=lambda resource, version: version == "etag-1",
    )
    assert failed.reason == ReuseMissReason.EXTERNAL_VALIDATION_FAILED
    assert entry is not None and passed.decision == ReuseDecision.HIT


class _NativePort:
    def __init__(self) -> None:
        self.materialized = []

    def encode(self, record: ContextRecord) -> NativeKVHandle:
        return NativeKVHandle(record.record_id, "model", "r1", "e1", "source_local", 128, 4096)

    def materialize(self, handle: NativeKVHandle, *, target_agent_uuid: str) -> bool:
        self.materialized.append((handle.record_uuid, target_agent_uuid))
        return True


def test_native_kv_reuse_requires_exact_encoding_compatibility() -> None:
    port = _NativePort()
    harness = SubagentHarness("session", root_agent_uuid="root", native_port=port)
    tool = DeclarativeTool("tool://read", lambda _: "large result", lambda _: _read_effect())
    parent = harness.execute_tool("root", tool, {})
    harness.spawn_subagent(
        "root",
        SubagentSpec(context_policy=ContextVisibilityPolicy(ancestor="routable")),
        agent_uuid="child",
    )
    requested = NativeKVHandle("request", "model", "r1", "e1", "source_local", 0, 0)
    hit = harness.execute_tool("child", tool, {}, requested_native=requested)
    repeated_hit = harness.execute_tool("child", tool, {}, requested_native=requested)
    incompatible = replace(requested, model_revision="r2")
    fallback = harness.execute_tool("child", tool, {}, requested_native=incompatible)

    assert parent.executed
    assert hit.reuse.kv_reused and hit.reuse.native_tokens_reused == 128
    assert repeated_hit.reuse.kv_reused
    assert not fallback.reuse.kv_reused and fallback.reuse.payload_reused
    assert port.materialized == [(parent.record.record_id, "child")]


def test_parent_can_route_into_completed_but_not_running_descendant() -> None:
    graph = AgentContextGraph("session")
    graph.start_root(
        "root",
        context_policy=ContextVisibilityPolicy(descendant=RecordVisibility.COMPLETED_ONLY),
    )
    graph.spawn("root", agent_uuid="child")
    evidence = graph.append_record(
        "child", ContextRecord("child:evidence", RecordType.TERMINAL_OUTPUT, "test passed")
    )
    assert evidence not in graph.visible_records("root")
    graph.stop("child", result_record_uuids=(evidence.record_id,))
    assert evidence in graph.visible_records("root")


def test_hierarchical_delegation_exposes_root_records_without_context_copy() -> None:
    graph = AgentContextGraph("session")
    graph.start_root("root")
    shared = graph.append_record("root", ContextRecord("root:spec", RecordType.GENERIC_DOCUMENT, "spec"))
    graph.spawn("root", agent_uuid="child", context_policy=ContextVisibilityPolicy(ancestor="routable"))
    graph.spawn("child", agent_uuid="grandchild", context_policy=ContextVisibilityPolicy(ancestor="routable"))

    visible = graph.visible_records("grandchild")
    assert shared in visible
    assert all(row.record_id != shared.record_id for row in graph.records("grandchild"))
