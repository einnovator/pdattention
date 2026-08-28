"""Contract tests for the unified Paper 4.5 runtime SDK."""

from __future__ import annotations

import json

import pytest
import torch
from click.testing import CliRunner

from data.agent_workflows import realistic_tool_catalog, workflow_executor, workflow_tasks
from pra_hf import (
    ContextPolicy,
    DiscoveryRequest,
    ExecutionAuthorization,
    HuggingFaceBackend,
    InMemorySessionService,
    KVInterval,
    KVMaterializer,
    MaterializationPlan,
    NativeKV,
    PackedNativeKVStore,
    PersistentResourceIndex,
    PRARuntime,
    PRARuntimeConfig,
    ResourceDiscoveryEngine,
    RuntimeKVCache,
    RuntimeKVCacheKey,
    RuntimeProfiler,
    TypeContextPolicy,
    TaskOperation,
    VLLMThinBackend,
)
from pra_hf.context_records import RecordType
from pra_hf.cli import cli


def _kv(tokens: int, *, offset: float = 0.0) -> NativeKV:
    key = torch.arange(tokens * 4, dtype=torch.float32).reshape(1, 2, tokens, 2) + offset
    return NativeKV(key, key + 1000)


def test_runtime_config_round_trip_preserves_model_and_system_fields(tmp_path):
    config = PRARuntimeConfig(
        pra={"consumption_layers": [-1], "top_k": 3, "selected_fraction": None},
        kv_layout="block_major",
        page_tokens=32,
        cache_max_bytes=4096,
        context_policy=ContextPolicy(
            max_native_index_tokens=2048,
            max_native_index_bytes=32_768,
            record_policies={
                RecordType.DB_RESULT: TypeContextPolicy(
                    max_native_index_tokens=512,
                    max_native_index_bytes=8192,
                )
            },
        ),
    )
    config.save_pretrained(tmp_path)
    restored = PRARuntimeConfig.from_pretrained(tmp_path)

    assert restored == config
    serialized = json.loads((tmp_path / "pra_runtime_config.json").read_text())
    assert serialized["schema_version"] == 1
    assert serialized["context_policy"]["max_native_index_tokens"] == 2048
    assert serialized["context_policy"]["record_policies"]["db_result"] == {
        "defer_native_index": None,
        "max_native_index_bytes": 8192,
        "max_native_index_tokens": 512,
        "override_native_index_limits": False,
        "storage": None,
        "unit_limit": 8,
    }


def test_runtime_init_cli_persists_native_ingestion_budget(tmp_path):
    target = tmp_path / "runtime"
    result = CliRunner().invoke(
        cli,
        [
            "runtime",
            "init",
            str(target),
            "--max-native-index-tokens",
            "1024",
            "--max-native-index-bytes",
            "16384",
            "--defer-native-index",
        ],
    )

    assert result.exit_code == 0, result.output
    policy = PRARuntimeConfig.from_pretrained(target).context_policy
    assert policy.max_native_index_tokens == 1024
    assert policy.max_native_index_bytes == 16384
    assert policy.defer_native_index is True


def test_materialization_plan_deduplicates_overlaps_before_budgeting():
    plan = MaterializationPlan.build(
        (
            KVInterval("memory://a", 3, 0, 8),
            KVInterval("memory://a", 3, 4, 12),
            KVInterval("memory://b", 3, 0, 8),
        ),
        max_tokens=16,
    )

    assert plan.requested_tokens == 24
    assert plan.unique_tokens == 16
    assert plan.dropped_tokens == 8
    assert plan.intervals == (
        KVInterval("memory://a", 3, 0, 12),
        KVInterval("memory://b", 3, 0, 4),
    )


def test_materializer_preserves_native_kv_heads_and_packs_per_layer():
    sources = {
        ("memory://a", 2): _kv(8),
        ("memory://b", 2): _kv(8, offset=100),
        ("memory://a", 3): _kv(8, offset=200),
    }
    plan = MaterializationPlan.build(
        (
            KVInterval("memory://a", 2, 2, 6),
            KVInterval("memory://b", 2, 0, 2),
            KVInterval("memory://a", 3, 1, 4),
        ),
        max_tokens=16,
    )
    result = KVMaterializer(layout="layer_major").materialize(sources, plan)

    assert result.layers[2].key.shape == (1, 2, 6, 2)
    assert result.layers[3].key.shape == (1, 2, 3, 2)
    assert result.logical_tokens == 9
    assert result.physical_bytes == sum(memory.nbytes for memory in result.layers.values())
    assert result.transfer_bytes == 0


def test_all_packed_layouts_restore_identical_logical_kv():
    sources = {
        ("memory://a", 2): _kv(9),
        ("memory://b", 2): _kv(9, offset=100),
        ("memory://a", 3): _kv(9, offset=200),
        ("memory://b", 3): _kv(9, offset=300),
    }
    plan = MaterializationPlan.build(
        (
            KVInterval("memory://a", 2, 1, 8),
            KVInterval("memory://b", 2, 3, 9),
            KVInterval("memory://a", 3, 0, 5),
            KVInterval("memory://b", 3, 4, 9),
        ),
        max_tokens=32,
    )
    reference = KVMaterializer().materialize(sources, plan)
    for layout in ("layer_major", "reference_major", "chunk_major", "block_major"):
        store = PackedNativeKVStore(sources, layout=layout, page_tokens=4)
        result = KVMaterializer(layout=layout).materialize(store, plan)
        assert result.logical_tokens == reference.logical_tokens
        assert store.nbytes == sum(memory.nbytes for memory in sources.values())
        assert store.index_bytes > 0
        for layer_id in reference.layers:
            assert torch.equal(result.layers[layer_id].key, reference.layers[layer_id].key)
            assert torch.equal(result.layers[layer_id].value, reference.layers[layer_id].value)


def test_lru_cache_accounts_reuse_and_evicts_by_physical_bytes():
    cache = RuntimeKVCache(max_bytes=10, max_entries=3)
    cache.put("a", "A", nbytes=6)
    assert cache.get("a") == "A"
    cache.put("b", "B", nbytes=6)

    assert cache.get("a") is None
    snapshot = cache.snapshot()
    assert snapshot["evictions"] == 1
    assert snapshot["bytes_reused"] == 6
    assert snapshot["resident_bytes"] == 6


def test_native_cache_keys_isolate_tenants_and_per_tenant_eviction() -> None:
    cache = RuntimeKVCache(max_bytes=32, max_entries=8, max_bytes_per_tenant=10)
    a1 = RuntimeKVCacheKey("tenant-a", "user", "session-a", "record-1", 3)
    a2 = RuntimeKVCacheKey("tenant-a", "user", "session-a", "record-2", 3)
    b1 = RuntimeKVCacheKey("tenant-b", "user", "session-b", "record-1", 3)
    cache.put(a1, "a1", nbytes=6)
    cache.put(b1, "b1", nbytes=6)
    cache.put(a2, "a2", nbytes=6)

    assert cache.get(a1) is None
    assert cache.get(a2) == "a2"
    assert cache.get(b1) == "b1"
    assert cache.snapshot()["tenant_resident_bytes"] == {
        "tenant-a": 6,
        "tenant-b": 6,
    }
    cache.clear_scope(tenant_id="tenant-a", session_id="session-a")
    assert cache.get(a2) is None
    assert cache.get(b1) == "b1"


def test_profiler_emits_stage_level_time_bytes_and_metadata():
    profiler = RuntimeProfiler(device="cpu")
    with profiler.stage("gather", input_bytes=64, metadata={"layout": "layer_major"}) as row:
        row["output_bytes"] = 32
    snapshot = profiler.snapshot()

    assert snapshot["total_seconds"] >= 0
    assert snapshot["events"][0]["input_bytes"] == 64
    assert snapshot["events"][0]["output_bytes"] == 32
    assert snapshot["events"][0]["metadata"]["layout"] == "layer_major"


def test_vllm_thin_boundary_does_not_export_semantic_scores_to_scheduler():
    backend = VLLMThinBackend()
    request = backend.prepare(
        "question",
        selected_uris=("memory://a",),
        materialized_tokens=16,
        metadata={"kv_layout": "block_major"},
    )

    assert request.selected_uris == ("memory://a",)
    assert "score" not in repr(request)
    assert backend.inspect()["scheduler_semantically_aware"] is False
    with pytest.raises(RuntimeError, match="not installed"):
        backend.generate("question")


class _FakeModelBackend:
    name = "fake"

    def __init__(self):
        self.references = []

    def add_reference(self, reference, *, text=None, uri=None):
        self.references.append((reference, text, uri))
        return {"reference": reference}

    def generate(self, prompt, **kwargs):
        return f"generated:{prompt}"

    def inspect(self):
        return {"references": len(self.references)}


def test_unified_facade_discovers_and_executes_with_separate_authority():
    resources = realistic_tool_catalog()
    task = workflow_tasks()[0]
    discovery = ResourceDiscoveryEngine(
        PersistentResourceIndex(resources),
        select_threshold=0.0,
        ask_threshold=0.0,
        margin_threshold=0.0,
    )
    runtime = PRARuntime(
        config=PRARuntimeConfig(),
        backend=_FakeModelBackend(),
        discovery=discovery,
        executor=workflow_executor(resources, task),
    )
    trace = runtime.discover_resources(
        DiscoveryRequest(query="search documents", tenant_id="paper6_5", top_k=1)
    )
    search = next(resource for resource in resources if resource.name == "search_document")
    generated = '<tool_call>{"name":"search_document","arguments":{"title":"quarterly"}}</tool_call>'
    denied = runtime.execute_tool(
        generated,
        selected_uris=(search.uri,),
        authorization=ExecutionAuthorization(frozenset()),
        call_id="denied",
    )
    accepted = runtime.execute_tool(
        generated,
        selected_uris=(search.uri,),
        authorization=ExecutionAuthorization(frozenset((search.uri,))),
        call_id="accepted",
    )

    assert trace.selected_uris
    assert denied.reason == "tool_not_authorized"
    assert accepted.executed
    assert runtime.inspect()["safe_executor_installed"] is True


def test_runtime_validates_model_task_operations_through_durable_session_graph():
    runtime = PRARuntime(
        config=PRARuntimeConfig(),
        backend=_FakeModelBackend(),
        session_service=InMemorySessionService(),
    )
    session = runtime.open_session(
        session_id="tasks", user_id="user", tenant_id="tenant"
    )

    state = runtime.apply_task_operations(session, (
        TaskOperation("create", "collect", "Collect evidence"),
        TaskOperation("create", "write", "Write answer", depends_on=("collect",)),
        TaskOperation("activate", "collect"),
    ))

    assert state.active_task_id == "collect"
    write = next(task for task in state.tasks.tasks if task.task_id == "write")
    assert write.depends_on == ("collect",)
    assert state.tasks.last_sequence == 3


def test_runtime_native_geometry_methods_delegate_or_fail_explicitly():
    runtime = PRARuntime(config=PRARuntimeConfig(), backend=_FakeModelBackend())

    with pytest.raises(RuntimeError, match="frozen routing"):
        runtime.route("question")
    with pytest.raises(RuntimeError, match="freeze native"):
        runtime.freeze_native_selection(({"chunk_id": "c"},))
