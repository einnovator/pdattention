from __future__ import annotations

from types import SimpleNamespace

import pytest

from pra_hf.adaptive_context_runtime import AdaptiveContextRuntime, ContextPolicy, TypeContextPolicy
from pra_hf.context_records import RecordType
from pra_hf.context_store import RecordScope
from pra_hf.large_record_index import (
    IndexLifecycleState,
    LargeRecordChannel,
    LargeRecordIndex,
    LargeRecordSearchPolicy,
)
from pra_hf.progressive_context import NativeIndexState, ProgressiveContextRuntime
from pra_hf.typed_context import CompressionBudget, select_payload


class _Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return SimpleNamespace(input_ids=list(range(len(text.split()))))


class _FakePRA:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.references = {}

    def add_reference(self, reference, *, text):
        self.references[reference] = text
        tokens = len(text.split())
        return SimpleNamespace(id=reference, uri=reference, tokens=tokens, chunks=1)

    def remove_reference(self, handle):
        self.references.pop(handle.uri, None)

    def stats(self):
        return {"routing_index_bytes": 0, "resident_detail_kv_bytes": 0}


def _payload():
    return {
        "columns": ["account", "status", "detail"],
        "rows": [
            {"account": "A-1", "status": "normal", "detail": "ordinary"},
            {"account": "B-7", "status": "failed", "detail": "ZX-91 timeout"},
            {"account": "C-3", "status": "normal", "detail": "ordinary"},
        ],
    }


def test_independent_channels_use_rrf_and_exact_selectors():
    index = LargeRecordIndex(_payload())

    result = index.search("Which account had ZX-91 timeout?", top_k=1)

    assert result.hits[0].unit_id == "rows:1"
    assert result.hits[0].selector == {"collection": "rows", "range": [1, 2]}
    assert result.trace.fusion == "rrf"
    assert result.trace.resolved_channels == (
        LargeRecordChannel.TYPED,
        LargeRecordChannel.BM25,
        LargeRecordChannel.EMBEDDING,
    )
    assert all(value > 0 for value in result.trace.index_bytes.values())
    assert select_payload(_payload(), result.hits[0].selector)["rows"][0]["account"] == "B-7"


def test_channel_policy_rejects_unavailable_native_and_traces_explicit_mode():
    index = LargeRecordIndex(_payload())
    with pytest.raises(RuntimeError, match="native_qk"):
        index.search("timeout", policy=LargeRecordSearchPolicy.NATIVE_ONLY)

    result = index.search(
        "timeout",
        policy=LargeRecordSearchPolicy.EXPLICIT,
        channels=[LargeRecordChannel.BM25],
    )
    assert result.trace.resolved_channels == (LargeRecordChannel.BM25,)
    assert result.trace.fusion == "single_channel_rank"


def test_size_gate_leaves_cheap_indexes_and_lazy_selected_native_available(tmp_path):
    runtime = AdaptiveContextRuntime(
        RecordScope("tenant", "session"),
        ContextPolicy(local_store=tmp_path, max_native_index_tokens=1),
    )
    progressive = ProgressiveContextRuntime(runtime, pra_model=_FakePRA())
    record = progressive.ingest(_payload(), record_type=RecordType.DB_RESULT)

    audit = progressive.prepare_native_index(record.record_id)
    gated_lifecycle = progressive.registry.large_record_lifecycle(record.record_id)
    search, regions = progressive.recover_large_record_native(
        record.record_id, "ZX-91 timeout", top_k=1
    )
    lifecycle = progressive.registry.large_record_lifecycle(record.record_id)

    assert audit.native_index_state == NativeIndexState.SKIPPED_SIZE_LIMIT
    assert audit.cheap_index_modes_built == ("typed", "bm25", "embedding")
    assert search.hits[0].unit_id == "rows:1"
    assert regions[0].selector == search.hits[0].selector
    assert lifecycle.typed_index.state == IndexLifecycleState.BUILT
    assert lifecycle.bm25_index.state == IndexLifecycleState.BUILT
    assert lifecycle.embedding_index.state == IndexLifecycleState.BUILT
    assert lifecycle.native_qk_index.state == IndexLifecycleState.SKIPPED_SIZE_LIMIT
    assert gated_lifecycle.detail_kv.state == IndexLifecycleState.DEFERRED
    assert lifecycle.detail_kv.state == IndexLifecycleState.LAZY


def test_type_aware_budget_is_authoritative_and_audited(tmp_path):
    policy = ContextPolicy(
        local_store=tmp_path,
        record_policies={
            RecordType.API_RESULT: TypeContextPolicy(
                unit_limit=8,
                compact_target_tokens=18,
                compact_max_tokens=24,
                compact_ratio_target=0.5,
            )
        },
    )
    runtime = AdaptiveContextRuntime(RecordScope("tenant", "session"), policy)
    record = runtime.ingest(
        {
            "operation": "lookup",
            "status": "ok",
            "schema": {"id": "string", "value": "string"},
            "results": [{"id": index, "value": f"value-{index}"} for index in range(20)],
        },
        record_type=RecordType.API_RESULT,
    )

    assert record.compression_strategy == "tool_api_status_schema_representatives"
    assert record.metadata["compact_target_tokens"] == 18
    assert record.metadata["compact_tokens"] <= 18
    assert record.metadata["original_tokens"] > record.metadata["compact_tokens"]


def test_compression_budget_validation():
    with pytest.raises(ValueError):
        CompressionBudget(compact_ratio_target=1.1)
