from __future__ import annotations

from types import SimpleNamespace

import pytest

from pra_hf.hf_live_kv import select_dynamic_cache
from pra_hf.live_history import (
    LiveKVInterval,
    LiveKVSelectionPlan,
    LiveKVSourceRegistry,
)
from experiments.paper4_5_agent.sparse_gate_common import sparse_causal_plan
from experiments.paper4_5_agent.run_vllm_live_agent_kv_gate import (
    _qualify_packaged_runtime,
)


class _ChatTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        text = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(value) for value in text] if tokenize else text


def test_live_plan_keeps_selected_width_and_position_extent_separate() -> None:
    plan = LiveKVSelectionPlan.create(
        100,
        (
            LiveKVInterval(0, 12, "system", "system"),
            LiveKVInterval(40, 55, "turn-4", "turn-4"),
            LiveKVInterval(90, 100, "active", "active"),
        ),
    )
    assert plan.selected_tokens == 37
    assert plan.source_position_base == 100
    assert plan.has_holes
    assert not plan.full_retention


def test_live_plan_rejects_overlap_and_duplicate_record_identity() -> None:
    with pytest.raises(ValueError, match="ordered and disjoint"):
        LiveKVSelectionPlan.create(20, ((0, 10), (9, 15)))
    with pytest.raises(ValueError, match="stable record"):
        LiveKVSelectionPlan.create(
            20,
            (
                LiveKVInterval(0, 5, "same", "g"),
                LiveKVInterval(8, 12, "same", "g"),
            ),
        )


def test_hf_full_selection_is_a_view_and_sparse_selection_reports_pack_copy() -> None:
    torch = pytest.importorskip("torch")
    keys = torch.arange(24, dtype=torch.float32).reshape(1, 1, 6, 4)
    values = keys + 100
    source = SimpleNamespace(layers=[SimpleNamespace(keys=keys, values=values)])

    full = select_dynamic_cache(source, LiveKVSelectionPlan.full(6))
    assert not full.physical_kv_copy
    assert full.selected_text_reencoded_tokens == 0
    assert full.cache.layers[0].keys.data_ptr() == keys.data_ptr()

    sparse = select_dynamic_cache(
        source,
        LiveKVSelectionPlan.create(6, ((0, 2), (4, 6))),
    )
    assert sparse.physical_kv_copy
    assert sparse.cache.layers[0].keys.shape[-2] == 4
    assert sparse.cache.layers[0].keys[0, 0, :, 0].tolist() == [0, 4, 16, 20]


def test_live_kv_registry_isolates_concurrent_sessions_and_rejects_stale_forks() -> None:
    registry = LiveKVSourceRegistry[bytes]()
    registry.register(
        "source-a", b"A", tenant_id="tenant", session_id="session-a", generation=3
    )
    registry.register(
        "source-b", b"B", tenant_id="tenant", session_id="session-b", generation=7
    )

    assert registry.borrow(
        "request-a",
        ("source-a",),
        tenant_id="tenant",
        session_id="session-a",
        expected_generations=(3,),
    ) == (b"A",)
    assert registry.borrow(
        "request-b",
        ("source-b",),
        tenant_id="tenant",
        session_id="session-b",
        expected_generations=(7,),
    ) == (b"B",)
    assert registry.view("source-a").active_request_ids == ("request-a",)
    assert registry.view("source-b").active_request_ids == ("request-b",)

    with pytest.raises(RuntimeError, match="scope"):
        registry.borrow(
            "cross-session",
            ("source-a",),
            tenant_id="tenant",
            session_id="session-b",
            expected_generations=(3,),
        )
    with pytest.raises(RuntimeError, match="Stale"):
        registry.borrow(
            "stale",
            ("source-a",),
            tenant_id="tenant",
            session_id="session-a",
            expected_generations=(2,),
        )

    assert registry.cancel("request-a")
    assert registry.view("source-a").active_request_ids == ()
    assert registry.view("source-b").active_request_ids == ("request-b",)


def test_live_kv_registry_offload_restore_and_termination_tombstone() -> None:
    registry = LiveKVSourceRegistry[bytes](
        dump=lambda value: b"disk:" + value,
        load=lambda value: bytes(value).removeprefix(b"disk:"),
    )
    registry.register(
        "source", b"resident", tenant_id="tenant", session_id="session", generation=1
    )
    registry.borrow(
        "active",
        ("source",),
        tenant_id="tenant",
        session_id="session",
        expected_generations=(1,),
    )
    with pytest.raises(RuntimeError, match="while requests borrow"):
        registry.offload("source")

    registry.release("active")
    assert registry.offload("source") == b"disk:resident"
    assert registry.view("source").tier == "offloaded"
    assert registry.borrow(
        "restored",
        ("source",),
        tenant_id="tenant",
        session_id="session",
        expected_generations=(1,),
    ) == (b"resident",)
    assert registry.view("source").tier == "hot"

    assert registry.terminate_session("tenant", "session") == 1
    assert registry.view("source") is None
    with pytest.raises(RuntimeError, match="terminated"):
        registry.register(
            "replacement",
            b"new",
            tenant_id="tenant",
            session_id="session",
            generation=2,
        )


def test_sparse_gate_drops_whole_old_causal_group_and_keeps_original_extent() -> None:
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "find"},
        {"role": "user", "content": "path"},
        {"role": "assistant", "content": "read"},
        {"role": "user", "content": "source"},
        {"role": "assistant", "content": "test"},
        {"role": "user", "content": "failure"},
        {"role": "assistant", "content": "edit"},
        {"role": "user", "content": "diff"},
    ]
    tokenizer = _ChatTokenizer()
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    source_tokens = len(prompt) - len("<assistant>")
    plan = sparse_causal_plan(
        tokenizer,
        messages,
        prompt,
        source_tokens=source_tokens,
        retention_fraction=0.75,
    )

    assert plan.has_holes
    assert plan.selected_tokens / source_tokens >= 0.75
    assert plan.source_position_base == source_tokens
    groups = {row.causal_group_id for row in plan.intervals}
    assert "preamble" in groups
    assert "turn:6" in groups
    assert "turn:8" in groups
    assert "turn:2" not in groups


def test_vllm_packaged_runtime_qualification_is_derived_fail_closed() -> None:
    provenance = {
        "inside_active_environment": True,
        "native_files": [{"path": "native.so", "sha256": "abc", "bytes": 1}],
    }

    qualified, failures = _qualify_packaged_runtime(
        vllm_version="0.29.0+cpu",
        metal_version="0.29.0",
        source_revision="7390805822b2d7a208b09d55bd07b7572f727e20",
        vllm_provenance=provenance,
        metal_provenance=provenance,
        artifact_overlay=False,
    )
    assert qualified
    assert failures == []

    qualified, failures = _qualify_packaged_runtime(
        vllm_version="0.28.0",
        metal_version="0.29.0",
        source_revision="7390805822b2d7a208b09d55bd07b7572f727e20",
        vllm_provenance=provenance,
        metal_provenance={**provenance, "native_files": []},
        artifact_overlay=False,
    )
    assert not qualified
    assert "vllm_core_version_mismatch" in failures
    assert "no_hashed_native_artifacts" in failures
