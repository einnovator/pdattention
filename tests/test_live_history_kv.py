from __future__ import annotations

from types import SimpleNamespace

import pytest

from pra_hf.hf_live_kv import (
    HFSparseAttentionMetrics,
    HFSparseKVSegment,
    HFLiveKVRequestCancelled,
    HFLiveKVRuntime,
    dense_reference_attention_mask,
    enable_qwen_sparse_live_kv,
    pack_dynamic_cache_reference,
    segmented_qwen_attention,
    select_dynamic_cache,
)
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


def test_hf_full_and_sparse_selection_are_source_views_without_pack_copy() -> None:
    torch = pytest.importorskip("torch")
    keys = torch.arange(24, dtype=torch.float32).reshape(1, 1, 6, 4)
    values = keys + 100
    source = SimpleNamespace(layers=[SimpleNamespace(keys=keys, values=values)])

    full = select_dynamic_cache(source, LiveKVSelectionPlan.full(6))
    assert not full.physical_kv_copy
    assert full.interval_pack_bytes == 0
    assert full.selected_text_reencoded_tokens == 0
    assert full.cache.layers[0].source_segments[0].keys.data_ptr() == keys.data_ptr()

    sparse = select_dynamic_cache(
        source,
        LiveKVSelectionPlan.create(6, ((0, 2), (4, 6))),
    )
    assert not sparse.physical_kv_copy
    assert sparse.interval_pack_bytes == 0
    segments = sparse.cache.layers[0].source_segments
    assert [segment.keys.shape[-2] for segment in segments] == [2, 2]
    assert [segment.position_start for segment in segments] == [0, 4]
    assert all(segment.keys.untyped_storage().data_ptr() == keys.untyped_storage().data_ptr() for segment in segments)

    packed = pack_dynamic_cache_reference(
        source,
        LiveKVSelectionPlan.create(6, ((0, 2), (4, 6))),
    )
    assert packed.physical_kv_copy
    assert packed.interval_pack_bytes == 2 * 4 * 4 * 4
    assert packed.cache.layers[0].keys[0, 0, :, 0].tolist() == [0, 4, 16, 20]


def test_hf_segmented_qwen_attention_matches_identical_dense_subset() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(17)
    query = torch.randn(1, 4, 2, 8)
    source_keys = torch.randn(1, 2, 8, 8)
    source_values = torch.randn(1, 2, 8, 8)
    tail_keys = torch.randn(1, 2, 2, 8)
    tail_values = torch.randn(1, 2, 2, 8)
    segments = (
        HFSparseKVSegment(source_keys[..., 0:2, :], source_values[..., 0:2, :], 0, 2),
        HFSparseKVSegment(source_keys[..., 4:6, :], source_values[..., 4:6, :], 4, 6),
        HFSparseKVSegment(tail_keys, tail_values, 8, 10),
    )
    positions = torch.tensor([8, 9])
    metrics = HFSparseAttentionMetrics()
    actual = segmented_qwen_attention(
        query, segments, positions, scaling=8 ** -0.5, metrics=metrics
    )

    dense_keys = torch.cat([segment.keys for segment in segments], dim=-2)
    dense_values = torch.cat([segment.values for segment in segments], dim=-2)
    dense_positions = torch.tensor([0, 1, 4, 5, 8, 9])
    dense_keys = dense_keys.repeat_interleave(2, dim=1)
    dense_values = dense_values.repeat_interleave(2, dim=1)
    logits = torch.matmul(query, dense_keys.transpose(2, 3)) * (8 ** -0.5)
    visible = dense_positions.view(1, 1, 1, -1) <= positions.view(1, 1, -1, 1)
    weights = torch.softmax(logits.masked_fill(~visible, -torch.inf), dim=-1)
    expected = torch.matmul(weights, dense_values)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert metrics.interval_pack_bytes == 0
    assert metrics.selected_text_reencoded_tokens == 0
    assert metrics.transient_attention_bytes > 0
    assert all(
        segment.keys.untyped_storage().data_ptr() == source_keys.untyped_storage().data_ptr()
        for segment in segments[:2]
    )


def test_hf_half_precision_reports_bounded_transient_kv_tiles() -> None:
    torch = pytest.importorskip("torch")
    query = torch.randn(1, 2, 1, 4, dtype=torch.float16)
    keys = torch.randn(1, 1, 5, 4, dtype=torch.float16)
    values = torch.randn_like(keys)
    metrics = HFSparseAttentionMetrics()
    output = segmented_qwen_attention(
        query,
        (HFSparseKVSegment(keys, values, 0, 5),),
        torch.tensor([4]),
        scaling=0.5,
        metrics=metrics,
        score_tile_tokens=2,
    )
    assert torch.isfinite(output).all()
    assert metrics.interval_pack_bytes == 0
    assert metrics.transient_kv_copy_bytes == 2 * keys.numel() * 4
    assert metrics.max_transient_kv_tile_bytes == 2 * 1 * 2 * 4 * 4
    assert metrics.max_transient_kv_tile_bytes < metrics.transient_kv_copy_bytes


@pytest.mark.parametrize("family", ["qwen2", "qwen3"])
def test_hf_qwen_wrapper_matches_dense_cache_for_same_sparse_subset(family: str) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if family == "qwen2":
        config_class = transformers.Qwen2Config
        model_class = transformers.Qwen2ForCausalLM
    else:
        config_class = transformers.Qwen3Config
        model_class = transformers.Qwen3ForCausalLM
    torch.manual_seed(23)
    config = config_class(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    model = model_class(config).eval()
    source_ids = torch.tensor([[1, 7, 8, 9, 10, 11, 12, 13]])
    with torch.inference_mode():
        source = model(source_ids, use_cache=True, return_dict=True)
    plan = LiveKVSelectionPlan.create(8, ((0, 2), (4, 6)))
    sparse = select_dynamic_cache(source.past_key_values, plan)
    dense = pack_dynamic_cache_reference(source.past_key_values, plan)
    assert enable_qwen_sparse_live_kv(model) == config.num_hidden_layers
    tail = torch.tensor([[14, 15]])
    positions = torch.tensor([8, 9])
    with torch.inference_mode():
        expected = model(
            tail,
            past_key_values=dense.cache,
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            attention_mask=dense_reference_attention_mask(
                plan, positions, dtype=model.dtype, device=positions.device
            ),
            use_cache=True,
            return_dict=True,
        ).logits
        actual = model(
            tail,
            past_key_values=sparse.cache,
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            use_cache=True,
            return_dict=True,
        ).logits

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert sparse.interval_pack_bytes == 0
    assert sparse.transient_attention_bytes > 0
    assert dense.interval_pack_bytes > 0


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


class _DeterministicHFModel:
    def __init__(self, torch, *, fail: bool = False) -> None:
        self.torch = torch
        self.fail = fail
        self.calls = []

    def parameters(self):
        return iter(())

    def __call__(self, **kwargs):
        if self.fail:
            raise RuntimeError("model failure")
        self.calls.append({
            "input_ids": kwargs["input_ids"].tolist(),
            "position_ids": kwargs["position_ids"].tolist(),
            "cache_position": kwargs["cache_position"].tolist(),
        })
        logits = self.torch.zeros((1, kwargs["input_ids"].shape[1], 8))
        logits[..., 3] = 1
        return SimpleNamespace(
            logits=logits,
            past_key_values=kwargs["past_key_values"],
        )


def _hf_cache(torch):
    keys = torch.arange(24, dtype=torch.float32).reshape(1, 1, 6, 4)
    return SimpleNamespace(
        layers=[SimpleNamespace(keys=keys, values=keys + 100)]
    )


def _hf_runtime(torch):
    return HFLiveKVRuntime(
        dump=lambda cache: cache,
        load=lambda cache: cache,
    )


def _begin_hf_request(runtime, request_id: str):
    return runtime.begin_request(
        request_id,
        "source",
        LiveKVSelectionPlan.create(6, ((0, 2), (4, 6))),
        tenant_id="tenant",
        session_id="session",
        expected_generation=1,
    )


def test_hf_request_borrow_spans_real_decode_and_finishes_exactly_once() -> None:
    torch = pytest.importorskip("torch")
    runtime = _hf_runtime(torch)
    runtime.register_source(
        "source", _hf_cache(torch), tenant_id="tenant", session_id="session",
        generation=1,
    )
    request = _begin_hf_request(runtime, "normal")
    assert runtime.registry.view("source").active_request_ids == ("normal",)
    with pytest.raises(RuntimeError, match="while requests borrow"):
        runtime.offload_source("source")

    model = _DeterministicHFModel(torch)
    result = request.generate(model, [5, 6], max_new_tokens=2)

    assert result.token_ids == (3, 3)
    assert result.source_position_base == 6
    assert result.selected_kv_tokens == 4
    assert result.selected_text_reencoded_tokens == 0
    assert result.physical_kv_copy is False
    assert result.interval_pack_bytes == 0
    assert model.calls[0]["position_ids"] == [[6, 7]]
    assert model.calls[1]["position_ids"] == [[8]]
    assert len(model.calls) == 2
    assert runtime.registry.view("source").active_request_ids == ()
    assert request.outcome == "finished"
    assert request.finish() is False
    assert runtime.snapshot()["terminal_counts"] == {
        "finished": 1, "cancelled": 0, "error": 0,
    }


def test_hf_request_cancellation_and_model_error_release_exactly_once() -> None:
    torch = pytest.importorskip("torch")
    runtime = _hf_runtime(torch)
    runtime.register_source(
        "source", _hf_cache(torch), tenant_id="tenant", session_id="session",
        generation=1,
    )

    request = _begin_hf_request(runtime, "cancel")
    checks = iter((False, True))
    with pytest.raises(HFLiveKVRequestCancelled):
        request.generate(
            _DeterministicHFModel(torch), [5], max_new_tokens=2,
            cancelled=lambda: next(checks),
        )
    assert request.outcome == "cancelled"
    assert request.cancel() is False

    request = _begin_hf_request(runtime, "error")
    with pytest.raises(RuntimeError, match="model failure"):
        request.generate(_DeterministicHFModel(torch, fail=True), [5], max_new_tokens=1)
    assert request.outcome == "error"
    assert request.fail() is False

    request = _begin_hf_request(runtime, "invalid")
    with pytest.raises(ValueError, match="non-empty wire tail"):
        request.generate(_DeterministicHFModel(torch), [], max_new_tokens=1)
    assert request.outcome == "error"
    assert request.fail() is False
    assert runtime.registry.view("source").active_request_ids == ()
    assert runtime.snapshot()["terminal_counts"] == {
        "finished": 0, "cancelled": 1, "error": 2,
    }


def test_hf_session_termination_cancels_borrows_then_tombstones() -> None:
    torch = pytest.importorskip("torch")
    runtime = _hf_runtime(torch)
    cache = _hf_cache(torch)
    runtime.register_source(
        "source", cache, tenant_id="tenant", session_id="session", generation=1,
    )
    request = _begin_hf_request(runtime, "active")

    assert runtime.terminate_session("tenant", "session") == 1
    assert request.outcome == "cancelled"
    assert runtime.registry.view("source") is None
    assert runtime.snapshot()["active_request_ids"] == ()
    with pytest.raises(RuntimeError, match="terminated"):
        runtime.register_source(
            "replacement", cache, tenant_id="tenant", session_id="session",
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
