from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from pra_hf.rag_mlx_native import (
    MLXNativeLayerKV,
    MLXNativeMemory,
    PositionBindingMode,
    _rope_inverse_frequencies,
    compose_cross_document_memory,
    encode_native_memory,
    encode_native_memory_with_mask,
    make_native_prompt_cache,
    native_memory_diagnostics,
    rebind_native_memories_global_packed,
    rebind_native_memories_to_receipt,
    repair_token_indices,
)
from pra_hf.crossdoc_composition import (
    CrossDocumentCompositionConfig,
    CrossDocumentCompositionMode,
    CrossDocumentResidualAdapterConfig,
    GistAttentionMask,
)
from pra_hf.crossdoc_mlx_adapter import (
    adapted_crossdoc_memory,
    create_mlx_crossdoc_residual_adapter,
    mlx_adapter_parameter_count,
    selective_boundary_reencode_memory,
)
from pra_hf.rag_causal_decomposition import (
    DocumentAttentionPolicy,
    build_document_attention_mask,
)
from pra_hf.rag_composition import (
    PositionPolicy,
    RAGPRAProfile,
    SelectedResource,
    compose_resources,
)
from pra_hf.sparse_crossdoc import (
    CrossDocumentAttentionCollector,
    top_attention_edge_plan,
)
from experiments.paper3_2_rag.run_crossdoc_adapter import (
    PreparedExample,
    _answer_logits,
    _loss_components,
)


class _Array:
    def __init__(self, nbytes: int):
        self.nbytes = nbytes


def test_native_memory_accounting_is_dependency_free() -> None:
    layer = MLXNativeLayerKV(_Array(12), _Array(20))
    memory = MLXNativeMemory((layer, layer), source_tokens=7)
    assert layer.nbytes == 32
    assert memory.nbytes == 64
    assert memory.position_base == 7


def test_native_memory_separates_physical_tokens_from_query_position() -> None:
    layer = MLXNativeLayerKV(_Array(12), _Array(20))
    memory = MLXNativeMemory((layer,), source_tokens=9, query_position_base=5)
    assert memory.source_tokens == 9
    assert memory.position_base == 5


@pytest.mark.parametrize(
    ("source_tokens", "query_position_base"), ((0, None), (1, -1))
)
def test_native_memory_rejects_invalid_geometry(
    source_tokens: int, query_position_base: int | None
) -> None:
    with pytest.raises(ValueError):
        MLXNativeMemory(
            (MLXNativeLayerKV(_Array(1), _Array(1)),),
            source_tokens,
            query_position_base,
        )


def test_native_memory_is_immutable() -> None:
    memory = MLXNativeMemory((MLXNativeLayerKV(_Array(1), _Array(1)),), 1)
    with pytest.raises(Exception):
        memory.source_tokens = 2
    assert replace(memory, source_tokens=2, source_positions=(0, 1)).source_tokens == 2


def test_repair_token_indices_supports_mechanistic_policies() -> None:
    assert repair_token_indices(20, 0.25, mode="prefix") == (0, 1, 2, 3, 4)
    boundary = repair_token_indices(
        20, 0.2, mode="boundary", resource_lengths=(10, 10)
    )
    assert boundary == (8, 9, 10, 11)
    later = repair_token_indices(
        20, 0.2, mode="later_prefix", resource_lengths=(8, 6, 6)
    )
    assert later == (8, 9, 14, 15)


def test_repair_token_indices_validates_resource_geometry() -> None:
    with pytest.raises(ValueError, match="do not match"):
        repair_token_indices(20, 0.5, mode="boundary", resource_lengths=(5, 5))


def test_receipt_rebinding_validates_source_token_geometry_before_mlx() -> None:
    memory = MLXNativeMemory((MLXNativeLayerKV(_Array(1), _Array(1)),), 2)
    receipt = SimpleNamespace(
        placements=(
            SimpleNamespace(source_positions=(0,), effective_positions=(4,)),
        ),
        query_position=8,
    )
    with pytest.raises(ValueError, match="source positions"):
        rebind_native_memories_to_receipt(object(), (memory,), receipt)


def test_rope_inverse_frequencies_honor_host_scaled_geometry() -> None:
    mx = pytest.importorskip("mlx.core")
    rope = SimpleNamespace(_freqs=mx.array([2.0, 4.0], dtype=mx.float32))
    result = _rope_inverse_frequencies(rope, dimensions=4)
    assert pytest.approx(result.tolist()) == [0.5, 0.25]


def _tiny_qwen():
    pytest.importorskip("mlx.core")
    qwen3 = pytest.importorskip("mlx_lm.models.qwen3")
    return qwen3.Model(
        qwen3.ModelArgs(
            model_type="qwen3",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=64,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rope_theta=1_000_000.0,
            head_dim=8,
            tie_word_embeddings=True,
        )
    )


def _tiny_llama3():
    pytest.importorskip("mlx.core")
    llama = pytest.importorskip("mlx_lm.models.llama")
    return llama.Model(
        llama.ModelArgs(
            model_type="llama",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=64,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rope_theta=500_000.0,
            head_dim=8,
            tie_word_embeddings=True,
            rope_scaling={
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 64,
                "rope_type": "llama3",
            },
        )
    )


def _packed_receipt(lengths: tuple[int, ...]):
    resources = tuple(
        SelectedResource(
            f"D{index}",
            f"D{index}:0",
            str(index) * 64,
            tuple(range(length)),
            index,
            1.0 / index,
        )
        for index, length in enumerate(lengths, 1)
    )
    return compose_resources(
        resources,
        selection_receipt_id="selection-1",
        profile=RAGPRAProfile.RAG_PLUS_PRA_NATIVE_REBOUND,
        position_policy=PositionPolicy.GLOBAL_PACKED,
        near_gap=0,
    )


def test_prerope_round_trip_matches_postrope_at_identical_positions() -> None:
    mx = pytest.importorskip("mlx.core")
    mx.random.seed(7)
    model = _tiny_qwen()
    tokens = (1, 2, 3, 4)
    post = encode_native_memory(model, tokens)
    pre = encode_native_memory(
        model,
        tokens,
        position_binding_mode=PositionBindingMode.PRE_ROPE,
        model_revision="tiny-qwen-test",
    )
    rebound = rebind_native_memories_to_receipt(model, (pre,), _packed_receipt((4,)))
    diagnostics = native_memory_diagnostics(post, rebound)
    assert diagnostics["max_key_abs_delta"] < 1e-5
    assert diagnostics["max_value_abs_delta"] == 0.0
    assert pre.pre_rope_storage
    assert pre.rope_contract is not None
    assert pre.rope_contract.model_revision == "tiny-qwen-test"


def test_block_isolated_packed_matches_independent_prerope_records() -> None:
    mx = pytest.importorskip("mlx.core")
    mx.random.seed(11)
    model = _tiny_qwen()
    segments = ((1, 2, 3), (4, 5))
    packed = tuple(token for segment in segments for token in segment)
    mask, _ = build_document_attention_mask(
        tuple(map(len, segments)), policy=DocumentAttentionPolicy.NO_CROSS_DOC
    )
    blocked = encode_native_memory_with_mask(model, packed, mask)
    independent = tuple(
        encode_native_memory(
            model,
            segment,
            position_binding_mode=PositionBindingMode.PRE_ROPE,
            model_revision="tiny-qwen-test",
        )
        for segment in segments
    )
    rebound = rebind_native_memories_to_receipt(
        model, independent, _packed_receipt(tuple(map(len, segments)))
    )
    diagnostics = native_memory_diagnostics(blocked, rebound)
    # Independent segment shapes can take a different MLX kernel path. The
    # dedicated shape-matched test below isolates exact RoPE rebinding from
    # this small, version-dependent floating-point amplification.
    assert diagnostics["max_key_abs_delta"] < 1e-3
    assert diagnostics["max_value_abs_delta"] < 5e-4

    query = mx.array([[6, 7]], dtype=mx.int32)
    blocked_logits = model(query, cache=make_native_prompt_cache(model, blocked))
    rebound_logits = model(query, cache=make_native_prompt_cache(model, rebound))
    mx.eval(blocked_logits, rebound_logits)
    assert max(abs(a - b) for a, b in zip(
        blocked_logits.flatten().tolist(), rebound_logits.flatten().tolist()
    )) < 2e-3


def test_teacher_attention_observer_and_full_oracle_replay_match_host_path() -> None:
    mx = pytest.importorskip("mlx.core")
    mx.random.seed(13)
    model = _tiny_qwen()
    tokens = (1, 2, 3, 4)
    full_mask, _ = build_document_attention_mask(
        (2, 2), policy=DocumentAttentionPolicy.FULL_CAUSAL
    )
    collector = CrossDocumentAttentionCollector(
        (2, 2),
        record_ids=("D1", "D2"),
        selection_receipt_id="selection-1",
        model_revision="tiny-qwen-test",
    )
    observed = encode_native_memory_with_mask(
        model,
        tokens,
        full_mask,
        attention_observer=collector.observe,
    )
    graph = collector.finalize()
    plan = top_attention_edge_plan(graph, 1.0)
    blocked_mask, _ = build_document_attention_mask(
        (2, 2), policy=DocumentAttentionPolicy.NO_CROSS_DOC
    )
    replayed = encode_native_memory_with_mask(
        model,
        tokens,
        blocked_mask,
        sparse_mask_provider=lambda layer, _heads: plan.mask_for_layer(
            layer,
            base_mask=blocked_mask,
            source_tokens=graph.source_tokens,
            target_tokens=graph.target_tokens,
        ),
    )
    host = encode_native_memory_with_mask(model, tokens, full_mask)
    observed_diagnostics = native_memory_diagnostics(host, observed)
    replayed_diagnostics = native_memory_diagnostics(host, replayed)
    assert observed_diagnostics["max_key_abs_delta"] < 1e-5
    assert observed_diagnostics["max_value_abs_delta"] < 1e-5
    assert replayed_diagnostics["max_key_abs_delta"] < 1e-5
    assert replayed_diagnostics["max_value_abs_delta"] < 1e-5


def test_zero_initialized_crossdoc_adapter_is_exact_identity() -> None:
    mx = pytest.importorskip("mlx.core")
    mx.random.seed(19)
    model = _tiny_qwen()
    segments = ((1, 2, 3), (4, 5), (6, 7))
    memories = tuple(
        encode_native_memory(
            model,
            segment,
            position_binding_mode=PositionBindingMode.PRE_ROPE,
            model_revision="tiny-qwen-test",
        )
        for segment in segments
    )
    receipt = _packed_receipt(tuple(map(len, segments)))
    independent = rebind_native_memories_to_receipt(model, memories, receipt)
    widths = tuple(
        int(layer.keys.shape[1] * layer.keys.shape[-1])
        for layer in memories[0].layers
    )
    adapter = create_mlx_crossdoc_residual_adapter(
        widths, CrossDocumentResidualAdapterConfig(rank=4), seed=19
    )
    adapted = adapted_crossdoc_memory(model, memories, receipt, adapter)
    diagnostics = native_memory_diagnostics(independent, adapted)
    assert diagnostics["max_key_abs_delta"] == 0.0
    assert diagnostics["max_value_abs_delta"] == 0.0
    assert mlx_adapter_parameter_count(adapter) == sum(24 * width for width in widths)


def test_selective_boundary_reencode_preserves_shape_and_first_record() -> None:
    mx = pytest.importorskip("mlx.core")
    mx.random.seed(23)
    model = _tiny_qwen()
    segments = ((1, 2, 3, 4), (5, 6, 7), (8, 9, 10))
    memories = tuple(
        encode_native_memory(
            model,
            segment,
            position_binding_mode=PositionBindingMode.PRE_ROPE,
            model_revision="tiny-qwen-test",
        )
        for segment in segments
    )
    receipt = _packed_receipt(tuple(map(len, segments)))
    independent = rebind_native_memories_to_receipt(model, memories, receipt)
    repaired, reencode = selective_boundary_reencode_memory(
        model,
        segments,
        memories,
        receipt,
        record_ids=("D1", "D2", "D3"),
        boundary_tokens=2,
    )
    assert repaired.source_tokens == independent.source_tokens == 10
    assert repaired.position_base == independent.position_base
    assert reencode.reencoded_tokens == 4
    assert reencode.context_native_tokens == 4
    assert reencode.boundary_count == 2
    for source, candidate in zip(independent.layers, repaired.layers):
        assert source.keys[:, :, :4, :].tolist() == candidate.keys[:, :, :4, :].tolist()
        assert source.values[:, :, :4, :].tolist() == candidate.values[:, :, :4, :].tolist()


def test_crossdoc_adapter_takes_a_finite_distillation_and_task_step() -> None:
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    optim = pytest.importorskip("mlx.optimizers")
    mx.random.seed(29)
    model = _tiny_qwen()
    segments = ((1, 2, 3), (4, 5, 6))
    packed = tuple(token for segment in segments for token in segment)
    receipt = _packed_receipt(tuple(map(len, segments)))
    teacher_pre = encode_native_memory(
        model,
        packed,
        position_binding_mode=PositionBindingMode.PRE_ROPE,
        model_revision="tiny-qwen-test",
    )
    teacher_post = rebind_native_memories_global_packed(model, (teacher_pre,))
    independent = tuple(
        encode_native_memory(
            model,
            segment,
            position_binding_mode=PositionBindingMode.PRE_ROPE,
            model_revision="tiny-qwen-test",
        )
        for segment in segments
    )
    query_tokens = (7, 8)
    answer_tokens = (9, 10)
    teacher_logits = mx.stop_gradient(
        _answer_logits(model, teacher_post, query_tokens, answer_tokens)
    )
    example = PreparedExample(
        seed=11,
        question=SimpleNamespace(example_id="tiny", answers=("x",)),
        candidate_receipt_id="candidate",
        selection_receipt_id="selection",
        selected_document_ids=("D1", "D2"),
        record_ids=("D1:0", "D2:0"),
        segments=segments,
        composition_receipt=receipt,
        teacher_pre=teacher_pre,
        teacher_post=teacher_post,
        independent_pre=independent,
        teacher_answer_logits=teacher_logits,
        query_tokens=query_tokens,
        answer_tokens=answer_tokens,
    )
    widths = tuple(
        int(layer.keys.shape[1] * layer.keys.shape[-1])
        for layer in independent[0].layers
    )
    config = CrossDocumentResidualAdapterConfig(rank=4)
    adapter = create_mlx_crossdoc_residual_adapter(widths, config, seed=29)

    def objective():
        return _loss_components(adapter, example, model, config, 2.0)[0]

    loss_and_grad = nn.value_and_grad(adapter, objective)
    before, gradients = loss_and_grad()
    optimizer = optim.AdamW(learning_rate=1e-2)
    optimizer.update(adapter, gradients)
    mx.eval(adapter.parameters(), optimizer.state, before)
    after = objective()
    mx.eval(after)
    assert float(before.item()) > 0
    assert float(after.item()) >= 0
    adapted = adapted_crossdoc_memory(model, independent, receipt, adapter)
    diagnostics = native_memory_diagnostics(
        rebind_native_memories_to_receipt(model, independent, receipt), adapted
    )
    assert diagnostics["max_key_abs_delta"] > 0


def test_shape_matched_prerope_rebind_is_exact() -> None:
    mx = pytest.importorskip("mlx.core")
    mx.random.seed(11)
    model = _tiny_qwen()
    segments = ((1, 2, 3), (4, 5))
    packed = tuple(token for segment in segments for token in segment)
    mask, _ = build_document_attention_mask(
        tuple(map(len, segments)), policy=DocumentAttentionPolicy.NO_CROSS_DOC
    )
    blocked = encode_native_memory_with_mask(model, packed, mask)
    pre_rope = encode_native_memory_with_mask(
        model,
        packed,
        mask,
        position_binding_mode=PositionBindingMode.PRE_ROPE,
        model_revision="tiny-qwen-test",
    )
    rebound = rebind_native_memories_global_packed(model, (pre_rope,))
    diagnostics = native_memory_diagnostics(blocked, rebound)
    assert diagnostics["max_key_abs_delta"] == 0.0
    assert diagnostics["max_value_abs_delta"] == 0.0

    query = mx.array([[6, 7]], dtype=mx.int32)
    blocked_logits = model(query, cache=make_native_prompt_cache(model, blocked))
    rebound_logits = model(query, cache=make_native_prompt_cache(model, rebound))
    mx.eval(blocked_logits, rebound_logits)
    assert blocked_logits.tolist() == rebound_logits.tolist()


def test_llama_piecewise_prerope_round_trip_uses_host_frequency_tensor() -> None:
    mx = pytest.importorskip("mlx.core")
    mx.random.seed(13)
    model = _tiny_llama3()
    tokens = (2, 3, 5, 7, 11)
    post = encode_native_memory(model, tokens)
    pre = encode_native_memory(
        model,
        tokens,
        position_binding_mode=PositionBindingMode.PRE_ROPE,
        model_revision="tiny-llama3-test",
    )
    rebound = rebind_native_memories_to_receipt(model, (pre,), _packed_receipt((5,)))
    diagnostics = native_memory_diagnostics(post, rebound)
    assert diagnostics["max_key_abs_delta"] < 1e-5
    assert diagnostics["max_value_abs_delta"] == 0.0
    assert pre.rope_contract is not None
    assert all(
        policy.startswith("host_piecewise_frequency_tensor")
        for policy in pre.rope_contract.scaling_policy
    )


@pytest.mark.parametrize(
    ("mode", "expected_local"),
    ((CrossDocumentCompositionMode.GIST_SA_APPEND, 2),
     (CrossDocumentCompositionMode.GIST_SA_BOUNDARY_8, 5),
     (CrossDocumentCompositionMode.GIST_SA_BOUNDARY_32, 5)),
)
def test_crossdoc_composition_is_request_local_and_deterministic(
    mode: CrossDocumentCompositionMode, expected_local: int
) -> None:
    mx = pytest.importorskip("mlx.core")
    mx.random.seed(19)
    model = _tiny_qwen()
    segments = ((1, 2, 3), (4, 5))
    persistent = tuple(
        encode_native_memory(
            model,
            segment,
            position_binding_mode=PositionBindingMode.PRE_ROPE,
            model_revision="tiny-qwen-test",
        )
        for segment in segments
    )
    before = tuple(
        tuple(layer.keys.tolist() for layer in memory.layers) for memory in persistent
    )
    config = CrossDocumentCompositionConfig(
        mode=mode, attention_mask=GistAttentionMask.ALL_TO_ALL
    )
    first, receipt = compose_cross_document_memory(
        model,
        persistent,
        _packed_receipt(tuple(map(len, segments))),
        record_ids=("D1", "D2"),
        config=config,
    )
    second, second_receipt = compose_cross_document_memory(
        model,
        persistent,
        _packed_receipt(tuple(map(len, segments))),
        record_ids=("D1", "D2"),
        config=config,
    )
    mx.eval(
        [(layer.keys, layer.values) for layer in first.layers],
        [(layer.keys, layer.values) for layer in second.layers],
    )
    assert first.source_tokens == 5 + expected_local
    assert first.position_base == 5 + expected_local
    assert receipt.request_local_native_tokens == expected_local
    assert receipt.gist_attention_edges == 4
    assert receipt.source_memory_digest == second_receipt.source_memory_digest
    for left, right in zip(first.layers, second.layers):
        assert left.keys.tolist() == right.keys.tolist()
        assert left.values.tolist() == right.values.tolist()
    after = tuple(
        tuple(layer.keys.tolist() for layer in memory.layers) for memory in persistent
    )
    assert before == after
