"""Offline correctness gates for the thin Hugging Face Qwen PRA adapter."""

from __future__ import annotations

import copy
import io
from dataclasses import replace

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import Qwen3Config, Qwen3ForCausalLM

from pra_torch.hf import (
    ATTENTION_INPUT_HIDDEN_STATE,
    CENTERED_ROPE_KEY,
    MEMORY_GATE_FIXED,
    MEMORY_GATE_PER_LAYER,
    MEMORY_GATE_SINGLE,
    PRAHFConfig,
    HFRoutingProjection,
    inject_pra,
)
from pra_torch.hf.late_band_lora import PRAHFConditionalOutputLoRA
from pra_torch.memory_batching import native_kv_attention
from pra_torch.memory import SelectedChunk


def _tiny_qwen() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=67,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=66,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).eval()


def _hf_config(**overrides) -> PRAHFConfig:
    values = {
        "layer_ids": (-1,),
        "model_max_context_tokens": 64,
        "max_prompt_direct_tokens": 16,
        "encoding_block_tokens": 16,
        "routing_chunk_tokens": 4,
        "max_materialized_memory_tokens": 16,
        "top_k_references": 1,
        "top_k_chunks_per_reference": 2,
        "trigger_threshold": float("-inf"),
        "kv_cache_residency": "cpu",
        "collect_detailed_timing": False,
    }
    values.update(overrides)
    return PRAHFConfig(**values)


def test_hf_default_names_attention_input_hidden_state_explicitly():
    assert PRAHFConfig().routing_representation == ATTENTION_INPUT_HIDDEN_STATE
    assert (
        PRAHFConfig(routing_representation="hidden_state").routing_representation
        == ATTENTION_INPUT_HIDDEN_STATE
    )


def test_hf_query_strategy_config_validates_runtime_modes():
    assert PRAHFConfig().query_strategy == "last"
    with pytest.raises(ValueError, match="query_strategy"):
        PRAHFConfig(query_strategy="question_mean")
    with pytest.raises(ValueError, match="query_window"):
        PRAHFConfig(query_strategy="uniform", query_window=0)
    with pytest.raises(ValueError, match="query_half_life"):
        PRAHFConfig(query_strategy="exponential", query_half_life=0)


def test_hf_memory_gate_config_and_parameter_ownership():
    with pytest.raises(ValueError, match="memory_gate_mode"):
        PRAHFConfig(memory_gate_mode="per_token")

    handle = inject_pra(
        _tiny_qwen(),
        _hf_config(layer_ids=(0, 1), memory_gate_mode=MEMORY_GATE_FIXED),
    )
    assert handle.memory_gate_parameters() == []
    assert handle.memory_gate_values() == {0: 1.0, 1: 1.0}

    handle.configure_memory_gate(MEMORY_GATE_SINGLE, initial_value=0.75)
    assert len(handle.memory_gate_parameters()) == 1
    assert handle.memory_gate_values() == {0: 0.75, 1: 0.75}

    handle.configure_memory_gate(MEMORY_GATE_PER_LAYER, initial_value=0.5)
    assert len(handle.memory_gate_parameters()) == 2
    assert handle.memory_gate_values() == {0: 0.5, 1: 0.5}


def test_hf_residual_adapter_is_lazy_and_counts_only_active_width():
    handle = inject_pra(
        _tiny_qwen(),
        _hf_config(layer_ids=(0, 1)),
    )
    assert handle.residual_adapter_parameters() == []

    handle.configure_residual_adapter(16)
    # Per layer: (D * B + B) down projection plus (B * D) up projection.
    expected = 2 * ((32 * 16 + 16) + (16 * 32))
    assert sum(
        parameter.numel() for parameter in handle.residual_adapter_parameters()
    ) == expected

    handle.configure_residual_adapter(0)
    assert handle.residual_adapter_parameters() == []


def test_hf_late_band_lora_is_lazy_validated_and_counts_active_rank():
    with pytest.raises(ValueError, match="late_band_lora_rank"):
        PRAHFConfig(late_band_lora_rank=-1)
    with pytest.raises(ValueError, match="late_band_lora_alpha"):
        PRAHFConfig(late_band_lora_alpha=0.0)
    with pytest.raises(ValueError, match="late_band_lora_dropout"):
        PRAHFConfig(late_band_lora_dropout=1.0)

    handle = inject_pra(_tiny_qwen(), _hf_config(layer_ids=(0, 1)))
    assert handle.late_band_lora_parameters() == []

    handle.configure_late_band_lora(4, alpha=4.0, dropout=0.0)
    # Per layer, output-projection LoRA owns D*r down and r*D up weights.
    expected = 2 * (32 * 4 + 4 * 32)
    assert sum(parameter.numel() for parameter in handle.late_band_lora_parameters()) == expected

    handle.configure_late_band_lora(0)
    assert handle.late_band_lora_parameters() == []


def test_hf_output_lora_supports_rectangular_native_projection():
    adapter = PRAHFConditionalOutputLoRA(
        64,
        32,
        4,
        alpha=4.0,
        dropout=0.0,
    )
    output = adapter(torch.randn(2, 5, 64))
    assert output.shape == (2, 5, 32)
    assert torch.count_nonzero(output) == 0


def test_qwen_runtime_query_strategy_aggregates_attention_input_states():
    handle = inject_pra(
        _tiny_qwen(),
        _hf_config(query_strategy="uniform", query_window=2),
    )
    adapter = handle.adapters[1]
    hidden = torch.arange(4 * 32, dtype=torch.float32).view(1, 4, 32)
    unused = torch.empty(1, 4, 4, 8)
    actual = adapter._routing_query_states(hidden, unused, unused)
    assert torch.equal(actual, hidden[:, -2:, :].mean(dim=1))


def test_learned_projection_changes_only_routing_width_not_native_detail_kv():
    torch.manual_seed(102)
    original = _tiny_qwen()
    projection = HFRoutingProjection(32, 8, "shared_linear").eval()
    projected = inject_pra(
        copy.deepcopy(original),
        _hf_config(),
        routing_projection=projection,
    )
    baseline = inject_pra(copy.deepcopy(original), _hf_config())
    ids = torch.tensor([[11, 12, 13, 14]])
    projected_chunk = projected.add_reference("mem://projected", ids).layer_memory[1].chunks[0]
    baseline_chunk = baseline.add_reference("mem://baseline", ids).layer_memory[1].chunks[0]

    assert projected_chunk.routing_gist.k.shape == (1, 8)
    assert projected_chunk.metadata["routing_gist_bytes"] == 8 * 4
    assert projected_chunk.routing_gist.metadata["routing_projection_width"] == 8
    assert torch.equal(projected_chunk.token_kv.k, baseline_chunk.token_kv.k)
    assert torch.equal(projected_chunk.token_kv.v, baseline_chunk.token_kv.v)
    hidden = torch.randn(1, 4, 32)
    unused = torch.empty(1, 4, 4, 8)
    query = projected.adapters[1]._routing_query_states(hidden, unused, unused)
    assert query.shape == (1, 8)


def test_centered_rope_config_rejects_invalid_policy_and_pooling():
    with pytest.raises(ValueError, match="center_policy"):
        PRAHFConfig(
            routing_representation=CENTERED_ROPE_KEY,
            centered_rope_center_policy="nearest",
        )
    with pytest.raises(ValueError, match="mean.*segment_mean"):
        PRAHFConfig(
            routing_representation=CENTERED_ROPE_KEY,
            gist_mode="kmeans",
        )


def test_qwen_disabled_adapter_has_exact_logits_hidden_states_and_generation_parity():
    torch.manual_seed(101)
    original = _tiny_qwen()
    adapted_model = copy.deepcopy(original)
    handle = inject_pra(adapted_model, _hf_config())
    ids = torch.tensor([[1, 7, 9, 3, 5]])

    with torch.no_grad():
        expected = original(ids, output_hidden_states=True, use_cache=True)
        actual = handle.model(ids, output_hidden_states=True, use_cache=True)
        expected_generation = original.generate(ids, max_new_tokens=3, do_sample=False)
        actual_generation = handle.model.generate(ids, max_new_tokens=3, do_sample=False)

    assert torch.equal(actual.logits, expected.logits)
    assert all(torch.equal(left, right) for left, right in zip(actual.hidden_states, expected.hidden_states))
    assert torch.equal(actual_generation, expected_generation)
    assert actual.past_key_values.get_seq_length() == expected.past_key_values.get_seq_length()


def test_qwen_reference_capture_keeps_native_gqa_layout_and_uses_memory():
    torch.manual_seed(103)
    handle = inject_pra(_tiny_qwen(), _hf_config())
    entry = handle.add_reference("mem://facts", torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18]]))
    chunks = entry.layer_memory[1].chunks

    assert len(chunks) == 2
    assert all(chunk.token_kv.k.shape[1] == 2 for chunk in chunks)
    assert all(chunk.token_kv.k.device.type == "cpu" for chunk in chunks)
    assert all(chunk.token_kv.position_state == "post_position" for chunk in chunks)
    assert all(
        chunk.metadata["routing_representation"] == ATTENTION_INPUT_HIDDEN_STATE
        for chunk in chunks
    )
    assert all(chunk.routing_gist.k.shape == (1, 32) for chunk in chunks)
    assert [chunk.logical_start for chunk in chunks] == [0, 4]

    query = chunks[0].routing_gist.k[0]
    cpu_hit = handle.cache.search(query, 1, handle.pra_config)[0][0].chunk_id
    if torch.cuda.is_available():
        gpu_hit = handle.cache.search(query.cuda(), 1, handle.pra_config)[0][0].chunk_id
        assert gpu_hit == cpu_hit

    handle.set_memory_enabled(True)
    with torch.no_grad():
        output = handle.model(torch.tensor([[1, 4, 8, 12]]), use_cache=False)
    diagnostics = handle.diagnostics_by_layer()[1]
    assert output.logits.shape == (1, 4, 67)
    assert diagnostics["hf_query_heads"] == 4
    assert diagnostics["hf_native_kv_heads"] == 2
    assert diagnostics["retrieved_physical_kv_tokens"] == 8


def test_qwen_capture_exposes_matched_pre_rope_features_and_post_rope_detail():
    torch.manual_seed(105)
    handle = inject_pra(_tiny_qwen(), _hf_config(routing_representation="pre_rope_key"))
    adapter = handle.adapters[1]
    ids = torch.tensor([[1, 7, 9, 3]])
    positions = torch.arange(ids.shape[1]).unsqueeze(0)

    adapter.begin_capture(positions)
    with torch.no_grad():
        handle.model(input_ids=ids, position_ids=positions, use_cache=False)
    captured = adapter.consume_capture()

    assert captured.pre_query.shape == (1, 4, 4, 8)
    assert captured.post_query.shape == captured.pre_query.shape
    assert captured.pre_key.shape == captured.detail_kv.k.shape == (1, 2, 4, 8)
    assert captured.hidden_states.shape == (1, 4, 32)
    assert captured.detail_kv.position_state == "post_position"
    assert torch.equal(captured.detail_kv.position_ids, positions)
    assert not torch.equal(captured.pre_key, captured.detail_kv.k)


def test_routing_representation_switch_preserves_materialized_post_rope_kv():
    torch.manual_seed(106)
    original = _tiny_qwen()
    ids = torch.tensor([[11, 12, 13, 14]])
    entries = {}
    for representation in (
        "post_rope_key",
        "pre_rope_key",
        ATTENTION_INPUT_HIDDEN_STATE,
    ):
        handle = inject_pra(
            copy.deepcopy(original),
            _hf_config(routing_representation=representation),
        )
        entries[representation] = handle.add_reference(f"mem://{representation}", ids)

    chunks = {name: entry.layer_memory[1].chunks[0] for name, entry in entries.items()}
    baseline = chunks["post_rope_key"]
    for representation, chunk in chunks.items():
        assert torch.equal(chunk.token_kv.k, baseline.token_kv.k)
        assert torch.equal(chunk.token_kv.v, baseline.token_kv.v)
        assert chunk.token_kv.position_state == "post_position"
        assert chunk.metadata["routing_representation"] == representation
    assert baseline.routing_gist.k.shape == (1, 16)
    assert chunks["pre_rope_key"].routing_gist.k.shape == (1, 16)
    assert chunks[ATTENTION_INPUT_HIDDEN_STATE].routing_gist.k.shape == (1, 32)


def test_hidden_state_segment_means_share_one_parent_native_kv_payload():
    torch.manual_seed(1061)
    handle = inject_pra(
        _tiny_qwen(),
        _hf_config(
            routing_representation=ATTENTION_INPUT_HIDDEN_STATE,
            gist_mode="segment_mean",
            gists_per_chunk=4,
        ),
    )
    entry = handle.add_reference("mem://segments", torch.tensor([[11, 12, 13, 14]]))
    chunk = entry.layer_memory[1].chunks[0]

    assert chunk.routing_gist.k.shape == (4, 32)
    assert chunk.routing_gist.metadata["segment_token_spans"] == [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
    ]
    assert chunk.token_kv.k.shape == (1, 2, 4, 8)
    assert chunk.token_kv.v.shape == (1, 2, 4, 8)
    assert chunk.metadata["routing_gist_bytes"] == 4 * 32 * 4


def test_centered_rope_mean_uses_fractional_center_and_preserves_native_payload():
    torch.manual_seed(1062)
    original = _tiny_qwen()
    ids = torch.tensor([[11, 12, 13, 14]])
    handles = {
        representation: inject_pra(
            copy.deepcopy(original),
            _hf_config(routing_representation=representation),
        )
        for representation in ("pre_rope_key", "post_rope_key", CENTERED_ROPE_KEY)
    }
    chunks = {
        name: handle.add_reference(f"mem://{name}", ids).layer_memory[1].chunks[0]
        for name, handle in handles.items()
    }
    centered = chunks[CENTERED_ROPE_KEY]
    pre_gist = chunks["pre_rope_key"].routing_gist.k
    adapter = handles[CENTERED_ROPE_KEY].adapters[1]
    expected = adapter.rotate_routing_keys(pre_gist, torch.tensor([1.5]))

    assert torch.allclose(centered.routing_gist.k, expected, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(
        centered.routing_gist.k,
        chunks["post_rope_key"].routing_gist.k,
        atol=1e-6,
        rtol=1e-6,
    )
    assert centered.routing_gist.metadata["exact_center_positions"] == [1.5]
    assert centered.routing_gist.metadata["applied_center_positions"] == [1.5]
    assert centered.routing_gist.metadata["source_token_spans"] == [[0, 4]]
    for name in ("pre_rope_key", "post_rope_key"):
        assert torch.equal(centered.token_kv.k, chunks[name].token_kv.k)
        assert torch.equal(centered.token_kv.v, chunks[name].token_kv.v)
    assert centered.token_kv.position_state == "post_position"

    query = centered.routing_gist.k[0]
    cpu_hit = handles[CENTERED_ROPE_KEY].cache.search(
        query, 1, handles[CENTERED_ROPE_KEY].pra_config
    )[0][0].chunk_id
    if torch.cuda.is_available():
        gpu_hit = handles[CENTERED_ROPE_KEY].cache.search(
            query.cuda(), 1, handles[CENTERED_ROPE_KEY].pra_config
        )[0][0].chunk_id
        assert gpu_hit == cpu_hit
    handles[CENTERED_ROPE_KEY].set_memory_enabled(True)
    with torch.no_grad():
        output = handles[CENTERED_ROPE_KEY].model(
            torch.tensor([[1, 4, 8, 12]]), use_cache=False
        )
    assert output.logits.shape == (1, 4, 67)


def test_centered_rope_segment_means_use_each_subspan_center():
    torch.manual_seed(1063)
    handle = inject_pra(
        _tiny_qwen(),
        _hf_config(
            routing_representation=CENTERED_ROPE_KEY,
            gist_mode="segment_mean",
            gists_per_chunk=4,
            routing_chunk_tokens=8,
        ),
    )
    entry = handle.add_reference(
        "mem://centered-segments",
        torch.tensor(
            [[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]]
        ),
    )
    chunk = entry.layer_memory[1].chunks[0]
    second = entry.layer_memory[1].chunks[1]

    assert chunk.routing_gist.k.shape == (4, 16)
    assert chunk.routing_gist.metadata["segment_token_spans"] == [
        [0, 2],
        [2, 4],
        [4, 6],
        [6, 8],
    ]
    assert chunk.routing_gist.metadata["exact_center_positions"] == [
        0.5,
        2.5,
        4.5,
        6.5,
    ]
    assert chunk.routing_gist.metadata["source_token_spans"] == [
        [0, 2],
        [2, 4],
        [4, 6],
        [6, 8],
    ]
    assert second.routing_gist.metadata["exact_center_positions"] == [
        8.5,
        10.5,
        12.5,
        14.5,
    ]
    assert second.routing_gist.metadata["source_token_spans"] == [
        [8, 10],
        [10, 12],
        [12, 14],
        [14, 16],
    ]


def test_centered_rope_fractional_floor_and_ceil_controls_are_distinct():
    torch.manual_seed(1064)
    original = _tiny_qwen()
    ids = torch.tensor([[11, 12, 13, 14]])
    gists = {}
    for policy in ("exact", "floor", "ceil"):
        handle = inject_pra(
            copy.deepcopy(original),
            _hf_config(
                routing_representation=CENTERED_ROPE_KEY,
                centered_rope_center_policy=policy,
            ),
        )
        chunk = handle.add_reference(f"mem://center-{policy}", ids).layer_memory[1].chunks[0]
        gists[policy] = chunk.routing_gist.k
        expected_center = {"exact": 1.5, "floor": 1.0, "ceil": 2.0}[policy]
        assert chunk.routing_gist.metadata["applied_center_positions"] == [
            expected_center
        ]

    assert not torch.allclose(gists["exact"], gists["floor"])
    assert not torch.allclose(gists["exact"], gists["ceil"])
    assert not torch.allclose(gists["floor"], gists["ceil"])


def test_native_centered_rope_composes_signed_displacements():
    torch.manual_seed(1065)
    handle = inject_pra(
        _tiny_qwen(),
        _hf_config(routing_representation=CENTERED_ROPE_KEY),
    )
    adapter = handle.adapters[1]
    keys = torch.randn(3, 16)
    first = torch.tensor([2.5, -3.0, 7.25])
    second = torch.tensor([-1.0, 4.5, -2.25])

    composed = adapter.rotate_routing_keys(
        adapter.rotate_routing_keys(keys, first), second
    )
    direct = adapter.rotate_routing_keys(keys, first + second)
    restored = adapter.rotate_routing_keys(
        adapter.rotate_routing_keys(keys, first), -first
    )

    assert torch.allclose(composed, direct, atol=2e-6, rtol=2e-6)
    assert torch.allclose(restored, keys, atol=2e-6, rtol=2e-6)


def test_pre_rope_gqa_routing_query_matches_native_query_groups():
    torch.manual_seed(107)
    handle = inject_pra(_tiny_qwen(), _hf_config(routing_representation="pre_rope_key"))
    adapter = handle.adapters[1]
    query = torch.randn(1, 4, 3, 8)

    actual = adapter.pra_core.prepare_pra_query(query)
    expected = query[:, :, -1, :].view(1, 2, 2, 8).mean(dim=2).reshape(1, 16)

    assert torch.equal(actual, expected)


def test_pre_rope_routing_cache_round_trip_and_device_parity():
    torch.manual_seed(108)
    handle = inject_pra(_tiny_qwen(), _hf_config(routing_representation="pre_rope_key"))
    entry = handle.add_reference("mem://round-trip", torch.tensor([[2, 4, 6, 8]]))
    stream = io.BytesIO()
    torch.save(entry, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    chunk = restored.layer_memory[1].chunks[0]

    assert chunk.metadata["routing_representation"] == "pre_rope_key"
    assert chunk.token_kv.position_state == "post_position"
    assert torch.equal(chunk.routing_gist.k, entry.layer_memory[1].chunks[0].routing_gist.k)

    query = chunk.routing_gist.k[0]
    cpu_hit = handle.cache.search(query, 1, handle.pra_config)[0][0].chunk_id
    if torch.cuda.is_available():
        gpu_hit = handle.cache.search(query.cuda(), 1, handle.pra_config)[0][0].chunk_id
        assert gpu_hit == cpu_hit


def test_native_kv_attention_replays_gqa_without_expanding_stored_memory():
    torch.manual_seed(107)
    query = torch.randn(1, 4, 3, 8)
    local_key = torch.randn(1, 2, 3, 8)
    local_value = torch.randn(1, 2, 3, 8)
    memory_key = torch.randn(1, 2, 5, 8)
    memory_value = torch.randn(1, 2, 5, 8)

    actual, _ = native_kv_attention(
        query,
        local_key,
        local_value,
        [memory_key],
        [memory_value],
    )
    full_key = torch.cat((memory_key, local_key), dim=2).repeat_interleave(2, dim=1)
    full_value = torch.cat((memory_value, local_value), dim=2).repeat_interleave(2, dim=1)
    scores = query @ full_key.transpose(-2, -1) * (8**-0.5)
    visible = torch.cat((torch.ones(3, 5, dtype=torch.bool), torch.tril(torch.ones(3, 3, dtype=torch.bool))), dim=1)
    expected = torch.softmax(scores.masked_fill(~visible[None, None], float("-inf")), dim=-1) @ full_value

    assert memory_key.shape[1] == 2
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_fixed_selected_set_has_identical_materialization_and_attention_output():
    torch.manual_seed(1081)
    handle = inject_pra(_tiny_qwen(), _hf_config())
    entry = handle.add_reference(
        "mem://fixed-selection", torch.tensor([[11, 12, 13, 14]])
    )
    chunk = entry.layer_memory[1].chunks[0]
    selected = handle.cache.search(
        chunk.routing_gist.k[0], 1, handle.pra_config
    )[0][0]
    core = handle.adapters[1].pra_core
    query = torch.randn(1, 4, 2, 8)
    local_key = torch.randn(1, 2, 2, 8)
    local_value = torch.randn(1, 2, 2, 8)

    routed = core.prepare_selected_memory(
        query,
        [[selected]],
        direct_tokens=2,
        rankings=[[{"selection_source": "router"}]],
    )
    oracle = core.prepare_selected_memory(
        query,
        [[selected]],
        direct_tokens=2,
        rankings=[[{"selection_source": "oracle"}]],
    )
    native_closure = core.prepare_selected_memory(
        query,
        [[replace(selected, metadata={"selection_policy": "native_qk_closure"})]],
        direct_tokens=2,
        rankings=[[{"selection_source": "native_qk_closure"}]],
    )
    routed_output, _, _ = core.apply_pra_attention(
        query, local_key, local_value, routed
    )
    oracle_output, _, _ = core.apply_pra_attention(
        query, local_key, local_value, oracle
    )
    native_output, _, _ = core.apply_pra_attention(
        query, local_key, local_value, native_closure
    )

    assert torch.equal(routed.keys[0], chunk.token_kv.k)
    assert torch.equal(routed.values[0], chunk.token_kv.v)
    assert torch.equal(routed.keys[0], oracle.keys[0])
    assert torch.equal(routed.values[0], oracle.values[0])
    assert torch.equal(routed_output, oracle_output)
    assert torch.equal(routed.keys[0], native_closure.keys[0])
    assert torch.equal(routed.values[0], native_closure.values[0])
    assert torch.equal(routed_output, native_output)


def test_qwen_capture_retains_exact_normalized_pre_rope_qk():
    """Gate-3 capture must equal Qwen's native projections before RoPE."""
    torch.manual_seed(10815)
    handle = inject_pra(_tiny_qwen(), _hf_config())
    adapter = handle.adapters[1]
    positions = torch.arange(4).unsqueeze(0)
    adapter.begin_capture(positions)
    with torch.no_grad():
        handle.model(torch.tensor([[1, 4, 8, 12]]), position_ids=positions, use_cache=False)
    captured = adapter.consume_capture()
    attention = adapter.original_attention
    shape = (*captured.hidden_states.shape[:2], -1, attention.head_dim)
    expected_q = attention.q_proj(captured.hidden_states).view(shape)
    expected_k = attention.k_proj(captured.hidden_states).view(shape)
    if hasattr(attention, "q_norm"):
        expected_q = attention.q_norm(expected_q)
    if hasattr(attention, "k_norm"):
        expected_k = attention.k_norm(expected_k)
    assert torch.equal(captured.pre_query, expected_q.transpose(1, 2))
    assert torch.equal(captured.pre_key, expected_k.transpose(1, 2))


def test_qwen_fixed_selection_override_replays_requested_chunk():
    torch.manual_seed(1082)
    handle = inject_pra(_tiny_qwen(), _hf_config())
    entry = handle.add_reference(
        "mem://oracle", torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18]])
    )
    chunks = entry.layer_memory[1].chunks
    routed = handle.cache.search(chunks[0].routing_gist.k[0], 1, handle.pra_config)[0]
    oracle = replace(routed[0], chunk=chunks[1], chunk_score=1.0)

    handle.configure_memory_layers({1}, fixed_selections={1: [[oracle]]})
    with torch.no_grad():
        handle.model(torch.tensor([[1, 4, 8, 12]]), use_cache=False)

    assert [hit.chunk_id for hit in handle.adapters[1].last_selected_chunks[0]] == [
        chunks[1].chunk_id
    ]
    assert handle.diagnostics_by_layer()[1]["retrieved_physical_kv_tokens"] == 4

    handle.configure_memory_layers(set())
    assert handle.adapters[1].fixed_selected_chunks is None


def test_qwen_fixed_gate_one_matches_trainable_gate_one_with_memory():
    torch.manual_seed(10821)
    original = _tiny_qwen()
    fixed = inject_pra(
        copy.deepcopy(original),
        _hf_config(memory_gate_mode=MEMORY_GATE_FIXED),
    )
    learned = inject_pra(
        copy.deepcopy(original),
        _hf_config(memory_gate_mode=MEMORY_GATE_SINGLE),
    )
    reference = torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18]])
    fixed.add_reference("mem://fixed-gate", reference)
    learned.add_reference("mem://fixed-gate", reference)
    fixed.set_memory_enabled(True)
    learned.set_memory_enabled(True)
    query = torch.tensor([[1, 4, 8, 12]])

    with torch.no_grad():
        expected = fixed.model(query, use_cache=False).logits
        actual = learned.model(query, use_cache=False).logits

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_qwen_zero_memory_gate_matches_disabled_memory_and_preserves_gradient_isolation():
    torch.manual_seed(10822)
    model = _tiny_qwen()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    handle = inject_pra(
        model,
        _hf_config(memory_gate_mode=MEMORY_GATE_SINGLE),
    )
    handle.add_reference(
        "mem://zero-gate", torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18]])
    )
    query = torch.tensor([[1, 4, 8, 12]])
    handle.set_memory_enabled(False)
    with torch.no_grad():
        local = handle.model(query, use_cache=False).logits

    handle.configure_memory_gate(MEMORY_GATE_SINGLE, initial_value=0.0)
    handle.set_memory_enabled(True)
    actual = handle.model(query, use_cache=False).logits
    torch.testing.assert_close(actual, local, rtol=1e-5, atol=1e-6)

    loss = actual.float().square().mean()
    loss.backward()
    gate_parameters = handle.memory_gate_parameters()
    assert len(gate_parameters) == 1
    assert gate_parameters[0].grad is not None
    assert all(
        parameter.grad is None
        for name, parameter in handle.model.named_parameters()
        if "pra_memory_gate" not in name
    )


def test_qwen_pra_off_ignores_trainable_gate_with_populated_cache_exactly():
    torch.manual_seed(10823)
    original = _tiny_qwen()
    handle = inject_pra(
        copy.deepcopy(original),
        _hf_config(memory_gate_mode=MEMORY_GATE_PER_LAYER),
    )
    handle.add_reference(
        "mem://disabled-gate", torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18]])
    )
    handle.configure_memory_gate(MEMORY_GATE_PER_LAYER, initial_value=9.0)
    handle.set_memory_enabled(False)
    query = torch.tensor([[1, 4, 8, 12]])

    with torch.no_grad():
        expected = original(query, use_cache=False).logits
        actual = handle.model(query, use_cache=False).logits

    assert torch.equal(actual, expected)


def test_qwen_zero_initialized_residual_adapter_matches_frozen_pra():
    torch.manual_seed(10824)
    original = _tiny_qwen()
    frozen = inject_pra(copy.deepcopy(original), _hf_config())
    adapted = inject_pra(copy.deepcopy(original), _hf_config())
    adapted.configure_residual_adapter(16)
    reference = torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18]])
    frozen.add_reference("mem://residual-init", reference)
    adapted.add_reference("mem://residual-init", reference)
    frozen.set_memory_enabled(True)
    adapted.set_memory_enabled(True)
    query = torch.tensor([[1, 4, 8, 12]])

    with torch.no_grad():
        expected = frozen.model(query, use_cache=False).logits
        actual = adapted.model(query, use_cache=False).logits

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_qwen_residual_adapter_receives_exclusive_gradients_and_bypasses_pra_off():
    torch.manual_seed(10825)
    model = _tiny_qwen()
    original = copy.deepcopy(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    handle = inject_pra(model, _hf_config())
    handle.configure_residual_adapter(16)
    handle.add_reference(
        "mem://residual-gradient",
        torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18]]),
    )
    query = torch.tensor([[1, 4, 8, 12]])

    handle.set_memory_enabled(False)
    with torch.no_grad():
        expected = original(query, use_cache=False).logits
        disabled = handle.model(query, use_cache=False).logits
    assert torch.equal(disabled, expected)

    handle.set_memory_enabled(True)
    output = handle.model(query, use_cache=False).logits
    output.float().square().mean().backward()
    trainable = handle.residual_adapter_parameters()
    assert trainable
    assert any(parameter.grad is not None for parameter in trainable)
    assert all(
        parameter.grad is None
        for name, parameter in handle.model.named_parameters()
        if "pra_residual_adapter" not in name
    )


def test_qwen_zero_initialized_lora_matches_frozen_pra_with_memory():
    torch.manual_seed(10826)
    original = _tiny_qwen()
    frozen = inject_pra(copy.deepcopy(original), _hf_config())
    adapted = inject_pra(copy.deepcopy(original), _hf_config())
    adapted.configure_late_band_lora(4, alpha=4.0)
    reference = torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18]])
    frozen.add_reference("mem://lora-init", reference)
    adapted.add_reference("mem://lora-init", reference)
    frozen.set_memory_enabled(True)
    adapted.set_memory_enabled(True)
    query = torch.tensor([[1, 4, 8, 12]])

    with torch.no_grad():
        expected = frozen.model(query, use_cache=False).logits
        actual = adapted.model(query, use_cache=False).logits

    assert torch.equal(actual, expected)


def test_qwen_conditional_lora_receives_exclusive_gradients_and_is_exact_when_off():
    torch.manual_seed(10827)
    model = _tiny_qwen()
    original = copy.deepcopy(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    handle = inject_pra(model, _hf_config())
    handle.configure_late_band_lora(4, alpha=4.0)
    handle.add_reference(
        "mem://lora-gradient",
        torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18]]),
    )
    query = torch.tensor([[1, 4, 8, 12]])

    handle.set_memory_enabled(False)
    with torch.no_grad():
        expected = original(query, use_cache=False).logits
        disabled = handle.model(query, use_cache=False).logits
    assert torch.equal(disabled, expected)

    handle.set_memory_enabled(True)
    output = handle.model(query, use_cache=False).logits
    output.float().square().mean().backward()
    trainable = handle.late_band_lora_parameters()
    assert trainable
    assert any(parameter.grad is not None for parameter in trainable)
    assert all(
        parameter.grad is None
        for name, parameter in handle.model.named_parameters()
        if "pra_late_band_lora" not in name
    )


def test_qwen_residual_and_lora_combine_with_exclusive_gradients_and_exact_bypass():
    torch.manual_seed(10828)
    model = _tiny_qwen()
    original = copy.deepcopy(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    handle = inject_pra(model, _hf_config())
    handle.configure_residual_adapter(16)
    handle.configure_late_band_lora(4, alpha=4.0)
    handle.add_reference(
        "mem://combo-gradient",
        torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18]]),
    )
    query = torch.tensor([[1, 4, 8, 12]])

    handle.set_memory_enabled(False)
    with torch.no_grad():
        expected = original(query, use_cache=False).logits
        disabled = handle.model(query, use_cache=False).logits
        expected_generation = original.generate(query, max_new_tokens=2, do_sample=False)
        disabled_generation = handle.model.generate(
            query, max_new_tokens=2, do_sample=False
        )
    assert torch.equal(disabled, expected)
    assert torch.equal(disabled_generation, expected_generation)

    handle.set_memory_enabled(True)
    output = handle.model(query, use_cache=False).logits
    output.float().square().mean().backward()
    trainable = handle.memory_use_parameters()
    assert len(trainable) == len(handle.residual_adapter_parameters()) + len(
        handle.late_band_lora_parameters()
    )
    assert any(parameter.grad is not None for parameter in trainable)
    assert all(
        parameter.grad is None
        for name, parameter in handle.model.named_parameters()
        if "pra_residual_adapter" not in name and "pra_late_band_lora" not in name
    )


def test_qwen_attention_diagnostics_are_explicit_and_ephemeral():
    torch.manual_seed(1083)
    handle = inject_pra(_tiny_qwen(), _hf_config())
    handle.add_reference("mem://attention-trace", torch.tensor([[11, 12, 13, 14]]))
    handle.configure_memory_layers({1})
    handle.set_attention_diagnostics(True)

    with torch.no_grad():
        handle.model(torch.tensor([[1, 4, 8, 12]]), use_cache=False)

    weights = handle.adapters[1].last_attention_weights
    assert weights is not None
    assert weights.shape == (1, 4, 4, 8)
    handle.set_attention_diagnostics(False)
    assert handle.adapters[1].last_attention_weights is None


def test_route_once_reuses_chunk_identity_but_not_cross_layer_kv():
    torch.manual_seed(1084)
    handle = inject_pra(_tiny_qwen(), _hf_config(layer_ids=(0, 1)))
    entry = handle.add_reference(
        "mem://multilayer", torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18]])
    )
    source_chunk = entry.layer_memory[1].chunks[0]
    selected = handle.cache.search(
        source_chunk.routing_gist.k[0], 1, handle.pra_config
    )
    mapped = handle.map_chunk_identities_to_layers(selected, {0, 1})

    for layer_id in (0, 1):
        hit = mapped[layer_id][0][0]
        assert hit.chunk_id == selected[0][0].chunk_id
        assert hit.layer_id == layer_id
        assert hit.chunk is entry.layer_memory[layer_id].chunks[0]
        assert hit.chunk.token_kv.k.shape == (1, 2, 4, 8)
        assert hit.chunk.token_kv.v.shape == (1, 2, 4, 8)
    assert (
        mapped[0][0][0].chunk.token_kv.k.data_ptr()
        != mapped[1][0][0].chunk.token_kv.k.data_ptr()
    )
    assert not torch.equal(
        mapped[0][0][0].chunk.token_kv.k,
        mapped[1][0][0].chunk.token_kv.k,
    )

    handle.configure_memory_layers({0, 1}, fixed_selections=mapped)
    with torch.no_grad():
        handle.model(torch.tensor([[1, 4, 8, 12]]), use_cache=False)
    expected_tokens = sum(hit.selected_token_count for hit in selected[0])
    assert all(
        handle.diagnostics_by_layer()[layer]["retrieved_physical_kv_tokens"]
        == expected_tokens
        for layer in (0, 1)
    )


def test_qwen_implicit_head_preserves_offsets_and_native_bound():
    torch.manual_seed(109)
    handle = inject_pra(_tiny_qwen(), _hf_config())
    prompt = torch.arange(1, 41).remainder(65).unsqueeze(0)
    prepared = handle.prepare_long_prompt(prompt)

    assert prepared.head_tokens == 24
    assert prepared.input_ids.shape == (1, 16)
    assert prepared.position_ids.tolist() == [list(range(24, 40))]
    assert handle.cache.has("#__head")
    head = handle.cache.get("#__head")
    assert head is not None
    chunks = head.layer_memory[1].chunks
    assert all(
        chunk.metadata["routing_representation"] == ATTENTION_INPUT_HIDDEN_STATE
        for chunk in chunks
    )
    assert all(chunk.token_kv.position_state == "post_position" for chunk in chunks)
    assert [chunk.logical_start for chunk in chunks] == [0, 4, 8, 12, 16, 20]
    assert handle.max_native_operation_tokens <= 16
    assert handle.native_limit_violations == 0

    handle.set_memory_enabled(True)
    with torch.no_grad():
        result = handle.model(
            input_ids=prepared.input_ids,
            attention_mask=prepared.attention_mask,
            position_ids=prepared.position_ids,
            use_cache=False,
        )
    assert result.logits.shape == (1, 16, 67)


def test_qwen_full_native_reference_matches_the_same_visible_prefix_logits():
    """Permanent E0 gate: native K/V must preserve ordinary-prefix semantics."""

    torch.manual_seed(110)
    visible = _tiny_qwen()
    handle = inject_pra(
        copy.deepcopy(visible),
        _hf_config(
            layer_ids=(0, 1),
            routing_chunk_tokens=16,
            max_materialized_memory_tokens=32,
        ),
    )
    reference = torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18]])
    query = torch.tensor([[1, 4, 8, 12]])
    entry = handle.add_reference("mem://prefix-equivalence", reference)
    fixed = {}
    for layer_id in (0, 1):
        fixed[layer_id] = [[
            SelectedChunk(
                entry=entry,
                chunk=chunk,
                reference_score=1.0,
                chunk_score=1.0,
                layer_id=layer_id,
                reference_rank=1,
                rank_within_reference=rank,
            )
            for rank, chunk in enumerate(entry.layer_memory[layer_id].chunks, start=1)
        ]]
    handle.configure_memory_layers({0, 1}, fixed_selections=fixed)
    handle.reset_memory_lifetime_trace()
    positions = torch.arange(reference.shape[1], reference.shape[1] + query.shape[1]).unsqueeze(0)

    with torch.no_grad():
        expected = visible(torch.cat((reference, query), dim=1), use_cache=False).logits[:, -query.shape[1]:]
        actual = handle.model(query, position_ids=positions, use_cache=False).logits

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    for layer_id in (0, 1):
        assert handle.memory_lifetime_by_layer()[layer_id] == ({
            "call_index": 0,
            "query_tokens": 4,
            "local_cache_tokens": 4,
            "active_native_tokens": 8,
        },)


def test_qwen_generation_cache_does_not_duplicate_local_history_with_memory():
    torch.manual_seed(113)
    handle = inject_pra(_tiny_qwen(), _hf_config())
    handle.add_reference("mem://generation", torch.tensor([[21, 22, 23, 24]]))
    handle.set_memory_enabled(True)
    prompt = torch.tensor([[1, 3, 5, 7]])

    with torch.no_grad():
        generated = handle.model.generate(prompt, max_new_tokens=2, do_sample=False)

    assert generated.shape == (1, 6)
    diagnostics = handle.diagnostics_by_layer()[1]
    # The last decode step sees the four prompt tokens plus one generated token.
    assert diagnostics["hf_cache_tokens"] == 5
    assert diagnostics["retrieved_physical_kv_tokens"] == 4
    trace = handle.memory_lifetime_by_layer()[1]
    assert len(trace) == 2
    assert [row["active_native_tokens"] for row in trace] == [4, 4]
    assert [row["query_tokens"] for row in trace] == [4, 1]


def test_hf_layer_selection_rejects_out_of_range_and_dispatches_llama():
    with pytest.raises(ValueError, match="outside"):
        inject_pra(_tiny_qwen(), _hf_config(layer_ids=(5,)))

    llama_config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    llama_config._attn_implementation = "eager"
    llama = transformers.LlamaForCausalLM(llama_config)
    handle = inject_pra(
        llama, _hf_config(layer_ids=(0,), model_max_context_tokens=64)
    )
    assert handle.adapters[0].family == "llama"
