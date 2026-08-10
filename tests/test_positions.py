import pytest
import torch

from pra_torch.config import PRAConfig
from pra_torch.attention import PRAttention
from pra_torch.memory import PRASimpleMemoryCache
from pra_torch.model import (
    PositionAwareTransformerBlock,
    TinyPRAModel,
    convert_sa_model_to_pra,
)
from pra_torch.positions import RotaryPositionEncoding
from experiments.paper1_5_rope.instrumented_model import (
    capture_self_attention,
    materialize_raw_rope_key,
)
from experiments.paper1_5_rope.position_policies import materialization_positions


def test_rope_position_zero_is_identity():
    rope = RotaryPositionEncoding(head_dim=8)
    tensor = torch.randn(2, 3, 1, 8)

    actual = rope.apply_rotary(tensor, torch.zeros(2, 1, dtype=torch.long))

    torch.testing.assert_close(actual, tensor)


def test_rope_attention_logits_are_invariant_to_common_translation():
    torch.manual_seed(7)
    rope = RotaryPositionEncoding(head_dim=8)
    query = torch.randn(2, 3, 5, 8)
    key = torch.randn(2, 3, 5, 8)
    positions = torch.arange(5)

    q0, k0 = rope.transform_qk(query, key, positions)
    q1, k1 = rope.transform_qk(query, key, positions + 10_000)

    torch.testing.assert_close(
        q0 @ k0.transpose(-2, -1),
        q1 @ k1.transpose(-2, -1),
        rtol=2e-4,
        atol=2e-4,
    )


def test_rope_preserves_vector_norm():
    rope = RotaryPositionEncoding(head_dim=8)
    tensor = torch.randn(2, 3, 5, 8)

    rotated = rope.apply_rotary(tensor, torch.arange(5) + 317)

    torch.testing.assert_close(rotated.norm(dim=-1), tensor.norm(dim=-1))


def test_rope_model_accepts_logical_offsets_beyond_native_context():
    cfg = PRAConfig(
        vocab_size=31,
        d_model=16,
        n_heads=2,
        n_layers=2,
        max_seq_len=8,
        model_variant="td_sa",
        position_encoding="rope",
    )
    model = TinyPRAModel(cfg).eval()

    logits = model(torch.tensor([[1, 2, 3, 4]]), position_offset=50_000)

    assert logits.shape == (1, 4, 31)
    assert all(isinstance(block, PositionAwareTransformerBlock) for block in model.blocks)


def test_absolute_model_still_rejects_offsets_beyond_position_table():
    cfg = PRAConfig(
        vocab_size=31,
        d_model=16,
        n_heads=2,
        n_layers=1,
        max_seq_len=8,
        model_variant="td_sa",
    )

    with pytest.raises(ValueError, match="positional table"):
        TinyPRAModel(cfg)(torch.tensor([[1, 2, 3, 4]]), position_offset=8)


def test_absolute_position_capacity_is_independent_of_native_operation_limit():
    cfg = PRAConfig(
        vocab_size=31,
        d_model=16,
        n_heads=2,
        n_layers=1,
        max_seq_len=16,
        model_max_context_tokens=4,
        model_variant="td_sa",
    )
    model = TinyPRAModel(cfg).eval()

    logits = model(torch.tensor([[1, 2, 3, 4]]), position_offset=12)

    assert logits.shape == (1, 4, 31)
    with pytest.raises(ValueError, match="positional table"):
        model(torch.tensor([[1, 2, 3, 4]]), position_offset=13)


def test_rope_sa_to_pra_conversion_preserves_no_memory_logits():
    torch.manual_seed(11)
    source_cfg = PRAConfig(
        vocab_size=37,
        d_model=24,
        n_heads=3,
        n_layers=2,
        max_seq_len=16,
        model_variant="td_sa",
        position_encoding="rope",
    )
    target_cfg = PRAConfig(
        **{
            **source_cfg.__dict__,
            "model_variant": "td_pra",
        }
    )
    source = TinyPRAModel(source_cfg).eval()
    converted = convert_sa_model_to_pra(source, target_cfg).eval()
    input_ids = torch.randint(0, source_cfg.vocab_size, (3, 12))

    with torch.no_grad():
        expected = source(input_ids, position_offset=torch.tensor([0, 20, 200]))
        actual = converted(
            input_ids,
            use_pra_memory=False,
            position_offset=torch.tensor([0, 20, 200]),
        )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_rope_rejects_odd_head_width():
    with pytest.raises(ValueError, match="even"):
        PRAConfig(
            d_model=15,
            n_heads=3,
            n_layers=1,
            position_encoding="rope",
        )


def test_rope_cache_keys_record_post_position_state_and_positions():
    cfg = PRAConfig(
        d_model=16,
        n_heads=2,
        n_layers=1,
        max_seq_len=8,
        model_variant="td_pra",
        position_encoding="rope",
    )
    attention = PRAttention(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        max_seq_len=cfg.max_seq_len,
        layer_id=0,
        pra_cache=PRASimpleMemoryCache(),
        config=cfg,
    )
    hidden = torch.randn(1, 4, cfg.d_model)
    positions = torch.arange(4) + 10_000

    cached = attention.project_kv(hidden, detach=False, position_ids=positions)
    raw_key = attention.split_heads(attention.k_proj(hidden))
    expected = attention.position_encoding.apply_rotary(raw_key, positions)

    assert cached.position_state == "post_position"
    torch.testing.assert_close(cached.position_ids, positions)
    torch.testing.assert_close(cached.k, expected)
    assert cached.k.shape == (1, cfg.n_heads, 4, cfg.d_model // cfg.n_heads)


def test_experimental_position_policies_preserve_chunk_order_and_spacing():
    source = torch.tensor([100, 101, 102, 103])
    expected_spacing = source[1:] - source[:-1]

    for policy in (
        "exact_logical",
        "local_chunk",
        "clipped",
        "log_compressed",
        "bucketed",
        "remote_past",
    ):
        assigned = materialization_positions(
            source,
            query_position=10_000,
            policy=policy,
            distance_limit=192,
        )
        torch.testing.assert_close(assigned[1:] - assigned[:-1], expected_spacing)
        assert int(assigned[-1]) < 10_000

    torch.testing.assert_close(
        materialization_positions(
            source,
            query_position=10_000,
            policy="exact_logical",
            distance_limit=192,
        ),
        source,
    )
    assert int(
        materialization_positions(
            source,
            query_position=10_000,
            policy="local_chunk",
            distance_limit=192,
        )[-1]
    ) == 9_999
    assert int(
        materialization_positions(
            source,
            query_position=10_000,
            policy="clipped",
            distance_limit=192,
        )[-1]
    ) == 10_000 - 192


def test_deferred_rope_materialization_matches_post_position_key():
    torch.manual_seed(13)
    raw_key = torch.randn(1, 2, 7, 8)
    positions = torch.arange(7) + 300
    rope = RotaryPositionEncoding(8)

    post_key = rope.apply_rotary(raw_key, positions)
    deferred_key, assigned = materialize_raw_rope_key(
        raw_key,
        positions,
        query_position=1_000,
        policy="exact_logical",
        distance_limit=192,
        rope=rope,
    )

    torch.testing.assert_close(deferred_key, post_key)
    torch.testing.assert_close(assigned, positions)


def test_instrumented_attention_exposes_shapes_and_preserves_causality():
    cfg = PRAConfig(
        vocab_size=31,
        d_model=16,
        n_heads=2,
        n_layers=1,
        max_seq_len=8,
        model_variant="td_sa",
        position_encoding="rope",
        dropout=0.0,
    )
    attention = PositionAwareTransformerBlock(cfg, 0).attn.eval()
    hidden = torch.randn(2, 5, cfg.d_model)
    capture = capture_self_attention(attention, hidden, torch.arange(5) + 100)

    assert capture.q_raw.shape == (2, 2, 5, 8)
    assert capture.k_positioned.shape == (2, 2, 5, 8)
    assert capture.value.shape == (2, 2, 5, 8)
    assert capture.attention_logits.shape == (2, 2, 5, 5)
    assert capture.output.shape == hidden.shape
    assert torch.isneginf(capture.attention_logits[..., 0, 1:]).all()
    assert torch.count_nonzero(capture.attention_probabilities[..., 0, 1:]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_deferred_rope_cpu_gpu_residency_parity():
    torch.manual_seed(17)
    raw_key = torch.randn(1, 2, 11, 8)
    positions = torch.arange(11) + 20_000
    cpu_rope = RotaryPositionEncoding(8)
    gpu_rope = RotaryPositionEncoding(8).cuda()

    for policy in ("exact_logical", "clipped", "bucketed"):
        cpu_key, cpu_positions = materialize_raw_rope_key(
            raw_key,
            positions,
            query_position=30_000,
            policy=policy,
            distance_limit=192,
            rope=cpu_rope,
        )
        gpu_key, gpu_positions = materialize_raw_rope_key(
            raw_key.cuda(),
            positions.cuda(),
            query_position=30_000,
            policy=policy,
            distance_limit=192,
            rope=gpu_rope,
        )
        torch.testing.assert_close(gpu_key.cpu(), cpu_key, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(gpu_positions.cpu(), cpu_positions)
