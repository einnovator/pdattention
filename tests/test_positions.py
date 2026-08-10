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
