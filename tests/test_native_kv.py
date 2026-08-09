import warnings

import pytest
import torch

from pra_torch.attention import PRAttention
from pra_torch.config import PRAConfig
from pra_torch.memory import (
    ChunkRoutingGist,
    LayerKV,
    LayerReferenceMemory,
    PRACacheEntry,
    PRASimpleMemoryCache,
    ReferenceChunkMemory,
)
from pra_torch.model import TinyPRAModel, convert_sa_model_to_pra
from pra_torch.pra_train import native_kv_gap_metrics
from pra_torch.native_metrics import recovered_context_benefit


def _config(**overrides):
    values = {
        "vocab_size": 31,
        "d_model": 16,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 32,
        "max_seq_len": 16,
        "dropout": 0.0,
        "trigger_threshold": -1.0,
    }
    values.update(overrides)
    return PRAConfig(**values)


def test_native_kv_is_default_and_has_no_transport_parameters():
    attention = PRAttention(
        16, 4, 16, 0, PRASimpleMemoryCache(), config=_config(n_layers=1)
    )

    assert attention.memory_transport == "native_kv"
    assert attention.mem_o_proj is None
    assert not any("mem_o_proj" in name for name, _parameter in attention.named_parameters())


def test_historical_native_kv_matches_full_causal_attention_tail():
    torch.manual_seed(11)
    attention = PRAttention(
        16, 4, 16, 0, PRASimpleMemoryCache(), config=_config(n_layers=1)
    ).eval()
    hidden = torch.randn(2, 7, 16)
    prefix_length = 4

    full = attention(hidden, use_pra_memory=False)[:, prefix_length:]
    prefix = hidden[:, :prefix_length]
    memory_k = attention.split_heads(attention.k_proj(prefix))
    memory_v = attention.split_heads(attention.v_proj(prefix))
    tail = attention.forward_native_kv(
        hidden[:, prefix_length:],
        [memory_k[row : row + 1] for row in range(hidden.shape[0])],
        [memory_v[row : row + 1] for row in range(hidden.shape[0])],
    )

    assert torch.allclose(tail, full, atol=1e-6, rtol=1e-6)


def test_sa_checkpoint_conversion_preserves_disabled_memory_logits():
    torch.manual_seed(23)
    source = TinyPRAModel(_config(model_variant="td_sa")).eval()
    target_config = _config(model_variant="td_pra", memory_transport="native_kv")

    converted = convert_sa_model_to_pra(source, target_config).eval()
    input_ids = torch.randint(0, source.cfg.vocab_size, (3, 9))

    with torch.no_grad():
        expected = source(input_ids, use_pra_memory=False)
        actual = converted(input_ids, use_pra_memory=False)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_legacy_cross_attention_flag_preserves_adapted_transport():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        config = _config(n_layers=1, use_cross_attention_memory=True)
    attention = PRAttention(16, 4, 16, 0, PRASimpleMemoryCache(), config=config)

    assert config.memory_transport == "cross_attention"
    assert attention.mem_o_proj is not None
    assert any("mem_o_proj" in name for name, _parameter in attention.named_parameters())


def test_invalid_transport_is_rejected():
    with pytest.raises(ValueError, match="memory_transport"):
        _config(memory_transport="new_branch")


def test_native_attention_reports_active_kv_and_transfer_volume():
    cache = PRASimpleMemoryCache()
    key = torch.randn(1, 4, 3, 4)
    value = torch.randn_like(key)
    gist = torch.randn(16)
    cache.put(
        PRACacheEntry(
            uri="mem://history",
            text="three historical positions",
            layer_memory={
                0: LayerReferenceMemory(
                    chunks=[
                        ReferenceChunkMemory(
                            chunk_id="history-0",
                            source_uri="mem://history",
                            token_start=0,
                            token_end=3,
                            token_kv=LayerKV(k=key, v=value),
                            routing_gist=ChunkRoutingGist(k=gist),
                        )
                    ]
                )
            },
        )
    )
    attention = PRAttention(16, 4, 16, 0, cache, config=_config(n_layers=1)).eval()

    output = attention(torch.randn(1, 5, 16))
    diagnostics = attention.last_diagnostics

    assert output.shape == (1, 5, 16)
    assert diagnostics["active_local_tokens"] == 5.0
    assert diagnostics["retrieved_token_kv"] == 3.0
    assert diagnostics["accessible_kv_tokens"] == 8.0
    assert diagnostics["active_memory_fraction"] == pytest.approx(3 / 8)
    assert diagnostics["retrieved_kv_storage_bytes"] == 2 * 3 * 16 * 4
    assert diagnostics["retrieved_kv_transfer_bytes"] == diagnostics[
        "retrieved_kv_storage_bytes"
    ]


def test_native_gap_metrics_keep_transport_and_retrieval_effects_separate():
    metrics = native_kv_gap_metrics(
        [
            {"condition": "sa_full", "loss": 2.0},
            {"condition": "sa_tail", "loss": 3.0},
            {"condition": "native_all", "loss": 2.1},
            {"condition": "native_oracle", "loss": 2.2},
            {"condition": "valid", "loss": 2.3},
            {"condition": "native_shuffled", "loss": 2.6},
            {"condition": "native_disabled", "loss": 3.0},
        ]
    )

    assert metrics["transport_gap"] == pytest.approx(0.1)
    assert metrics["sparse_gap"] == pytest.approx(0.1)
    assert metrics["routing_gap"] == pytest.approx(0.1)
    assert metrics["memory_benefit_oracle"] == pytest.approx(0.8)
    assert metrics["content_causality_oracle"] == pytest.approx(0.4)
    assert metrics["dependency_gain"] == pytest.approx(1.0)
    assert metrics["rcb_oracle"] == pytest.approx(0.8)


def test_recovered_context_benefit_marks_low_dependency_targets_undefined():
    assert recovered_context_benefit(
        sa_full_loss=1.0, sa_tail_loss=1.0, pra_loss=0.9
    ) is None
    assert recovered_context_benefit(
        sa_full_loss=1.0, sa_tail_loss=2.0, pra_loss=0.5
    ) == pytest.approx(1.5)
