from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from pra_hf import PRAConfig, PRAForCausalLM, PRAMemoryAdapter


class TinyTokenizer:
    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        values = [2 + (ord(char) % 61) for char in text]
        if add_special_tokens:
            values.insert(0, 1)
        return SimpleNamespace(input_ids=torch.tensor([values], dtype=torch.long))

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(int(value)) for value in token_ids)


def _model():
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


def _config():
    return PRAConfig(
        routing_layer=1,
        consumption_layers=(0, 1),
        chunk_tokens=4,
        selected_fraction=0.5,
        max_direct_context=16,
        native_operation_limit=64,
        max_materialized_tokens=16,
        context_safety_reserve_tokens=0,
        encoding_block_tokens=16,
    )


def _artifact() -> PRAMemoryAdapter:
    pra = PRAForCausalLM.from_model(_model(), TinyTokenizer(), pra_config=_config())
    pra._handle.configure_late_band_lora(2, alpha=2.0, dropout=0.0)
    state = {
        name: parameter.detach().fill_(0.125).clone()
        for name, parameter in pra.model.named_parameters()
        if name.startswith(PRAMemoryAdapter.STATE_PREFIX) and parameter.requires_grad
    }
    return PRAMemoryAdapter(
        rank=2,
        alpha=2.0,
        dropout=0.0,
        layer_ids=(0, 1),
        state_dict=state,
        metadata={
            "base_model": "offline/tiny-qwen",
            "model_family": "qwen",
            "training_seed": 11,
        },
    )


def test_memory_adapter_round_trip_and_public_model_load(tmp_path):
    artifact = _artifact()
    artifact.save_pretrained(tmp_path)
    restored = PRAMemoryAdapter.from_pretrained(tmp_path)
    pra = PRAForCausalLM.from_model(
        _model(),
        TinyTokenizer(),
        pra_config=_config(),
        memory_adapter=restored,
    )

    assert restored.parameter_count == 2 * (32 * 2 + 2 * 32)
    assert pra.stats()["memory_adapter_parameters"] == restored.parameter_count
    loaded = {
        name: parameter
        for name, parameter in pra.model.named_parameters()
        if name in restored.state_dict
    }
    assert set(loaded) == set(restored.state_dict)
    assert all(torch.equal(value, restored.state_dict[name]) for name, value in loaded.items())
    assert not any(value.requires_grad for value in loaded.values())
    assert (tmp_path / "README.md").is_file()


def test_memory_adapter_load_requires_empty_cache_and_matching_layers(tmp_path):
    artifact = _artifact()
    artifact.save_pretrained(tmp_path)
    pra = PRAForCausalLM.from_model(_model(), TinyTokenizer(), pra_config=_config())
    pra.add_reference("abcdefgh")
    with pytest.raises(RuntimeError, match="Clear references"):
        pra.load_memory_adapter(tmp_path)

    mismatch = PRAMemoryAdapter(
        rank=artifact.rank,
        alpha=artifact.alpha,
        dropout=artifact.dropout,
        layer_ids=(1,),
        state_dict=artifact.state_dict,
    )
    with pytest.raises(ValueError, match="expects layers"):
        PRAForCausalLM.from_model(
            _model(), TinyTokenizer(), pra_config=_config(), memory_adapter=mismatch
        )


def test_experiment_checkpoint_rejects_residual_combinations(tmp_path):
    checkpoint = tmp_path / "combo.pt"
    torch.save(
        {
            "variant": {"lora_rank": 8, "residual_width": 32},
            "seed": 11,
            "state_dict": {"pra_late_band_lora.weight": torch.ones(1)},
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="LoRA-only"):
        PRAMemoryAdapter.from_experiment_checkpoint(checkpoint, layer_ids=(14, 27))
