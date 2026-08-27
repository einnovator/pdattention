"""Product-facing tests for configuration, router artifacts, and references."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM

from pra_hf import PRAConfig, PRAForCausalLM, PRARouter


class TinyTokenizer:
    """Deterministic tokenizer sufficient for the offline public-API gates."""

    all_special_ids = (0, 1)

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        values = [2 + (ord(char) % 61) for char in text]
        if add_special_tokens:
            values.insert(0, 1)
        return SimpleNamespace(input_ids=torch.tensor([values], dtype=torch.long))

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(int(value)) for value in token_ids)

    def convert_ids_to_tokens(self, token_ids):
        return [str(int(value)) for value in token_ids]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(f"{row['role']}: {row['content']}" for row in messages)


def _model():
    config = LlamaConfig(
        vocab_size=67,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=66,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config).eval()


def _config(**overrides):
    values = {
        "routing_layer": -1,
        "consumption_layers": (-1,),
        "chunk_tokens": 4,
        "selected_fraction": 0.5,
        "max_direct_context": 16,
        "native_operation_limit": 64,
        "max_materialized_tokens": 16,
        "context_safety_reserve_tokens": 0,
        "encoding_block_tokens": 16,
    }
    values.update(overrides)
    return PRAConfig(**values)


def test_product_config_serializes_and_fraction_precedes_top_k(tmp_path):
    config = _config(selected_fraction=0.25, top_k=99)
    config.save_pretrained(tmp_path)
    restored = PRAConfig.from_pretrained(tmp_path)

    assert restored == config
    assert restored.selection_policy == "selected_fraction"
    assert _config(selected_fraction=None, top_k=3).selection_policy == "top_k"


def test_router_hf_style_artifact_round_trip(tmp_path):
    torch.manual_seed(301)
    router = PRARouter(
        32,
        8,
        metadata={
            "base_model": "offline/tiny-llama",
            "model_family": "llama",
            "training_datasets": ["QASPER"],
        },
    )
    router.save_pretrained(tmp_path)
    restored = PRARouter.from_pretrained(tmp_path)

    assert restored.metadata["model_family"] == "llama"
    assert restored.parameter_count == 2 * 32 * 8
    assert torch.equal(
        restored.query_projection.weight, router.query_projection.weight
    )
    assert not any(parameter.requires_grad for parameter in restored.parameters())
    assert (tmp_path / "README.md").is_file()


def test_reference_lifecycle_fraction_routing_generation_and_stats(tmp_path):
    torch.manual_seed(302)
    pra = PRAForCausalLM.from_model(
        _model(), TinyTokenizer(), pra_config=_config()
    )
    first = pra.add_reference("abcdefgh")
    path = tmp_path / "facts.txt"
    path.write_text("ijklmnop", encoding="utf-8")
    second = pra.add_reference_file(path)

    assert first.uri.startswith("memory://")
    assert second.uri.startswith("file://")
    result = pra.generate("question", max_new_tokens=2, return_details=True)
    stats = pra.stats()
    assert result.generated_tokens == 2
    assert result.stats["candidate_chunks"] == 4
    assert result.stats["requested_chunks"] == 2
    assert result.stats["requested_chunk_fraction"] == 0.5
    assert result.stats["materialized_kv_token_fraction"] == 0.5
    assert stats["family"] == "llama"
    assert stats["routing_index_bytes"] > 0
    assert stats["resident_detail_kv_bytes"] > stats["routing_index_bytes"]

    pra.remove_reference(first)
    assert len(pra.stats()["references"]) == 1
    pra.clear_references()
    assert pra.stats()["references"] == []


def test_route_only_uses_production_selection_without_generation():
    torch.manual_seed(306)
    pra = PRAForCausalLM.from_model(
        _model(), TinyTokenizer(), pra_config=_config(selected_fraction=None, top_k=2)
    )
    pra.add_reference("abcdefghijklmnop")

    result = pra.route("question")

    assert result.prompt_tokens > 0
    assert result.selected
    assert result.stats["requested_chunks"] == len(result.selected)
    assert result.stats["generation_seconds"] == 0.0


def test_router_cannot_change_after_reference_ingestion(tmp_path):
    pra = PRAForCausalLM.from_model(_model(), TinyTokenizer(), pra_config=_config())
    pra.add_reference("abcdefgh")
    PRARouter(32, 8).save_pretrained(tmp_path)
    with pytest.raises(RuntimeError, match="Clear references"):
        pra.load_router(tmp_path)


def test_iterative_product_path_closes_before_native_memory_is_enabled(monkeypatch):
    torch.manual_seed(303)
    pra = PRAForCausalLM.from_model(
        _model(),
        TinyTokenizer(),
        pra_config=_config(
            routing_mode="iterative",
            routing_depth=2,
            branch_top_k=1,
            beam_size=1,
            max_unique_chunks=2,
            selected_fraction=None,
        ),
    )
    pra.add_reference("abcdefghijklmnop")
    events = []
    original = pra._handle.configure_memory_layers

    def record(layers, *args, **kwargs):
        events.append((set(layers), kwargs.get("fixed_selections")))
        return original(layers, *args, **kwargs)

    monkeypatch.setattr(pra._handle, "configure_memory_layers", record)
    result = pra.generate("question", max_new_tokens=1, return_details=True)

    assert events[0][0] == set()
    assert events[1][0] == set(pra.consumption_layers)
    assert events[1][1] is not None
    assert result.stats["selection_policy"] == "iterative_closure"
    assert result.stats["retrieval_graphs"][0]["nodes"]


def test_hybrid_iterative_product_path_preserves_discovery_materialization_boundary():
    torch.manual_seed(304)
    pra = PRAForCausalLM.from_model(
        _model(),
        TinyTokenizer(),
        pra_config=_config(
            routing_mode="hybrid_iterative",
            routing_depth=2,
            branch_top_k=1,
            beam_size=1,
            max_unique_chunks=2,
            selected_fraction=None,
        ),
    )
    pra.add_reference("alpha points to beta")
    pra.add_reference("beta stores the answer")

    result = pra.generate("find alpha", max_new_tokens=1, return_details=True)
    graph = result.stats["retrieval_graphs"][0]

    assert result.stats["selection_policy"] == "hybrid_iterative_closure"
    assert graph["root"]["discovery_mode"] == "iterative_hybrid"
    assert graph["nodes"]
    assert all(node["discovery_channels"] for node in graph["nodes"])
    assert all(node["materialized"] for node in graph["nodes"] if node["final_selected"])
    assert all(node["materialized"] for node in result.stats["retrieval_graphs"][0]["nodes"])
    assert result.stats["requested_chunks"] <= 2


def test_local_iterative_config_maps_parent_and_local_scales():
    config = PRAConfig(
        routing_mode="local_iterative",
        chunk_tokens=256,
        local_gist_tokens=32,
        consumption_layers=(-1,),
    )
    internal = config.to_internal(28)
    assert internal.routing_chunk_tokens == 256
    assert internal.gist_mode == "segment_mean"
    assert internal.gists_per_chunk == 8
    assert internal.store_associative_gists is True
    assert config.selection_policy == "local_iterative_closure"

    one_shot = PRAConfig(consumption_layers=(-1,)).to_internal(28)
    assert one_shot.store_associative_gists is False


def test_local_iterative_model_requires_asymmetric_router():
    with pytest.raises(ValueError, match="routing adapter"):
        PRAForCausalLM.from_model(
            _model(),
            TinyTokenizer(),
            pra_config=_config(
                routing_mode="local_iterative",
                chunk_tokens=8,
                local_gist_tokens=2,
            ),
        )


def test_local_iterative_product_path_materializes_unique_parents(monkeypatch):
    torch.manual_seed(304)
    router = PRARouter(32, 8, architecture="asymmetric_linear").freeze()
    pra = PRAForCausalLM.from_model(
        _model(),
        TinyTokenizer(),
        pra_config=_config(
            routing_mode="local_iterative",
            chunk_tokens=8,
            local_gist_tokens=2,
            routing_depth=2,
            branch_top_k=1,
            beam_size=1,
            max_unique_chunks=2,
            selected_fraction=None,
        ),
        router=router,
    )
    pra.add_reference("abcdefghijklmnopqrstuvwxyz")
    events = []
    original = pra._handle.configure_memory_layers

    def record(layers, *args, **kwargs):
        events.append((set(layers), kwargs.get("fixed_selections")))
        return original(layers, *args, **kwargs)

    monkeypatch.setattr(pra._handle, "configure_memory_layers", record)
    result = pra.generate("question", max_new_tokens=1, return_details=True)
    graph = result.stats["retrieval_graphs"][0]

    assert events[0][0] == set()
    assert events[1][0] == set(pra.consumption_layers)
    assert result.stats["selection_policy"] == "local_iterative_closure"
    assert result.stats["requested_chunks"] <= 2
    assert len({node["parent_chunk_id"] for node in graph["nodes"]}) <= 2
    assert all(node["materialized"] for node in graph["nodes"])
    assert graph["costs"]["native_qk_comparisons"] == 0
