import pytest
import torch

from data.datamodules import PRADataModule
from pra_torch.cache_services import build_cache_from_metadata, collect_reference_metadata, create_cache, create_resolver
from pra_torch.config import CacheServiceConfig, PRAConfig, ResolverServiceConfig
from pra_torch.model import TinyPRAModel


def build_cache_for_example(model, tokenizer, sample, device, *, resolver_config=None, cache_config=None):
    """Test helper for single-sample cache construction."""
    return build_cache_from_metadata(
        model,
        tokenizer,
        [{"references": sample.references}],
        device,
        resolver_config=resolver_config,
        cache_config=cache_config,
    )


def tiny_model(vocab_size: int) -> TinyPRAModel:
    cfg = PRAConfig(vocab_size=vocab_size, max_seq_len=64, d_model=32, n_heads=4, n_layers=2, device="cpu")
    return TinyPRAModel(cfg)


def test_collect_reference_metadata_extracts_documents_and_handles():
    dm = PRADataModule("stage0_synthetic_memory", "data", max_examples=1, batch_size=1, max_seq_len=64).load()
    batch = next(iter(dm.train_loader()))

    documents, summaries, handles = collect_reference_metadata(batch["metadata"])

    assert documents
    assert summaries
    assert handles
    assert all(handle.uri in documents for handle in handles)


def test_build_cache_from_metadata_uses_service_configs():
    dm = PRADataModule("stage0_synthetic_memory", "data", max_examples=1, batch_size=1, max_seq_len=64).load()
    batch = next(iter(dm.train_loader()))
    model = tiny_model(dm.tokenizer.vocab_size)

    cache = build_cache_from_metadata(
        model,
        dm.tokenizer,
        batch["metadata"],
        "cpu",
        resolver_config=ResolverServiceConfig(type="in_memory"),
        cache_config=CacheServiceConfig(type="simple"),
    )

    assert model.pra_cache is cache
    assert len(cache.entries) == len(batch["metadata"][0]["references"])
    assert all(entry.layer_memory for entry in cache.all_entries())


def test_build_cache_for_example_helper_lives_in_tests():
    dm = PRADataModule("stage0_synthetic_memory", "data", max_examples=1, batch_size=1, max_seq_len=64).load()
    model = tiny_model(dm.tokenizer.vocab_size)

    cache = build_cache_for_example(model, dm.tokenizer, dm.dataset[0], "cpu")

    assert cache.entries
    assert model.pra_cache is cache


def test_service_factories_reject_unknown_types():
    with pytest.raises(ValueError):
        create_resolver(ResolverServiceConfig(type="unknown"), {}, {})
    with pytest.raises(ValueError):
        create_cache(CacheServiceConfig(type="unknown"))


def test_cache_builder_accepts_prebuilt_cache_instance():
    dm = PRADataModule("stage0_synthetic_memory", "data", max_examples=1, batch_size=1, max_seq_len=64).load()
    batch = next(iter(dm.train_loader()))
    model = tiny_model(dm.tokenizer.vocab_size)
    prebuilt_cache = create_cache(CacheServiceConfig(type="simple"))

    cache = build_cache_from_metadata(model, dm.tokenizer, batch["metadata"], "cpu", cache=prebuilt_cache)

    assert cache is prebuilt_cache
    first = cache.all_entries()[0]
    assert torch.is_tensor(first.layer_memory[0].chunks[0].routing_gist.k)


def test_cache_builder_can_restore_previous_model_cache_without_attaching_result():
    dm = PRADataModule(
        "stage0_synthetic_memory", "data", max_examples=1, batch_size=1, max_seq_len=64
    ).load()
    batch = next(iter(dm.train_loader()))
    model = tiny_model(dm.tokenizer.vocab_size)
    previous = model.pra_cache

    built = build_cache_from_metadata(
        model,
        dm.tokenizer,
        batch["metadata"],
        "cpu",
        attach_to_model=False,
    )

    assert built is not previous
    assert built.all_entries()
    assert model.pra_cache is previous
