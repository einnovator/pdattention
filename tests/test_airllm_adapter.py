from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from pra_torch.hf.injection import PRAHFModel

from pra_hf.airllm_adapter import (
    AirLLMPRAAdapter,
    InMemoryAirLLMPRAStore,
    wrap_airllm_hf_model,
    _map_airllm_parameter_name,
)


class FakeAirLLM:
    def __init__(self, layers: int = 4) -> None:
        self.layers = [torch.nn.Identity() for _ in range(layers)]
        self._streamed_indices = list(range(layers))
        self.device = torch.device("cpu")
        self._executor = None

    def run(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value = layer(value)
        return value


def test_layer_streamed_adapter_activates_and_releases_selected_detail() -> None:
    model = FakeAirLLM()
    store = InMemoryAirLLMPRAStore({1: 128, 3: 256})
    adapter = AirLLMPRAAdapter(store, consumer_layers=(1, 3)).bind(model)
    try:
        model.run(torch.ones(1))
    finally:
        adapter.close()
    assert store.active == set()
    assert adapter.summary()["pra_bytes_read"] == 384
    assert [row["event"] for row in adapter.summary()["events"]] == [
        "activate",
        "release",
        "activate",
        "release",
    ]


def test_hot_adapter_reuses_detail_without_second_physical_load() -> None:
    model = FakeAirLLM()
    store = InMemoryAirLLMPRAStore({2: 512})
    adapter = AirLLMPRAAdapter(
        store, consumer_layers=(2,), residency_mode="hot"
    ).bind(model)
    try:
        model.run(torch.ones(1))
        model.run(torch.ones(1))
        assert store.active == {2}
    finally:
        adapter.close()
    assert store.active == set()
    assert store.loads[2] == 1
    assert adapter.summary()["cache_hits"] == 1


def test_close_drains_prefetch_before_releasing_request_detail() -> None:
    model = FakeAirLLM()
    store = InMemoryAirLLMPRAStore({1: 128, 3: 256})
    adapter = AirLLMPRAAdapter(
        store,
        consumer_layers=(1, 3),
        prefetch_mode="independent_parallel",
    ).bind(model)

    model.layers[1](torch.ones(1))
    adapter.close()

    assert store.active == set()
    assert store.loaded == set()


def test_native_capability_is_opt_in() -> None:
    assert AirLLMPRAAdapter().capabilities().integration_level.value == "E0"
    native = AirLLMPRAAdapter(InMemoryAirLLMPRAStore({}), storage_managed=True)
    assert native.capabilities().integration_level.value == "E2"


def test_mlx_airllm_path_is_rejected_for_native_hf_wrapping() -> None:
    with pytest.raises(TypeError, match="HF-backed"):
        wrap_airllm_hf_model(SimpleNamespace(tokenizer=object()))


def test_airllm_checkpoint_keys_only_remap_in_pra_layers() -> None:
    name = "model.layers.7.self_attn.q_proj.weight"
    assert _map_airllm_parameter_name(name, frozenset({7})) == (
        "model.layers.7.self_attn.original_attention.q_proj.weight"
    )
    assert _map_airllm_parameter_name(name, frozenset({6})) == name
    assert _map_airllm_parameter_name("model.layers.7.mlp.up_proj.weight", frozenset({7})) == (
        "model.layers.7.mlp.up_proj.weight"
    )


def test_streamed_hf_handle_accepts_an_explicit_execution_device() -> None:
    handle = object.__new__(PRAHFModel)
    handle._device_override = None
    handle.set_execution_device("cpu")
    assert handle.device == torch.device("cpu")
