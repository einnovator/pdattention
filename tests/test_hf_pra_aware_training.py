from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from pra_hf.pra_aware_training import (
    gemma_layer_topology,
    hf_parameter_summary,
    install_hf_adaptation_regime,
)


class Attention(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(self, hidden):
        return self.o_proj(
            self.q_proj(hidden) + self.k_proj(hidden) + self.v_proj(hidden)
        )


class MLP(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.up_proj = nn.Linear(width, width * 2, bias=False)
        self.down_proj = nn.Linear(width * 2, width, bias=False)

    def forward(self, hidden):
        return self.down_proj(torch.relu(self.up_proj(hidden)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()
        self.mlp = MLP()

    def forward(self, hidden):
        return hidden + self.self_attn(hidden) + self.mlp(hidden)


class Host(nn.Module):
    def __init__(self, layers=4):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(Block() for _ in range(layers))

    def forward(self, hidden):
        for layer in self.model.layers:
            hidden = layer(hidden)
        return hidden


def test_gemma_topology_replaces_exact_native_global_slots():
    config = SimpleNamespace(
        layer_types=(
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ),
        sliding_window=4096,
    )
    topology = gemma_layer_topology(config)
    assert tuple(row.layer_index for row in topology if row.pra_enabled) == (2, 5)
    assert topology[0].native_window == 4096
    assert topology[2].converted_attention_type == "pra"


@pytest.mark.parametrize(
    ("placement", "expected"),
    [
        ("every_second_global", (5,)),
        ("late_global", (2, 5)),
        ("all_eligible", (0, 1, 2, 3, 4, 5)),
    ],
)
def test_gemma_topology_compact_ablation_contract(placement, expected):
    config = SimpleNamespace(
        layer_types=("sliding_attention", "sliding_attention", "full_attention")
        * 2,
        sliding_window=128,
    )
    topology = gemma_layer_topology(config, placement)
    assert tuple(row.layer_index for row in topology if row.pra_enabled) == expected


def test_consumer_lora_is_function_preserving_and_excludes_kv():
    torch.manual_seed(7)
    model = Host().eval()
    hidden = torch.randn(2, 3, 8)
    before = model(hidden)
    targets = install_hf_adaptation_regime(
        model,
        "consumer_lora",
        pra_layer_ids=(1, 3),
        lora_rank=2,
        lora_alpha=4,
    )
    torch.testing.assert_close(model(hidden), before)
    assert targets
    assert not any(
        path.endswith(".k_proj") or path.endswith(".v_proj") for path in targets
    )
    assert any(".mlp." in path for path in targets)
    assert 0 < hf_parameter_summary(model)["trainable_fraction"] < 1


def test_interface_and_broad_scopes_expand_in_declared_order():
    interface = Host()
    interface_targets = install_hf_adaptation_regime(
        interface, "interface_lora", pra_layer_ids=(1,), lora_rank=2
    )
    assert any(path.endswith(".k_proj") for path in interface_targets)
    assert all("layers.1." in path for path in interface_targets)

    broad = Host()
    broad_targets = install_hf_adaptation_regime(
        broad, "broad_lora", pra_layer_ids=(1,), lora_rank=2
    )
    assert len(broad_targets) > len(interface_targets)
    assert any("layers.0." in path for path in broad_targets)


def test_frozen_and_full_weight_regimes_expose_opposite_boundaries():
    frozen = Host()
    install_hf_adaptation_regime(frozen, "frozen_pra", pra_layer_ids=(1,))
    assert hf_parameter_summary(frozen)["trainable_parameters"] == 0

    full = Host()
    install_hf_adaptation_regime(full, "full_weight_pra", pra_layer_ids=(1,))
    summary = hf_parameter_summary(full)
    assert summary["trainable_parameters"] == summary["total_parameters"]
