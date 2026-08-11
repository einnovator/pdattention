"""Tests for the tiny frozen-backbone routing projection."""

import pytest
import torch

from pra_torch.hf import HFRoutingProjection, load_hf_routing_projection


@pytest.mark.parametrize(
    ("architecture", "parameters"),
    (("shared_linear", 32 * 8), ("asymmetric_linear", 2 * 32 * 8)),
)
def test_linear_routing_projection_shapes_normalization_and_parameter_count(
    architecture, parameters
):
    adapter = HFRoutingProjection(32, 8, architecture)
    query = adapter.project_query(torch.randn(2, 32))
    memory = adapter.project_memory(torch.randn(5, 32))
    assert query.shape == (2, 8)
    assert memory.shape == (5, 8)
    assert torch.allclose(query.norm(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.allclose(memory.norm(dim=-1), torch.ones(5), atol=1e-6)
    assert adapter.scores(torch.randn(2, 32), torch.randn(5, 32)).shape == (2, 5)
    assert adapter.parameter_count == parameters


def test_shared_projection_reuses_exact_module_and_asymmetric_does_not():
    shared = HFRoutingProjection(8, 4, "shared_linear")
    asymmetric = HFRoutingProjection(8, 4, "asymmetric_linear")
    assert shared.query_projection is shared.memory_projection
    assert asymmetric.query_projection is not asymmetric.memory_projection


def test_mlp_projection_and_invalid_inputs():
    adapter = HFRoutingProjection(16, 4, "asymmetric_mlp")
    assert adapter.scores(torch.randn(1, 16), torch.randn(3, 16)).shape == (1, 3)
    with pytest.raises(ValueError, match="shape"):
        adapter.project_query(torch.randn(1, 15))
    with pytest.raises(ValueError, match="Unsupported"):
        HFRoutingProjection(16, 4, "bilinear")


def test_routing_projection_checkpoint_restores_frozen_parameters(tmp_path):
    original = HFRoutingProjection(16, 4, "shared_linear")
    path = tmp_path / "projection.pt"
    torch.save(
        {
            "state_dict": original.state_dict(),
            "input_width": 16,
            "routing_width": 4,
            "architecture": "shared_linear",
        },
        path,
    )
    restored = load_hf_routing_projection(path)
    assert torch.equal(
        restored.query_projection.weight,
        original.query_projection.weight,
    )
    assert not any(parameter.requires_grad for parameter in restored.parameters())
