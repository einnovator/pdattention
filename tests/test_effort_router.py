from __future__ import annotations

import torch

from pra_hf import (
    AutoregressiveEffortRouter,
    HashingQueryEncoder,
    MultiHeadEffortRouter,
    RouterActionSpace,
    default_effort_profiles,
    profile_actions,
)


def _targets(space: RouterActionSpace, profile_index: int, batch: int = 3) -> dict[str, torch.Tensor]:
    indexed = space.index_targets(profile_actions(default_effort_profiles()[profile_index]))
    return {name: torch.full((batch,), index, dtype=torch.long) for name, index in indexed.items()}


def test_action_space_is_derived_from_supported_profiles() -> None:
    profiles = default_effort_profiles()
    space = RouterActionSpace.from_profiles(profiles, core_only=True)
    assert [field.name for field in space.fields][:2] == [
        "query_region_policy",
        "facet_policy",
    ]
    assert space.field("neighbors").values == (2, 4, 8)
    assert space.index_targets(profile_actions(profiles[1]))["hops"] == 1


def test_independent_multi_head_router_has_one_categorical_head_per_field() -> None:
    torch.manual_seed(7)
    space = RouterActionSpace.from_profiles(default_effort_profiles(), core_only=True)
    router = MultiHeadEffortRouter(5, space, hidden_width=12)
    features = torch.randn(3, 5)
    targets = _targets(space, 1)
    logits = router(features)
    assert set(logits) == {field.name for field in space.fields}
    assert logits["retained_roots"].shape == (3, 3)
    assert torch.isfinite(router.loss(features, targets))


def test_semantic_router_requires_and_uses_encoder_state() -> None:
    torch.manual_seed(11)
    space = RouterActionSpace.from_profiles(default_effort_profiles(), core_only=True)
    encoder = HashingQueryEncoder(width=16)
    semantic = encoder.encode(["direct question", "bridge relation"])
    assert torch.equal(semantic, encoder.encode(["direct question", "bridge relation"]))
    router = MultiHeadEffortRouter(
        4,
        space,
        semantic_width=16,
        hidden_width=12,
        architecture="R2_encoder_mlp",
    )
    logits = router(torch.randn(2, 4), semantic)
    assert logits["query_region_policy"].shape == (2, 3)


def test_autoregressive_router_conditions_heads_without_cartesian_classes() -> None:
    torch.manual_seed(13)
    space = RouterActionSpace.from_profiles(default_effort_profiles(), core_only=True)
    router = AutoregressiveEffortRouter(6, space, hidden_width=12, context_width=5)
    features = torch.randn(3, 6)
    targets = _targets(space, 2)
    loss = router.loss(features, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in router.parameters())
    decision = router.decide(features[0])
    assert set(decision.actions) == {field.name for field in space.fields}
    assert decision.architecture == "R3A_autoregressive"
