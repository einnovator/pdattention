from __future__ import annotations

from dataclasses import replace

import pytest

from pra_hf.rag_composition import (
    PROFILE_CONTRACTS,
    CompositionReceipt,
    MaterializationMode,
    PositionPolicy,
    RAGPRAProfile,
    SelectedResource,
    SelectorRole,
    compose_resources,
    permutation_orders,
    permute_resources,
)


def _resources() -> tuple[SelectedResource, ...]:
    return (
        SelectedResource("D1", "D1:0", "1" * 64, (10, 11, 12), 1, 0.9),
        SelectedResource("D2", "D2:0", "2" * 64, (40, 41), 2, 0.6),
        SelectedResource("D3", "D3:0", "3" * 64, (100, 101, 102, 103), 3, 0.2),
    )


def _compose(policy: PositionPolicy, resources=None, **kwargs) -> CompositionReceipt:
    return compose_resources(
        resources or _resources(),
        selection_receipt_id="selection-1",
        profile=RAGPRAProfile.RAG_PLUS_PRA_NATIVE_REBOUND,
        position_policy=policy,
        **kwargs,
    )


def test_profile_contracts_cover_the_canonical_rag_pra_design() -> None:
    assert set(PROFILE_CONTRACTS) == set(RAGPRAProfile)
    assert (
        PROFILE_CONTRACTS[RAGPRAProfile.RAG_ONLY_TEXT].materialization
        is MaterializationMode.SELECTED_TEXT
    )
    assert PROFILE_CONTRACTS[
        RAGPRAProfile.RAG_PLUS_PRA_NATIVE_CONTIGUOUS
    ].requires_frozen_external_selection
    assert (
        PROFILE_CONTRACTS[RAGPRAProfile.RAG_PLUS_PRA_SELECTED].default_selector_role
        is SelectorRole.PRA_SECOND_STAGE
    )


@pytest.mark.parametrize("policy", tuple(PositionPolicy))
def test_every_position_policy_preserves_identity_and_internal_geometry(policy) -> None:
    receipt = _compose(policy)
    expected = {row.identity: row.source_positions for row in _resources()}
    assert {
        (row.resource_id, row.chunk_id, row.source_sha256): row.source_positions
        for row in receipt.placements
    } == expected
    for row in receipt.placements:
        assert tuple(
            right - left for left, right in zip(row.source_positions, row.source_positions[1:])
        ) == tuple(
            right - left
            for left, right in zip(row.effective_positions, row.effective_positions[1:])
        )


def test_global_packed_matches_fresh_serialization() -> None:
    receipt = _compose(PositionPolicy.GLOBAL_PACKED)
    assert [row.effective_positions for row in receipt.placements] == [
        (0, 1, 2),
        (3, 4),
        (5, 6, 7, 8),
    ]


def test_source_local_and_resource_adjacent_have_expected_geometry() -> None:
    source = _compose(PositionPolicy.SOURCE_LOCAL)
    assert all(row.source_positions == row.effective_positions for row in source.placements)

    adjacent = _compose(PositionPolicy.RESOURCE_ADJACENT, query_position=128, near_gap=4)
    assert {row.effective_positions[-1] for row in adjacent.placements} == {123}
    assert len(
        {position for row in adjacent.placements for position in row.effective_positions}
    ) < sum(len(row.effective_positions) for row in adjacent.placements)


def test_near_bands_do_not_overlap_and_rank_distance_places_rank_one_nearest() -> None:
    bands = _compose(PositionPolicy.NON_OVERLAPPING_NEAR_BANDS)
    positions = [set(row.effective_positions) for row in bands.placements]
    assert all(not left.intersection(right) for index, left in enumerate(positions) for right in positions[index + 1 :])

    ranked = _compose(PositionPolicy.RANK_DISTANCE)
    by_rank = sorted(ranked.placements, key=lambda row: row.rank)
    assert by_rank[0].effective_positions[-1] > by_rank[1].effective_positions[-1]
    assert by_rank[1].effective_positions[-1] > by_rank[2].effective_positions[-1]


def test_score_and_random_policies_are_deterministic() -> None:
    assert _compose(PositionPolicy.SCORE_DISTANCE).receipt_id == _compose(
        PositionPolicy.SCORE_DISTANCE
    ).receipt_id
    first = _compose(PositionPolicy.RANDOM_DISTANCE, random_seed=17)
    second = _compose(PositionPolicy.RANDOM_DISTANCE, random_seed=17)
    changed = _compose(PositionPolicy.RANDOM_DISTANCE, random_seed=19)
    assert first.receipt_id == second.receipt_id
    assert first.receipt_id != changed.receipt_id


def test_d1_d2_permutation_changes_order_but_not_resources() -> None:
    original = _resources()[:2]
    reversed_resources = permute_resources(original, ("D2", "D1"))
    assert tuple(row.resource_id for row in reversed_resources) == ("D2", "D1")
    assert {row.identity for row in original} == {row.identity for row in reversed_resources}
    assert _compose(PositionPolicy.GLOBAL_PACKED, original).receipt_id != _compose(
        PositionPolicy.GLOBAL_PACKED, reversed_resources
    ).receipt_id


def test_permutation_orders_include_canonical_reverse_and_seeded_variants() -> None:
    first = permutation_orders(("D1", "D2", "D3"), seed=17, max_random=3)
    second = permutation_orders(("D1", "D2", "D3"), seed=17, max_random=3)
    assert first == second


def test_permutation_orders_can_disable_random_variants() -> None:
    assert permutation_orders(("D1", "D2", "D3"), max_random=0) == (
        ("D1", "D2", "D3"),
        ("D3", "D2", "D1"),
    )
    assert first[:2] == (("D1", "D2", "D3"), ("D3", "D2", "D1"))
    assert len(first) == 5
    assert len(set(first)) == len(first)


def test_composition_receipt_roundtrip_and_tamper_detection() -> None:
    receipt = _compose(PositionPolicy.GLOBAL_PACKED)
    restored = CompositionReceipt.from_dict(receipt.to_dict())
    assert restored.receipt_id == receipt.receipt_id

    changed = receipt.to_dict()
    changed["query_position"] = receipt.query_position + 1
    with pytest.raises(ValueError, match="digest"):
        CompositionReceipt.from_dict(changed)

    with pytest.raises(ValueError, match="internal geometry"):
        replace(receipt.placements[0], effective_positions=(0, 2, 3))


def test_invalid_permutation_and_duplicate_identity_are_rejected() -> None:
    resources = _resources()
    with pytest.raises(ValueError, match="exactly once"):
        permute_resources(resources, ("D1", "D2", "missing"))
    with pytest.raises(ValueError, match="unique"):
        compose_resources(
            (resources[0], resources[0]),
            selection_receipt_id="selection-1",
            profile=RAGPRAProfile.RAG_PLUS_PRA_NATIVE_REBOUND,
            position_policy=PositionPolicy.GLOBAL_PACKED,
        )
