import pytest
import torch

from experiments.paper2_5_iterative_pra.run_chunk_granularity import _atomic_native
from pra_hf.chunk_granularity import (
    chunk_spans,
    contracted_chain_depth,
    evaluate_oracle_recovery,
    evidence_topology,
    facet_parent_statistics,
    incremental_facet_coverage,
    minimum_recovery_depth,
    normalize_facet_scores,
    path_facet_coverage,
)


def test_rechunking_is_deterministic_and_uses_exact_half_open_boundaries():
    expected = ((0, 4), (3, 7), (6, 10))
    assert chunk_spans(10, 4, overlap=1) == expected
    assert chunk_spans(10, 4, overlap=1) == expected
    assert chunk_spans(8, 4) == ((0, 4), (4, 8))


@pytest.mark.parametrize("size,overlap", [(0, 0), (4, -1), (4, 4)])
def test_rechunking_rejects_invalid_geometry(size, overlap):
    with pytest.raises(ValueError):
        chunk_spans(10, size, overlap)


def test_evidence_mapping_root_containment_and_group_collisions():
    topology = evidence_topology(
        16,
        [(1, 3), (3, 6), (12, 14)],
        chunk_size=4,
    )
    assert topology.evidence_parent_groups == ((0,), (0, 1), (3,))
    assert topology.oracle_parent_ids == (0, 1, 3)
    assert topology.later_oracle_parent_ids == (1, 3)
    assert topology.root_parent_id == 0
    assert topology.evidence_group_collisions == 1
    assert topology.root_contains_multiple_groups
    assert not topology.root_contains_only_initial_evidence
    assert topology.evidence_tokens_per_parent == (3, 2, 0, 2)
    assert topology.root_oracle_fraction == pytest.approx(3 / 7)


def test_root_can_contain_all_evidence_or_only_initial_group():
    all_in_root = evidence_topology(12, [(1, 2), (3, 5)], chunk_size=6)
    initial_only = evidence_topology(12, [(1, 2), (8, 10)], chunk_size=6)
    assert all_in_root.root_contains_all_evidence
    assert all_in_root.root_contains_multiple_groups
    assert initial_only.root_contains_only_initial_evidence
    assert initial_only.later_oracle_parent_ids == (1,)


def test_oracle_evaluation_is_post_hoc_and_minimum_depth_is_exact():
    topology = evidence_topology(12, [(1, 2), (8, 10)], chunk_size=4)
    partial = evaluate_oracle_recovery([0], topology)
    complete = evaluate_oracle_recovery([0, 2], topology)
    assert partial.oracle_recall == 0.5
    assert partial.later_oracle_recall == 0.0
    assert not partial.complete_oracle
    assert complete.complete_oracle
    assert minimum_recovery_depth({0: [0], 1: [0, 1], 2: [0, 1, 2]}, topology) == 2
    assert minimum_recovery_depth({0: [0], 1: [0, 1]}, topology) is None


def test_facet_parent_matrix_statistics_and_complementary_coverage():
    scores = torch.tensor([[0.0, 1.0, 0.5], [1.0, 0.0, 0.5]])
    stats = facet_parent_statistics(scores)
    assert stats[0].winning_facet == 1
    assert stats[1].winning_facet == 0
    assert stats[2].normalized_entropy == pytest.approx(1.0)
    normalized = normalize_facet_scores(scores)
    assert path_facet_coverage(normalized, [0]) == pytest.approx(0.5)
    assert incremental_facet_coverage(normalized, [0], 1) == pytest.approx(0.5)
    assert incremental_facet_coverage(normalized, [0, 1], 2) == pytest.approx(0.0)


def test_controlled_chain_contraction_reduces_observed_depth():
    assert [contracted_chain_depth(8, group) for group in (1, 2, 4)] == [8, 4, 2]
    assert contracted_chain_depth(1, 2) == 0


def test_native_windows_split_exactly_at_16_and_preserve_canonical_32():
    query = torch.arange(32 * 2 * 3).reshape(1, 32, 2, 3)
    key = torch.arange(32 * 1 * 3).reshape(1, 32, 1, 3)
    mask = torch.ones(1, 32, dtype=torch.bool)
    feature = {
        "local_spans": [(0, 32)],
        "local_pre_query": query,
        "local_pre_key": key,
        "local_token_mask": mask,
    }
    q16, k16, m16, parents16 = _atomic_native(feature, 16)
    q32, k32, m32, parents32 = _atomic_native(feature, 32)
    assert q16.shape == (2, 16, 2, 3)
    assert torch.equal(q16.reshape_as(query), query)
    assert torch.equal(k16.reshape_as(key), key)
    assert m16.all()
    assert parents16.tolist() == [0, 1]
    assert q32.data_ptr() == query.data_ptr()
    assert k32.data_ptr() == key.data_ptr()
    assert m32.data_ptr() == mask.data_ptr()
    assert parents32.tolist() == [0]
