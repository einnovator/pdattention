import pytest

from pra_hf.channel_geometry import (
    headroom_decomposition,
    jaccard,
    new_address_tokens,
    oracle_channel,
    precision_recall,
    reciprocal_rank_fusion,
    select_observable_channel,
    useful_address,
)


def test_precision_recall_and_overlap_use_unique_chunk_identities():
    precision, recall = precision_recall(["a", "a", "b"], ["b", "c"])
    assert precision == 0.5
    assert recall == 0.5
    assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)


def test_oracle_channel_is_deterministic_on_ties():
    channel, recall = oracle_channel(
        {"gist": 0.5, "exact": 0.8, "bm25": 0.8, "approx": 0.2,
         "hybrid": 0.4, "iterative_hybrid": 0.3}
    )
    assert (channel, recall) == ("exact", 0.8)


def test_new_address_requires_first_hop_and_later_gold_overlap():
    assert new_address_tokens(
        ["where", "alice"], ["alice", "points", "cobalt"], ["cobalt", "payload"]
    ) == {"cobalt"}


def test_selector_rejects_gold_geometry_and_uses_observable_signals():
    with pytest.raises(ValueError, match="Gold-derived"):
        select_observable_channel({"evidence_regions": 2})
    assert select_observable_channel(
        {"query_rare_fraction": 0.3, "exact_top_score": 0.9}
    ) == "exact"
    assert select_observable_channel({"new_address_observed": True}) == "iterative_hybrid"


def test_rank_fusion_does_not_mix_incomparable_raw_scores():
    scores = reciprocal_rank_fusion(
        {"semantic": {"a": 1, "b": 2}, "exact": {"b": 1, "a": 2}}
    )
    assert scores["a"] == scores["b"]


def test_headroom_and_useful_address_have_distinct_semantics():
    assert headroom_decomposition(0.8, 0.6, 0.5) == pytest.approx((0.2, 0.1))
    assert useful_address(exposed=True, gold_linked=True, successor_rank=3, rank_limit=4)
    assert not useful_address(
        exposed=True, gold_linked=False, successor_rank=1, rank_limit=4
    )
