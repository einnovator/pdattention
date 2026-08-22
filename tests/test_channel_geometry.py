import pytest

from pra_hf.channel_geometry import (
    jaccard,
    new_address_tokens,
    oracle_channel,
    precision_recall,
    select_observable_channel,
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
