from __future__ import annotations

import pytest
import torch

from pra_torch.materialization import (
    PROMPT_HISTORY_DOMAIN,
    LogicalDomainBounds,
    LogicalInterval,
    MissingLogicalKVError,
    allocate_interval_budget,
    domain_bounds_from_shards,
    evidence_centered_interval,
    gather_logical_kv,
    logical_domain,
    shards_from_chunks,
    union_intervals,
)
from pra_torch.attention import PRAttention
from pra_torch.config import PRAConfig
from pra_torch.memory import (
    ChunkRoutingGist,
    LayerKV,
    LayerReferenceMemory,
    PRACacheEntry,
    PRASimpleMemoryCache,
    ReferenceChunkMemory,
    SelectedChunk,
)


def _chunk(uri: str, start: int, end: int, *, chunk_id: str | None = None):
    positions = torch.arange(start, end, dtype=torch.float32)
    key = positions.view(1, 1, -1, 1).expand(1, 2, -1, 3).clone()
    value = (positions + 100).view(1, 1, -1, 1).expand(1, 2, -1, 3).clone()
    return ReferenceChunkMemory(
        chunk_id=chunk_id or f"{uri}:{start}:{end}",
        source_uri=uri,
        token_start=start,
        token_end=end,
        logical_start=start,
        logical_end=end,
        token_kv=LayerKV(k=key, v=value, position_ids=torch.arange(start, end)),
        routing_gist=ChunkRoutingGist(k=torch.zeros(6)),
        metadata={"encoding_block_tokens": 4},
    )


def _positions(result):
    return result.key[0, 0, :, 0].tolist()


def test_gather_inside_one_and_multiple_storage_boundaries_in_order():
    shards = shards_from_chunks(
        [_chunk("mem://a", 0, 4), _chunk("mem://a", 4, 8), _chunk("mem://a", 8, 12)]
    )
    inside = gather_logical_kv(
        shards, [LogicalInterval("mem://a", 1, 3)], device="cpu"
    )
    assert _positions(inside) == [1, 2]

    one = gather_logical_kv(
        shards, [LogicalInterval("mem://a", 2, 6)], device="cpu"
    )
    assert _positions(one) == [2, 3, 4, 5]
    assert one.stats.cross_shard_interval_count == 1
    assert one.stats.storage_shards_touched == 2

    multiple = gather_logical_kv(
        shards, [LogicalInterval("mem://a", 1, 11)], device="cpu"
    )
    assert _positions(multiple) == list(range(1, 11))
    assert multiple.key.shape == (1, 2, 10, 3)
    assert multiple.value.shape == multiple.key.shape
    assert [fragment.shard_id for fragment in multiple.fragments] == [
        "mem://a:0:4",
        "mem://a:4:8",
        "mem://a:8:12",
    ]


def test_overlap_is_union_deduplicated_and_evidence_density_is_exact():
    shards = shards_from_chunks(
        [
            _chunk("mem://a", 0, 6, chunk_id="left"),
            _chunk("mem://a", 4, 10, chunk_id="right"),
        ]
    )
    intervals = [
        LogicalInterval("mem://a", 1, 7, 2, 4),
        LogicalInterval("mem://a", 3, 9, 7, 8),
    ]
    result = gather_logical_kv(shards, intervals, device="cpu")
    assert _positions(result) == list(range(1, 9))
    assert len(result.logical_positions) == len(set(result.logical_positions))
    assert result.stats.requested_tokens_pre_dedup == 12
    assert result.stats.deduplicated_tokens == 8
    assert result.stats.evidence_tokens == 3
    assert result.stats.non_evidence_tokens == 5
    assert result.stats.evidence_density == pytest.approx(3 / 8)


def test_expansion_clips_at_explicit_resource_boundaries():
    bounds = LogicalDomainBounds("mem://a", 0, 10)
    interval = evidence_centered_interval(
        "mem://a", 1, 3, radius_left=8, radius_right=20, bounds=bounds
    )
    assert (interval.start, interval.end) == (0, 10)
    with pytest.raises(ValueError, match="different materialization domains"):
        evidence_centered_interval(
            "mem://b", 1, 3, radius_left=1, radius_right=1, bounds=bounds
        )


def test_head_and_prompt_tail_form_one_continuous_domain():
    assert logical_domain("#__head") == PROMPT_HISTORY_DOMAIN
    assert logical_domain("#__prompt") == PROMPT_HISTORY_DOMAIN
    shards = shards_from_chunks(
        [_chunk("#__head", 0, 4), _chunk("#__prompt", 4, 8)]
    )
    result = gather_logical_kv(
        shards,
        [LogicalInterval(PROMPT_HISTORY_DOMAIN, 2, 6)],
        device="cpu",
    )
    assert _positions(result) == [2, 3, 4, 5]
    assert result.stats.cross_shard_interval_count == 1


def test_domains_never_fill_each_others_logical_gaps():
    shards = shards_from_chunks(
        [_chunk("mem://a", 0, 4), _chunk("mem://b", 4, 8)]
    )
    with pytest.raises(MissingLogicalKVError):
        gather_logical_kv(
            shards, [LogicalInterval("mem://a", 2, 6)], device="cpu"
        )


def test_union_preserves_domain_order_and_separate_evidence_spans():
    intervals = [
        LogicalInterval("mem://b", 2, 5, 3, 4),
        LogicalInterval("mem://a", 0, 2),
        LogicalInterval("mem://b", 4, 7, 6, 7),
    ]
    merged = union_intervals(intervals)
    assert [(row.domain, row.start, row.end) for row in merged] == [
        ("mem://b", 2, 7),
        ("mem://a", 0, 2),
    ]


@pytest.mark.parametrize(
    "strategy", ["equal", "evidence_length_proportional", "minimum_core_remainder"]
)
def test_fixed_budget_allocates_every_evidence_region(strategy):
    intervals = [
        LogicalInterval("mem://a", 0, 12, 4, 6),
        LogicalInterval("mem://b", 0, 12, 3, 8),
        LogicalInterval("mem://c", 0, 12, 5, 6),
    ]
    allocated = allocate_interval_budget(
        intervals,
        total_budget=9,
        strategy=strategy,
        minimum_per_region=2,
    )
    assert len(allocated) == 3
    assert sum(interval.token_count for interval in allocated) == 9
    assert all(interval.token_count >= 2 for interval in allocated)
    assert [interval.domain for interval in allocated] == [
        "mem://a",
        "mem://b",
        "mem://c",
    ]


def test_fixed_budget_rejects_silent_region_starvation_and_radius_zero_works():
    intervals = [
        LogicalInterval("mem://a", 0, 4, 1, 2),
        LogicalInterval("mem://b", 0, 4, 1, 2),
    ]
    with pytest.raises(ValueError, match="too small"):
        allocate_interval_budget(
            intervals, total_budget=1, strategy="equal", minimum_per_region=1
        )
    radius_zero = evidence_centered_interval(
        "mem://a",
        1,
        2,
        radius_left=0,
        radius_right=0,
        bounds=LogicalDomainBounds("mem://a", 0, 4),
    )
    assert (radius_zero.start, radius_zero.end) == (1, 2)


def test_minimum_core_remainder_preserves_complete_evidence_or_fails():
    intervals = [
        LogicalInterval("mem://a", 0, 10, 2, 6),
        LogicalInterval("mem://b", 0, 10, 3, 6),
    ]
    allocated = allocate_interval_budget(
        intervals,
        total_budget=9,
        strategy="minimum_core_remainder",
    )
    assert sum(interval.token_count for interval in allocated) == 9
    assert [(interval.evidence_start, interval.evidence_end) for interval in allocated] == [
        (2, 6),
        (3, 6),
    ]
    with pytest.raises(ValueError, match="too small"):
        allocate_interval_budget(
            intervals,
            total_budget=6,
            strategy="minimum_core_remainder",
        )


def test_domain_bounds_and_whole_parent_equivalence():
    shards = shards_from_chunks(
        [_chunk("mem://a", 0, 4), _chunk("mem://a", 4, 8)]
    )
    assert domain_bounds_from_shards(shards)["mem://a"] == LogicalDomainBounds(
        "mem://a", 0, 8
    )
    gathered = gather_logical_kv(
        shards, [LogicalInterval("mem://a", 0, 8)], device="cpu"
    )
    expected_key = torch.cat([shard.kv.k for shard in shards], dim=2)
    expected_value = torch.cat([shard.kv.v for shard in shards], dim=2)
    torch.testing.assert_close(gathered.key, expected_key)
    torch.testing.assert_close(gathered.value, expected_value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA transfer accounting")
def test_cpu_resident_gqa_shards_transfer_only_gathered_kv():
    shards = shards_from_chunks([_chunk("mem://a", 0, 8)])
    result = gather_logical_kv(
        shards, [LogicalInterval("mem://a", 2, 5)], device="cuda"
    )
    assert result.key.shape == (1, 2, 3, 3)
    assert result.key.is_cuda and result.value.is_cuda
    expected_bytes = 2 * result.key.numel() * result.key.element_size()
    assert result.stats.transferred_kv_bytes == expected_bytes


def _logical_attention(mode: str, *, max_tokens: int = 8):
    chunks = [_chunk("mem://a", 0, 4), _chunk("mem://a", 4, 8)]
    entry = PRACacheEntry(
        uri="mem://a",
        text="eight tokens",
        layer_memory={0: LayerReferenceMemory(chunks)},
    )
    cache = PRASimpleMemoryCache()
    cache.put(entry)
    config = PRAConfig(
        vocab_size=16,
        d_model=6,
        n_heads=2,
        n_layers=1,
        max_seq_len=16,
        model_max_context_tokens=16,
        max_materialized_memory_tokens=max_tokens,
        trigger_threshold=-1.0,
        detail_materialization=mode,
    )
    attention = PRAttention(6, 2, 16, 0, cache, config=config)
    selected = SelectedChunk(
        entry=entry,
        chunk=chunks[0],
        reference_score=1.0,
        chunk_score=1.0,
        layer_id=0,
        reference_rank=1,
        rank_within_reference=1,
        metadata={
            "materialization_intervals": [
                {
                    "source_uri": "mem://a",
                    "start": 2,
                    "end": 6,
                    "evidence_start": 3,
                    "evidence_end": 5,
                }
            ]
        },
    )
    return attention, selected


def test_core_logical_mode_materializes_cross_chunk_window_and_metrics():
    attention, selected = _logical_attention("logical_intervals")
    key, value, retained, duplicates, _moved, _seconds, stats = attention._materialize(
        [selected], torch.zeros(1, 2, 1, 3), direct_tokens=2
    )
    assert key[0, 0, :, 0].tolist() == [2, 3, 4, 5]
    assert value[0, 0, :, 0].tolist() == [102, 103, 104, 105]
    assert retained == [selected]
    assert duplicates == 0
    assert stats["requested_tokens_pre_dedup"] == 4
    assert stats["materialized_native_kv_tokens"] == 4
    assert stats["evidence_kv_tokens"] == 2
    assert stats["evidence_density"] == pytest.approx(0.5)
    assert stats["cross_shard_interval_count"] == 1


def test_core_logical_mode_enforces_actual_native_kv_budget():
    attention, selected = _logical_attention("logical_intervals", max_tokens=3)
    with pytest.raises(ValueError, match="only 3 are available"):
        attention._materialize(
            [selected], torch.zeros(1, 2, 1, 3), direct_tokens=2
        )


def test_core_gist_plus_local_accounts_for_physical_native_gist():
    attention, selected = _logical_attention("gist_plus_logical_intervals")
    key, _value, _retained, _duplicates, _moved, _seconds, stats = attention._materialize(
        [selected], torch.zeros(1, 2, 1, 3), direct_tokens=2
    )
    assert key.shape == (1, 2, 5, 3)
    assert key[0, 0, 0, 0].item() == pytest.approx(1.5)
    assert key[0, 0, 1:, 0].tolist() == [2, 3, 4, 5]
    assert stats["materialized_gist_kv_tokens"] == 1
    assert stats["materialized_native_kv_tokens"] == 5
    assert stats["evidence_density"] == pytest.approx(2 / 5)


def test_core_native_gist_only_preserves_gqa_layout_without_intervals():
    attention, selected = _logical_attention("native_gist_only")
    selected = SelectedChunk(
        **{
            **selected.__dict__,
            "metadata": {"selection_source": "oracle_parent"},
        }
    )
    key, value, _retained, _duplicates, _moved, _seconds, stats = attention._materialize(
        [selected], torch.zeros(1, 2, 1, 3), direct_tokens=2
    )
    assert key.shape == value.shape == (1, 2, 1, 3)
    assert key[0, 0, 0, 0].item() == pytest.approx(1.5)
    assert stats["materialized_gist_kv_tokens"] == 1
    assert stats["evidence_kv_tokens"] == 0


def test_core_logical_mode_returns_empty_when_selection_is_empty():
    attention, _selected = _logical_attention("logical_intervals")
    key, value, retained, _duplicates, _moved, _seconds, stats = attention._materialize(
        [], torch.zeros(1, 2, 1, 3), direct_tokens=2
    )
    assert key.shape == value.shape == (1, 2, 0, 3)
    assert retained == []
    assert stats["memory_tokens_materialized"] == 0
