from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pra_hf import (
    FrozenNativeAnchor,
    FrozenNativeSelection,
    PRAConfig,
    NativeMaterializationMode,
    build_native_materialization_plan,
    evidence_token_intervals,
    expand_frozen_intervals,
    intervals_cover,
    materialization_profile,
)
from pra_torch.memory import (
    ChunkRoutingGist,
    LayerKV,
    LayerReferenceMemory,
    PRACacheEntry,
    ReferenceChunkMemory,
)


class _Encoding:
    def __init__(self, values):
        self.input_ids = values


class _WhitespaceTokenizer:
    def __call__(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return _Encoding(text.split())


def _chunk(uri: str, start: int, end: int, layer: int) -> ReferenceChunkMemory:
    count = end - start
    values = torch.arange(start, end, dtype=torch.float32).view(1, 1, count, 1)
    return ReferenceChunkMemory(
        f"{uri}#chunk={start}:{end}",
        uri,
        start,
        end,
        LayerKV(values, values + layer, position_ids=torch.arange(start, end)),
        ChunkRoutingGist(torch.ones(1, 1)),
        logical_start=start,
        logical_end=end,
    )


def _entry(uri: str = "record://a", total: int = 80) -> PRACacheEntry:
    layers = {}
    for layer in (3, 7):
        layers[layer] = LayerReferenceMemory([
            _chunk(uri, 0, 32, layer),
            _chunk(uri, 24, 56, layer),
            _chunk(uri, 48, total, layer),
        ])
    return PRACacheEntry(uri, "", layer_memory=layers)


def _frozen(uri: str = "record://a") -> FrozenNativeSelection:
    return FrozenNativeSelection((
        FrozenNativeAnchor(uri, f"{uri}#chunk=24:56", 24, 56, 0.8, 0.7),
    ))


def test_symmetric_expansion_respects_record_boundary() -> None:
    intervals = expand_frozen_intervals(
        _frozen(), {"record://a": 80}, target_span_tokens=64
    )
    assert intervals[0].start == 8
    assert intervals[0].end == 72
    edge = FrozenNativeSelection((
        FrozenNativeAnchor("record://a", "edge", 0, 32),
    ))
    assert expand_frozen_intervals(edge, {"record://a": 80}, target_span_tokens=64)[0].end == 64


def test_full_record_and_overlapping_anchors_merge_without_crossing_records() -> None:
    frozen = FrozenNativeSelection((
        FrozenNativeAnchor("record://a", "a1", 0, 32),
        FrozenNativeAnchor("record://a", "a2", 24, 56),
        FrozenNativeAnchor("record://b", "b1", 0, 20),
    ))
    intervals = expand_frozen_intervals(
        frozen, {"record://a": 80, "record://b": 20}, full_selected_record=True
    )
    assert [(row.reference_uri, row.start, row.end) for row in intervals] == [
        ("record://a", 0, 80),
        ("record://b", 0, 20),
    ]


def test_plan_slices_unique_native_tokens_at_every_consumption_layer() -> None:
    plan = build_native_materialization_plan(
        [_entry()], _frozen(), consumption_layers=(3, 7), target_span_tokens=64
    )
    assert plan.unique_native_tokens == 64
    assert plan.query_position_offset == 80
    assert plan.frozen.source_identity == _frozen().source_identity
    for layer, rows in plan.selections_by_layer.items():
        assert sum(row.selected_token_count for row in rows) == 64
        assert all(row.layer_id == layer for row in rows)
        assert all(row.reference_uri == "record://a" for row in rows)
        spans = [(row.logical_start, row.logical_end) for row in rows]
        assert spans[0][0] == 8
        assert spans[-1][1] == 72
        assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))


def test_interval_normalizer_reports_raw_overlap_without_crossing_records() -> None:
    frozen = FrozenNativeSelection((
        FrozenNativeAnchor("record://a", "record://a#chunk=0:32", 0, 32),
        FrozenNativeAnchor("record://a", "record://a#chunk=24:56", 24, 56),
    ))
    plan = build_native_materialization_plan(
        [_entry()], frozen, consumption_layers=(3,)
    )

    assert plan.raw_interval_count == 2
    assert plan.raw_native_tokens == 64
    assert plan.unique_native_tokens == 56
    assert plan.overlap_removed_tokens == 8
    assert plan.duplication_ratio == pytest.approx(64 / 56)


def test_query_offset_uses_longest_record_without_merging_reference_positions() -> None:
    second = _entry("record://b", total=64)
    frozen = FrozenNativeSelection((
        FrozenNativeAnchor("record://a", "record://a#chunk=24:56", 24, 56),
        FrozenNativeAnchor("record://b", "record://b#chunk=0:32", 0, 32),
    ))
    plan = build_native_materialization_plan(
        [_entry(), second], frozen, consumption_layers=(3,), full_selected_record=True
    )

    assert plan.query_position_offset == 80
    assert [(row.reference_uri, row.start, row.end) for row in plan.intervals] == [
        ("record://a", 0, 80),
        ("record://b", 0, 64),
    ]


def test_asymmetric_window_and_named_profiles_resolve_to_normal_config() -> None:
    interval = expand_frozen_intervals(
        _frozen(),
        {"record://a": 80},
        left_context_tokens=4,
        right_context_tokens=12,
    )[0]
    assert (interval.start, interval.end) == (20, 68)
    assert materialization_profile("paper3_default").routing_chunk_tokens == 32
    assert materialization_profile("paper3_default").routing_chunk_overlap_tokens == 0
    config = PRAConfig(materialization_profile="paper8_full_record_diagnostic")
    assert config.materialization_mode == NativeMaterializationMode.FULL_SELECTED_RECORD.value
    assert config.chunk_tokens == 32


def test_paper3_2_import_keeps_changed_record_modes_out_of_qualified_set() -> None:
    policy_path = (
        Path(__file__).parents[1]
        / "docs/papers/shared/results/paper4_5_runtime_productization/rag"
        / "paper3_2_runtime_policy.json"
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    assert [row["mode"] for row in policy["qualified"]] == [
        "contiguous_immutable_native_block"
    ]
    pending = {row["mode"] for row in policy["calibration_pending"]}
    assert {
        "independent_record_composition",
        "partial_materialization",
        "query_conditioned_composition_repair",
    } <= pending


def test_evidence_interval_coverage_distinguishes_answer_from_semantic_context() -> None:
    tokenizer = _WhitespaceTokenizer()
    text = "Task t7 Acme Atlas status verified authoritative verification code is saffron"
    annotated = evidence_token_intervals(
        tokenizer,
        text,
        answer="saffron",
        semantic_anchors=("Acme Atlas", "verification"),
    )
    answer_only = expand_frozen_intervals(
        FrozenNativeSelection((FrozenNativeAnchor("record://a", "answer", *annotated.answer),)),
        {"record://a": annotated.full_record[1]},
    )
    assert intervals_cover(answer_only, "record://a", annotated.answer)
    assert not intervals_cover(answer_only, "record://a", annotated.semantic)
    assert intervals_cover(
        expand_frozen_intervals(
            FrozenNativeSelection((FrozenNativeAnchor("record://a", "full", 0, annotated.full_record[1]),)),
            {"record://a": annotated.full_record[1]},
        ),
        "record://a",
        annotated.semantic,
    )


def test_materialization_configuration_is_explicit_and_mutually_exclusive() -> None:
    config = PRAConfig(materialization_target_tokens=128)
    assert config.materialization_target_tokens == 128
    try:
        PRAConfig(materialization_target_tokens=128, materialization_full_selected_record=True)
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("Conflicting materialization geometry must be rejected.")
