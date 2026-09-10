from __future__ import annotations

import pytest

from pra_hf.deployment import HuggingFaceEngineAdapter
from pra_hf.engine_qualification import (
    E0_REQUIRED_INVARIANTS,
    E1_REQUIRED_INVARIANTS,
    E2_REQUIRED_INVARIANTS,
    LivePrefixKVObservation,
    SelectedKVRange,
    SelectedKVSource,
    STATEFUL_AGENT_REQUIRED_INVARIANTS,
    FrozenSelection,
    QualificationManifest,
    assert_selector_frozen,
    qualification_gaps,
    stateful_agent_qualification_gaps,
)
from pra_hf.product_matrix import ProductMatrixRow
from pra_mlx import MLXEngineAdapter
from pra_sglang import SGLangEngineAdapter
from pra_vllm import VLLMEngineAdapter


def _row(**overrides) -> ProductMatrixRow:
    values = {
        "row_id": "engine-condition",
        "model_family": "qwen",
        "model_id": "Qwen/Qwen3-0.6B",
        "model_revision": "revision",
        "model_size": 600_000_000,
        "model_variant": "instruct",
        "engine": "test",
        "engine_version": "1",
        "hardware": "test accelerator",
        "profile": "BALANCED",
        "profile_status": "MEASURED",
        "workload": "matched_e0_e2",
        "dataset": "qasper",
        "quality_metric": "task_success",
        "quality_score": 1.0,
        "task_success": 1.0,
        "visible_tokens": 32.0,
        "ttft_p50_ms": 10.0,
        "requests_per_second": 2.0,
        "evidence_tier": "HELD_OUT",
        "evidence_provenance": "results.json",
        "experiment_status": "NATURAL_WORKLOAD",
        "verified_invariants": tuple(E0_REQUIRED_INVARIANTS),
    }
    values.update(overrides)
    return ProductMatrixRow(**values)


def test_frozen_selection_digest_is_stable_and_manifest_serializes() -> None:
    selection = FrozenSelection.create(
        example_id="a",
        query="question",
        candidate_ids=("a", "b"),
        selected_ids=("b",),
        selected_intervals=(("b", 4, 8),),
    )
    manifest = QualificationManifest("manifest", selections=(selection,))

    assert len(selection.digest) == 64
    assert manifest.to_dict()["selections"][0]["digest"] == selection.digest


def test_e2_gate_requires_native_metrics_invariants_and_frozen_selector() -> None:
    e0 = _row()
    assert qualification_gaps(e0, "E0") == ()
    assert "active_kv_tokens" in qualification_gaps(e0, "E2")

    complete = _row(
        integration_level="E2",
        representation="E2_HOT",
        selector_digest="digest",
        active_kv_tokens=16.0,
        active_kv_bytes=4096.0,
        consumer_layers=(20, 21),
        exact_pair_parity=1.0,
        verified_invariants=tuple(
            E0_REQUIRED_INVARIANTS | E1_REQUIRED_INVARIANTS | E2_REQUIRED_INVARIANTS
        ),
    )
    assert qualification_gaps(complete, "E2") == ()


def test_matched_representations_reject_independent_selector_outputs() -> None:
    left = _row(selector_digest="selection-a")
    right = _row(row_id="right", selector_digest="selection-b")

    with pytest.raises(ValueError, match="selector digest"):
        assert_selector_frozen((left, right))


def test_stateful_agent_gate_is_stricter_than_static_e2_parity() -> None:
    incomplete = stateful_agent_qualification_gaps(
        exact_pairs=4,
        total_pairs=5,
        verified_invariants=("generated_history_replay_exact",),
    )
    assert "sequential_state_exact:4/5" in incomplete
    assert "invariant:boundary_token_preserved" in incomplete

    assert stateful_agent_qualification_gaps(
        exact_pairs=5,
        total_pairs=5,
        verified_invariants=tuple(STATEFUL_AGENT_REQUIRED_INVARIANTS),
    ) == ()


def test_live_prefix_kv_gate_rejects_selected_text_rematerialization() -> None:
    observation = LivePrefixKVObservation(
        source=SelectedKVSource.TEXT_REMATERIALIZED,
        source_prefix_id="session-a:turn-4",
        source_token_count=100,
        ranges=(SelectedKVRange("r1", "r1", "turn-1", 0, 100),),
        selected_kv_tokens=100,
        reused_kv_tokens=40,
        selected_text_reencoded_tokens=60,
        source_positions_preserved=True,
        single_attention_normalization=True,
        exact_live_prefix_continuation=False,
    )

    gaps = observation.qualification_gaps(require_full_retention=True)
    assert "source:text_rematerialized" in gaps
    assert "reused_token_count" in gaps
    assert "selected_text_reencoded:60" in gaps
    assert "full_retention_prefix_equivalence" in gaps


def test_live_prefix_kv_gate_accepts_exact_noncontiguous_subset_and_full_prefix() -> None:
    subset = LivePrefixKVObservation(
        source="live_prefix_capture",
        source_prefix_id="session-a:turn-4",
        source_token_count=100,
        ranges=(
            SelectedKVRange("r1", "r1", "turn-1", 0, 20),
            SelectedKVRange("r3", "r3", "turn-3", 70, 100),
        ),
        selected_kv_tokens=50,
        reused_kv_tokens=50,
        selected_text_reencoded_tokens=0,
        source_positions_preserved=True,
        single_attention_normalization=True,
    )
    assert subset.qualification_gaps() == ()
    assert "full_retention_coverage" in subset.qualification_gaps(
        require_full_retention=True
    )

    full = LivePrefixKVObservation(
        source="live_prefix_capture",
        source_prefix_id="session-a:turn-4",
        source_token_count=100,
        ranges=(
            SelectedKVRange("r1", "r1", "turn-1", 0, 40),
            SelectedKVRange("r2", "r2", "turn-2", 40, 100),
        ),
        selected_kv_tokens=100,
        reused_kv_tokens=100,
        selected_text_reencoded_tokens=0,
        source_positions_preserved=True,
        single_attention_normalization=True,
        exact_live_prefix_continuation=True,
    )
    assert full.qualification_gaps(require_full_retention=True) == ()


def test_detached_native_backends_do_not_claim_live_agent_prefix_subsets() -> None:
    detached = object()
    capabilities = (
        HuggingFaceEngineAdapter(detached).capabilities(),
        MLXEngineAdapter("http://unused", native_executor=detached).capabilities(),
        VLLMEngineAdapter("http://unused", native_executor=detached).capabilities(),
        SGLangEngineAdapter("http://unused", native_executor=detached).capabilities(),
    )

    assert all(row.native_kv for row in capabilities)
    assert all(not row.live_prefix_kv_subset for row in capabilities)
    assert all(not row.zero_selected_text_reencoding for row in capabilities)
