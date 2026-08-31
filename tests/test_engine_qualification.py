from __future__ import annotations

import pytest

from pra_hf.engine_qualification import (
    E0_REQUIRED_INVARIANTS,
    E1_REQUIRED_INVARIANTS,
    E2_REQUIRED_INVARIANTS,
    FrozenSelection,
    QualificationManifest,
    assert_selector_frozen,
    qualification_gaps,
)
from pra_hf.product_matrix import ProductMatrixRow


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
