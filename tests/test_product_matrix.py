from __future__ import annotations

import json

import pytest

from experiments.paper4_5_runtime.build_product_matrix_v2 import (
    _mlx_m4_cross_model_rows,
    _mlx_m4_pressure_rows,
    _mlx_consumer_scaling_rows,
    _vllm_cuda_concurrency_rows,
)
from pra_hf.product_matrix import ProductMatrix, ProductMatrixRow, optional_number


def _row(**overrides):
    values = {
        "row_id": "qwen-mlx-balanced-qasper",
        "model_family": "qwen",
        "model_id": "Qwen/Qwen3-0.6B",
        "model_revision": "revision",
        "model_size": 600_000_000,
        "model_variant": "instruct",
        "engine": "mlx",
        "engine_version": "0.32",
        "hardware": "Apple M5 16GB",
        "profile": "BALANCED",
        "profile_status": "MEASURED",
        "workload": "matched_e0_e2",
        "dataset": "qasper",
        "quality_metric": "f1",
        "quality_score": 0.8,
        "quality_reference": 0.8,
        "quality_delta": 0.0,
        "evidence_tier": "HELD_OUT",
        "evidence_provenance": "results/qasper.json",
        "experiment_status": "MEASURED",
    }
    values.update(overrides)
    return ProductMatrixRow(**values)


def test_product_matrix_round_trip_uses_null_for_unknown_metrics(tmp_path) -> None:
    matrix = ProductMatrix("2026-08-v1", (_row(ttft_ms=None),))
    path = tmp_path / "matrix.json"

    matrix.write(path)
    restored = ProductMatrix.read(path)

    assert restored == matrix
    payload = json.loads(path.read_text())["rows"][0]
    assert payload["ttft_ms"] is None
    assert payload["metric_statuses"]["ttft_ms"] == "NOT_MEASURED"
    assert payload["metric_provenance"]["quality_score"] == "results/qasper.json"


def test_product_matrix_rejects_unknown_fields_and_duplicate_ids() -> None:
    payload = _row().to_dict()
    payload["invented_metric"] = 1
    with pytest.raises(ValueError, match="invented_metric"):
        ProductMatrixRow.from_dict(payload)
    with pytest.raises(ValueError, match="unique"):
        ProductMatrix("duplicate", (_row(), _row()))


def test_optional_number_does_not_turn_unknown_markers_into_zero() -> None:
    assert optional_number("NOT_MEASURED") is None
    assert optional_number(None) is None
    assert optional_number("12.5") == 12.5


def test_quality_adjusted_throughput_and_cost_are_derived_from_measured_inputs() -> None:
    row = _row(
        task_success=0.75,
        requests_per_second=4.0,
        accelerator_cost_per_hour=2.0,
        hourly_cost_source="internal hardware accounting",
    )

    assert row.successful_requests_per_second == 3.0
    assert row.successful_tasks_per_accelerator_hour == 10_800.0
    assert row.cost_per_successful_task == pytest.approx(2.0 / 10_800.0)


def test_throughput_without_quality_is_rejected() -> None:
    with pytest.raises(ValueError, match="Throughput rows"):
        _row(quality_score=None, task_success=None, requests_per_second=2.0)


def test_engine_evidence_rows_preserve_candidate_boundaries() -> None:
    mlx_rows = _mlx_m4_cross_model_rows()
    pressure_rows = _mlx_m4_pressure_rows()
    cuda_rows = _vllm_cuda_concurrency_rows()

    assert len(mlx_rows) == 18
    assert len(pressure_rows) == 12
    assert len(cuda_rows) == 12
    compact = next(row for row in mlx_rows if row.model_variant == "native_int8_resident")
    assert compact.representation == "E2_COLD_INT8_RESIDENT"
    assert compact.cold_bytes is not None
    assert compact.hot_bytes is None
    assert all(row.profile_status == "RESEARCH_ONLY" for row in pressure_rows)
    assert all(row.integration_level == "E2" for row in pressure_rows)
    assert all(row.integration_level == "E1" for row in cuda_rows)
    assert all(row.profile_status == "RESEARCH_ONLY" for row in cuda_rows)


def test_mlx_consumer_scaling_keeps_balanced_all_layer_and_sparse_pending() -> None:
    rows = _mlx_consumer_scaling_rows()

    assert rows
    balanced_native = [
        row for row in rows if row.model_variant == "E2_CONCAT_WARM"
    ]
    reduced = [
        row for row in rows if row.representation == "E2_SEGMENTED_CANDIDATE"
    ]
    assert balanced_native
    assert all(row.profile == "BALANCED" for row in balanced_native)
    assert all(row.profile_status == "MEASURED" for row in balanced_native)
    assert all(len(row.consumer_layers) > 0 for row in balanced_native)
    assert reduced
    assert all(row.profile_status == "CALIBRATION_PENDING" for row in reduced)
    assert all(row.profile == "REDUCED_CANDIDATE" for row in reduced)
