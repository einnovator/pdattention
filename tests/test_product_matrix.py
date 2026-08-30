from __future__ import annotations

import json

import pytest

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
    assert json.loads(path.read_text())["rows"][0]["ttft_ms"] is None


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
