from __future__ import annotations

import json

import pytest

from experiments.paper3_2_rag.build_publication_artifacts import (
    _native_record_scale_aggregate_summary,
)


def test_native_record_scale_aggregate_keeps_replication_metadata(tmp_path) -> None:
    packed = {
        "selector": "minilm",
        "representation": "PACKED_RAG_TEXT",
        "order_name": "canonical",
        "examples": 50,
        "token_f1": 0.2,
        "gold_answer_mean_nll": 1.5,
    }
    records = {
        **packed,
        "representation": "PRA_EXPLICIT_RECORDS",
        "token_f1": 0.21,
        "gold_answer_mean_nll": 1.6,
        "exact_output_agreement_with_packed": 0.1,
        "first_step_js_vs_packed": 0.05,
    }
    manifest = {
        "model": {"id": "model", "revision": "sha"},
        "seeds": [11, 23, 37, 71, 101],
        "conditions": [packed, records],
        "representation_deltas": {
            "minilm|PRA_EXPLICIT_RECORDS": {
                "token_f1_delta": {
                    "mean": 0.01,
                    "bootstrap_95_ci": [-0.02, 0.04],
                },
                "gold_nll_delta": {
                    "mean": 0.1,
                    "bootstrap_95_ci": [0.05, 0.15],
                },
            }
        },
        "reuse": {"mean_reused_native_tokens": {"mean": 100.0}},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    row = _native_record_scale_aggregate_summary([path])[0]

    assert row["seed_count"] == 5
    assert row["examples"] == 50
    assert row["token_f1_delta"] == pytest.approx(0.01)
    assert row["token_f1_delta_95_ci"] == pytest.approx([-0.02, 0.04])
