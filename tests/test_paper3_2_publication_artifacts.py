from __future__ import annotations

import gzip
import json

import pytest

from experiments.paper3_2_rag.build_publication_artifacts import (
    _adapter_delta_histogram,
    _native_record_scale_aggregate_summary,
    _transport_systems_plot,
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


def test_transport_systems_plot_writes_both_formats(tmp_path) -> None:
    conditions = []
    for condition, regime, ttft, total, visible, native in [
        ("PRA_SELECTED_CONTEXT_NO_ADAPTOR", "WARM", 1000, 1100, 2100, 0),
        ("PRA_NATIVE_MEMORY_NO_ADAPTOR", "WARM", 80, 200, 80, 2020),
        ("PRA_SELECTED_CONTEXT_NO_ADAPTOR", "PREFIX_WARM", 70, 160, 2100, 0),
    ]:
        conditions.append(
            {
                "condition": condition,
                "regime": regime,
                "selector_profile": "pra_strong_reranker",
                "ttft_ms_mean": ttft,
                "total_latency_ms_mean": total,
                "visible_prompt_tokens_mean": visible,
                "selected_native_kv_tokens_mean": native,
            }
        )

    output = tmp_path / "transport"
    _transport_systems_plot({"conditions": conditions}, output)

    assert output.with_suffix(".pdf").is_file()
    assert output.with_suffix(".png").is_file()


def test_adapter_delta_histogram_uses_paired_examples(tmp_path) -> None:
    rows = []
    for example_id, baseline, repaired in [("a", 0.1, 0.2), ("b", 0.3, 0.2)]:
        rows.extend(
            [
                {
                    "seed": 11,
                    "example_id": example_id,
                    "condition": "C_INDEPENDENT_PRA",
                    "token_f1": baseline,
                },
                {
                    "seed": 11,
                    "example_id": example_id,
                    "condition": "R_TRAINED_RESIDUAL",
                    "token_f1": repaired,
                },
            ]
        )
    source = tmp_path / "rows.jsonl.gz"
    with gzip.open(source, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")

    output = tmp_path / "adapter"
    _adapter_delta_histogram(source, output)

    assert output.with_suffix(".pdf").is_file()
    assert output.with_suffix(".png").is_file()
