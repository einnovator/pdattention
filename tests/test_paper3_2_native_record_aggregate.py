from __future__ import annotations

import json
import subprocess
import sys

import pytest


def _manifest(seed: int, packed_f1: float, record_f1: float) -> dict:
    common = {
        "examples": 2,
        "exact_match": 0.0,
        "official_score": 0.0,
        "supporting_document_coverage": 1.0,
        "gold_chunk_recall": 1.0,
        "false_selected_document_fraction": 0.0,
        "answer_string_availability": 1.0,
        "selected_source_tokens": 100.0,
        "visible_prompt_tokens": 10.0,
        "newly_encoded_native_tokens": 100.0,
        "reused_native_tokens": 0.0,
        "ttft_ms": 5.0,
        "total_latency_ms": 10.0,
        "reranker_latency_ms": 2.0,
        "first_step_logit_agreement_with_packed": None,
    }
    return {
        "seed": seed,
        "model": {"id": "model", "revision": "sha"},
        "condition_summary": [
            {
                **common,
                "selector": "minilm",
                "representation": "PACKED_RAG_TEXT",
                "order_name": "canonical",
                "token_f1": packed_f1,
                "gold_answer_mean_nll": 2.0,
                "exact_output_agreement_with_packed": None,
                "first_step_js_vs_packed": None,
            },
            {
                **common,
                "selector": "minilm",
                "representation": "PRA_EXPLICIT_RECORDS",
                "order_name": "canonical",
                "token_f1": record_f1,
                "gold_answer_mean_nll": 1.5,
                "exact_output_agreement_with_packed": 0.25,
                "first_step_logit_agreement_with_packed": 0.0,
                "first_step_js_vs_packed": 0.1,
            },
        ],
        "order_sensitivity": [
            {
                "packed_mean_pairwise_js": 0.2,
                "record_mean_pairwise_js": 0.1,
                "packed_unique_outputs": 3,
                "record_unique_outputs": 2,
                "packed_token_f1_variance": 0.02,
                "record_token_f1_variance": 0.01,
            }
        ],
        "reuse_summary": {
            "mean_overlap_fraction": 0.5,
            "mean_exact_prefix_reusable_tokens": 0.0,
            "mean_newly_encoded_native_tokens": 50.0,
            "mean_reused_native_tokens": 50.0,
            "mean_token_f1": record_f1,
            "mean_packed_token_f1": packed_f1,
            "mean_token_f1_delta_vs_packed": record_f1 - packed_f1,
        },
    }


def test_native_record_aggregate_preserves_seed_level_deltas(tmp_path) -> None:
    manifests = []
    for seed, packed, record in ((11, 0.2, 0.3), (23, 0.4, 0.35)):
        path = tmp_path / f"seed{seed}.json"
        path.write_text(json.dumps(_manifest(seed, packed, record)), encoding="utf-8")
        manifests.append(path)
    output = tmp_path / "aggregate.json"
    command = [
        sys.executable,
        "experiments/paper3_2_rag/aggregate_native_records.py",
    ]
    for path in manifests:
        command.extend(("--manifest", str(path)))
    command.extend(("--output", str(output)))
    subprocess.run(command, check=True)

    aggregate = json.loads(output.read_text(encoding="utf-8"))
    assert aggregate["seeds"] == [11, 23]
    delta = aggregate["representation_deltas"]["minilm|PRA_EXPLICIT_RECORDS"]
    assert delta["token_f1_delta"]["seed_values"] == pytest.approx([0.1, -0.05])
    assert delta["gold_nll_delta"]["mean"] == -0.5
    assert aggregate["order_sensitivity"]["record_mean_pairwise_js"]["mean"] == 0.1
    assert aggregate["reuse"]["mean_reused_native_tokens"]["mean"] == 50.0
