from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from experiments.paper3_2_rag.aggregate_prerope_causal import aggregate


def _write_run(root: Path, seed: int, a_f1: float, b_f1: float) -> Path:
    run = root / f"seed{seed}"
    run.mkdir()
    manifest = {
        "dataset": "multihoprag",
        "model": "model",
        "model_revision": "revision",
        "reranker": "reranker",
        "reranker_revision": "reranker-revision",
        "token_budget": 2048,
        "max_resources": 4,
        "seed": seed,
        "summary": {
            "b_minus_c": {
                "max_layer_key_rmse": 1e-5 * seed,
                "max_layer_value_rmse": 2e-5 * seed,
            }
        },
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    common = {
        "example_id": f"example-{seed}",
        "exact_match": 0.0,
        "official_multihop_rag_score": 0.0,
        "cross_document_attention_edges_allowed": 0,
        "request_rope_transform_ms": 0.0,
        "first_step_js_divergence": None,
    }
    rows = [
        {
            **common,
            "condition": "A_FULL_CAUSAL_RAG",
            "token_f1": a_f1,
            "gold_answer_mean_nll": 2.0,
            "prediction": "a",
            "first_step_logits_sha256": "a",
        },
        {
            **common,
            "condition": "B_NO_CROSS_DOC_RAG",
            "token_f1": b_f1,
            "gold_answer_mean_nll": 2.2,
            "prediction": "same",
            "first_step_logits_sha256": "b",
            "first_step_js_divergence": 0.1,
        },
        {
            **common,
            "condition": "C_PRA_PRE_ROPE_EXACT_PACKED_OFFSETS",
            "token_f1": b_f1,
            "gold_answer_mean_nll": 2.2,
            "prediction": "same",
            "first_step_logits_sha256": "b",
            "first_step_js_divergence": 0.001,
            "request_rope_transform_ms": 1.5,
        },
    ]
    with gzip.open(run / "condition_results.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    with gzip.open(run / "bc_layer_diagnostics.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "layers": [
                        {
                            "layer": 0,
                            "key_rmse": 1e-5 * seed,
                            "key_max_abs_delta": 2e-5 * seed,
                            "value_rmse": 2e-5 * seed,
                            "value_max_abs_delta": 3e-5 * seed,
                        }
                    ]
                }
            )
            + "\n"
        )
    return run / "manifest.json"


def test_prerope_aggregate_keeps_seed_as_replication_unit(tmp_path: Path) -> None:
    manifests = (
        _write_run(tmp_path, 11, 0.3, 0.2),
        _write_run(tmp_path, 23, 0.4, 0.2),
    )
    result = aggregate(manifests)
    assert result["replication_unit"] == "seed_cohort"
    assert result["a_minus_b"]["seed_mean_token_f1_delta"] == pytest.approx(0.15)
    assert result["a_minus_b"]["exact_two_sided_sign_flip_p_token_f1"] == 0.5
    assert result["b_minus_c"]["output_match_rate"] == 1.0
    assert result["b_minus_c"]["first_step_logit_hash_match_rate"] == 1.0
    assert result["b_minus_c"]["mean_first_step_js_divergence"] == 0.001
    assert result["b_minus_c"]["seed_mean_token_f1_delta"] == 0.0
    assert result["b_minus_c"]["layerwise"][0]["pairs"] == 2
    assert result["b_minus_c"]["layerwise"][0]["max_key_rmse"] == pytest.approx(
        23e-5
    )


def test_prerope_aggregate_rejects_mixed_protocols(tmp_path: Path) -> None:
    first = _write_run(tmp_path, 11, 0.3, 0.2)
    second = _write_run(tmp_path, 23, 0.4, 0.2)
    value = json.loads(second.read_text(encoding="utf-8"))
    value["model_revision"] = "changed"
    second.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen protocol"):
        aggregate((first, second))
