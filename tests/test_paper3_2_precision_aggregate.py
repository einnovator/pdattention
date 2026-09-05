from __future__ import annotations

import gzip
import json
from pathlib import Path

from experiments.paper3_2_rag.aggregate_precision_sweep import aggregate


def _run(root: Path, mode: str, delta: float) -> Path:
    run = root / mode
    run.mkdir()
    manifest = {
        "seed": 11,
        "precision": {"precision_mode": mode, "kv_dtype": "float16"},
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    common = {
        "example_id": "q1", "token_f1": 0.5, "exact_match": 0.0,
        "official_multihop_rag_score": 0.5, "gold_answer_mean_nll": 2.0,
        "first_step_logits_sha256": "same", "first_step_js_divergence": 0.0,
        "first_step_logit_max_abs_delta": 0.0,
    }
    rows = [
        {**common, "condition": "B_NO_CROSS_DOC_RAG", "prediction": "answer"},
        {
            **common, "condition": "C_PRA_PRE_ROPE_EXACT_PACKED_OFFSETS",
            "prediction": "answer", "gold_answer_mean_nll": 2.0 + delta,
            "first_step_js_divergence": delta,
            "first_step_logit_max_abs_delta": delta * 2,
        },
        {
            **common, "condition": "D_GIST_SA_APPEND", "prediction": "answer",
            "request_composition_ms": 1.0, "request_local_native_tokens": 2,
            "cross_document_interaction_edges": 4,
        },
    ]
    with gzip.open(run / "condition_results.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    with gzip.open(run / "bc_layer_diagnostics.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write(json.dumps({"layers": [{
            "layer": 0, "key_rmse": delta, "value_rmse": delta * 2,
            "key_max_abs_delta": delta * 3, "value_max_abs_delta": delta * 4,
        }]}) + "\n")
    return run / "manifest.json"


def test_precision_aggregate_keeps_modes_and_layerwise_errors_separate(tmp_path: Path) -> None:
    result = aggregate((_run(tmp_path, "FP16", 0.001), _run(tmp_path, "INT4", 0.1)))
    assert [row["precision_mode"] for row in result["precision_conditions"]] == [
        "FP16", "INT4"
    ]
    assert result["precision_conditions"][0]["output_match_rate"] == 1.0
    assert result["precision_conditions"][0]["max_layer_key_rmse"] == 0.001
    assert result["precision_conditions"][1]["mean_gold_nll_abs_delta"] > 0.09
    assert len(result["composition_conditions"]) == 2
