from __future__ import annotations

import gzip
import json

from experiments.paper3_2_rag.aggregate_crossdoc_scale import _write, aggregate


def test_scale_aggregate_keeps_model_and_precision_identity(tmp_path) -> None:
    run = tmp_path / "model"
    run.mkdir()
    manifest = {
        "model": "org/model-4bit",
        "model_revision": "a" * 40,
        "precision": {"precision_mode": "INT4"},
        "seed": 11,
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    row = {
        "condition": "D_GIST_SA_APPEND",
        "token_f1": 0.5,
        "official_multihop_rag_score": 0.4,
        "gold_answer_mean_nll": 2.0,
        "first_step_js_divergence": 0.01,
        "request_composition_ms": 1.0,
        "request_local_native_tokens": 4,
        "request_composition_flops_estimate": 1024,
    }
    with gzip.open(run / "condition_results.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    result = aggregate((run / "manifest.json",))
    assert result["conditions"][0]["model"] == "org/model-4bit"
    assert result["conditions"][0]["precision_mode"] == "INT4"
    assert result["conditions"][0]["request_composition_flops_estimate"] == 1024
    output = tmp_path / "aggregate"
    _write(result, output)
    table = (output / "generated_scale_table.tex").read_text(encoding="utf-8")
    assert "model & D gist append" in table
