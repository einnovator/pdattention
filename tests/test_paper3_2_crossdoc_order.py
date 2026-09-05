from __future__ import annotations

import gzip
import json

from experiments.paper3_2_rag.aggregate_crossdoc_order import aggregate
from experiments.paper3_2_rag.run_prerope_causal_decomposition import (
    _distribution_snapshot,
)


def _manifest(root, order: str, prediction: str):
    run = root / order
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps({"seed": 11, "record_order": order}), encoding="utf-8"
    )
    snapshot = {
        "token_ids": [1, 2],
        "probabilities": [0.6, 0.3] if order == "canonical" else [0.5, 0.4],
        "tail_probability": 0.1,
    }
    row = {
        "condition": "D_GIST_SA_APPEND",
        "example_id": "q1",
        "prediction": prediction,
        "token_f1": 0.5,
        "gold_answer_mean_nll": 2.0,
        "first_step_distribution_topk": snapshot,
    }
    with gzip.open(run / "condition_results.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    return run / "manifest.json"


def test_distribution_snapshot_and_order_aggregate(tmp_path) -> None:
    snapshot = _distribution_snapshot([0.0, 1.0, 2.0], 2)
    assert snapshot is not None
    assert snapshot["token_ids"] == [2, 1]
    assert 0.0 < snapshot["tail_probability"] < 1.0
    result = aggregate(
        (_manifest(tmp_path, "canonical", "a"), _manifest(tmp_path, "reverse", "b"))
    )
    row = next(
        row for row in result["conditions"] if row["condition"] == "D_GIST_SA_APPEND"
    )
    assert row["output_flip_rate"] == 1.0
    assert row["mean_pairwise_topk_tail_js"] > 0.0
