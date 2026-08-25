"""Fast artifact and accounting tests for the runtime benchmark."""

from __future__ import annotations

import csv
import json

from pra_hf.runtime_benchmark import run_runtime_microbenchmark, write_runtime_benchmark


def test_cpu_microbenchmark_emits_raw_summary_cache_and_capabilities(tmp_path):
    result = run_runtime_microbenchmark(
        device="cpu",
        candidate_tokens=64,
        selected_tokens=8,
        batches=(1,),
        kv_heads=2,
        head_dim=4,
        warmups=0,
        repeats=2,
        include_compile=False,
    )
    paths = write_runtime_benchmark(result, tmp_path)

    assert result["protocol"]["quality_selection_frozen"] is True
    assert {"indexed_gather", "interval_pack", "layout_build", "layout_gather"} <= {
        row["study"] for row in result["rows"]
    }
    assert any(row["status"] == "measured" for row in result["summary"])
    assert result["cache"]["hits"] > 0
    assert all(path.is_file() for path in paths.values())
    assert json.loads(paths["json"].read_text())["protocol"]["candidate_tokens"] == 64
    with paths["summary"].open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) >= 2
