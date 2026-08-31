"""Create Paper 4.5-compatible AirLLM product rows from raw artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx", type=Path, required=True)
    parser.add_argument("--controlled", type=Path, required=True)
    parser.add_argument("--cuda", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mlx = json.loads(args.mlx.read_text(encoding="utf-8"))
    controlled = json.loads(args.controlled.read_text(encoding="utf-8"))
    cuda = (
        json.loads(args.cuda.read_text(encoding="utf-8"))
        if args.cuda and args.cuda.exists()
        else None
    )
    rows = mlx["rows"]
    full = next(row for row in rows if row["condition"] == "full_context" and row["distractor_tokens"] == 512)
    selected = next(row for row in rows if row["condition"] == "selected_text" and row["distractor_tokens"] == 512)
    product_rows = [
        {
            "engine": "airllm",
            "backend": "mlx",
            "hardware": "Apple M5 16GB",
            "model": mlx["model"],
            "pra_level": "E0",
            "profile": "SELECTED_TEXT",
            "evidence_tier": mlx["evidence_tier"],
            "status": "SMOKE",
            "source_visible_tokens": full["input_tokens"],
            "selected_visible_tokens": selected["input_tokens"],
            "visible_token_reduction": 1.0 - selected["input_tokens"] / full["input_tokens"],
            "ttft_seconds": selected["ttft_seconds"],
            "tokens_per_second": selected["tokens_per_second"],
            "shard_bytes": mlx["shard_bytes"],
            "minimum_measured_memory_bytes": mlx["process_rss_after_bytes"],
            "quality": "12-token answer-prefix agreement",
        },
        {
            "engine": "airllm",
            "backend": "controlled_hf_lifecycle",
            "hardware": "parameterized",
            "model": "8L/4KVH/64Dh control",
            "pra_level": "E2_LIFECYCLE_ONLY",
            "profile": "BALANCED_LAYER_STREAMED",
            "evidence_tier": controlled["evidence_tier"],
            "status": "CONTROLLED_MODEL",
            "source_visible_tokens": 65536,
            "selected_native_tokens": 128,
            "peak_working_set_mib": controlled["native_64k_peak_mib"],
            "full_context_peak_mib": controlled["full_64k_peak_mib"],
            "quality": None,
        },
    ]
    if cuda is not None:
        product_rows.append(
            {
                "engine": "airllm",
                "backend": "hf_cuda",
                "hardware": cuda.get("device"),
                "model": cuda.get("model_id") or mlx["model"],
                "pra_level": "E1",
                "profile": "REFERENCE_CORRECTNESS",
                "evidence_tier": cuda.get("evidence_tier"),
                "status": cuda.get("status"),
                "disabled_exact_sequence": cuda.get("disabled_exact_sequence"),
                "error": cuda.get("error"),
            }
        )
    output = {
        "schema_version": "paper4.5-product-row-v1",
        "paper": "6.6",
        "rows": product_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
