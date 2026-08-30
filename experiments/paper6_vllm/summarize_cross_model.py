"""Build the vLLM cross-model matched E0/E2 replication table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean


REGIMES = (
    "cold_one_shot",
    "warm_repeated",
    "multi_query_same_resource",
    "concurrent_shared_resource",
)


def _cohort_row(model_id: str, summary: dict[str, object]) -> dict[str, object]:
    """Reduce one already-normalized matched benchmark without mixing engines."""

    parity = [row for row in summary["parity"] if row["engine"] == "vllm-metal"]
    aggregates = [
        row for row in summary["aggregates"] if row["engine"] == "vllm-metal"
    ]
    paired = sum(int(row["paired_requests"]) for row in parity)
    exact = sum(
        int(row["paired_requests"]) * float(row["exact_output_parity"])
        for row in parity
    )
    result: dict[str, object] = {
        "model_id": model_id,
        "datasets": len({str(row["dataset"]) for row in parity}),
        "paired_requests": paired,
        "exact_pairs": int(round(exact)),
        "exact_output_parity": exact / paired,
    }
    for regime in REGIMES:
        ratios = []
        for dataset in sorted({str(row["dataset"]) for row in parity}):
            rows = [
                row
                for row in aggregates
                if row["regime"] == regime and row["dataset"] == dataset
            ]
            by_condition = {str(row["condition"]): row for row in rows}
            e0 = by_condition["e0_selected_text"]
            e2 = by_condition["e2_native_kv"]
            if regime == "cold_one_shot":
                ratio = float(e2["cold_end_to_end_completion_ms"]) / float(
                    e0["cold_end_to_end_completion_ms"]
                )
            elif regime == "concurrent_shared_resource":
                ratio = float(e0["requests_per_second"]) / float(
                    e2["requests_per_second"]
                )
            else:
                ratio = float(e2["completion_p50_ms"]) / float(
                    e0["completion_p50_ms"]
                )
            ratios.append(ratio)
        result[f"{regime}_e2_over_e0"] = fmean(ratios)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort",
        nargs=2,
        action="append",
        metavar=("MODEL_ID", "SUMMARY_JSON"),
        required=True,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = [
        _cohort_row(
            model_id,
            json.loads(Path(summary_path).read_text(encoding="utf-8")),
        )
        for model_id, summary_path in args.cohort
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cross_model_matched_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment": "paper6_vllm_cross_model_matched_e0_e2_v1",
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "cross_model_matched_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & Exact pairs & Cold & Warm & Multi-query & Concurrent \\",
        r"\midrule",
    ]
    for row in rows:
        label = str(row["model_id"]).split("/")[-1].replace("_", r"\_")
        table.append(
            f"{label} & {row['exact_pairs']}/{row['paired_requests']} & "
            f"{float(row['cold_one_shot_e2_over_e0']):.3f} & "
            f"{float(row['warm_repeated_e2_over_e0']):.3f} & "
            f"{float(row['multi_query_same_resource_e2_over_e0']):.3f} & "
            f"{float(row['concurrent_shared_resource_e2_over_e0']):.3f} \\\\"
        )
    table.extend((r"\bottomrule", r"\end{tabular}"))
    (args.output_dir / "generated_cross_model_matched_table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
