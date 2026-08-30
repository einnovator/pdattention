"""Summarize the larger five-seed routed natural-QA cohort."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
from statistics import fmean


DATASETS = ("qasper", "hotpotqa", "2wikimultihopqa")


def summarize(payloads: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for payload in payloads:
        rows = list(payload["rows"])
        routed = [row for row in rows if row["condition"] == "routed_native"]
        by_condition: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            by_condition[str(row["condition"])].append(row)
        for condition, selected in sorted(by_condition.items()):
            result.append(
                {
                    "dataset": payload["dataset"],
                    "condition": condition,
                    "sampled_examples": len(selected),
                    "unique_examples": len({row["example_id"] for row in selected}),
                    "evidence_recall_at_4": fmean(
                        float(row["evidence_recall_at_4"]) for row in routed
                    ),
                    "token_f1": fmean(float(row["token_f1"]) for row in selected),
                    "gold_answer_logprob": fmean(
                        float(row["gold_answer_logprob"]) for row in selected
                    ),
                    "routing_ms": fmean(float(row["routing_ms"]) for row in routed),
                    "index_build_ms": fmean(
                        float(row["index_build_ms"]) for row in routed
                    ),
                }
            )
    return result


def _write_table(path: Path, rows: list[dict[str, object]]) -> None:
    labels = {
        "qasper": "QASPER",
        "hotpotqa": "HotpotQA",
        "2wikimultihopqa": "2Wiki",
    }
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Dataset & Condition & Sampled/unique & Recall@4 & F1 & $\log p(y)$ & Route ms \\",
        r"\midrule",
    ]
    for row in rows:
        condition = str(row["condition"]).replace("_", r"\_")
        lines.append(
            f"{labels[str(row['dataset'])]} & "
            f"{condition} & "
            f"{row['sampled_examples']}/{row['unique_examples']} & "
            f"{float(row['evidence_recall_at_4']):.3f} & "
            f"{float(row['token_f1']):.3f} & "
            f"{float(row['gold_answer_logprob']):.2f} & "
            f"{float(row['routing_ms']):.2f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payloads = [
        json.loads(
            (args.input_dir / f"routed_answer_quality_{dataset}.json").read_text(
                encoding="utf-8"
            )
        )
        for dataset in DATASETS
    ]
    rows = summarize(payloads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "expanded_routed_summary.json").write_text(
        json.dumps({"schema_version": "1.0", "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "expanded_routed_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_table(args.output_dir / "generated_expanded_routed_table.tex", rows)


if __name__ == "__main__":
    main()
