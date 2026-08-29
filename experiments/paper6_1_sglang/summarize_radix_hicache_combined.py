"""Generate the Paper 6.1 combined Radix/HiCache table."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


LABELS = {
    "selected_A": "A: selected after L2 promotion",
    "ordinary_B_after_cleanup": "B: ordinary after cleanup",
    "reselected_C": "C: reselected from warm L1",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    conditions = [
        condition
        for row in payload["rows"]
        for condition in row["conditions"]
    ]
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Request & Recovery & Radix separated & One copy & Mean ms \\",
        r"\midrule",
    ]
    for name, label in LABELS.items():
        rows = [row for row in conditions if row["condition"] == name]
        recovery = sum(bool(row["exact_recovery"]) for row in rows)
        separation = sum(
            bool(row["selected_tokens_excluded_from_radix_length"])
            for row in rows
        )
        one_copy = [row["exactly_one_selected_copy"] for row in rows]
        one_copy_text = (
            "--"
            if all(value is None for value in one_copy)
            else f"{sum(value is True for value in one_copy)}/{len(one_copy)}"
        )
        mean_ms = statistics.mean(row["completion_latency_ms"] for row in rows)
        lines.append(
            f"{label} & {recovery}/{len(rows)} & "
            f"{separation}/{len(rows)} & {one_copy_text} & {mean_ms:.1f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
