"""Generate Paper 6 table and throughput curve from the APC cohort."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _rate(values: list[bool]) -> str:
    return f"{sum(values)}/{len(values)}"


def _rows(payload: dict, condition: str, concurrency: int | None = None) -> list[dict]:
    return [
        row
        for row in payload["rows"]
        if row["condition"] == condition
        and (concurrency is None or row["concurrency"] == concurrency)
    ]


def _outputs(rows: list[dict]) -> list[dict]:
    return [output for row in rows for output in row["outputs"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    table_rows: list[tuple[str, int, str, float, str]] = []
    for concurrency in payload["concurrency_levels"]:
        rows = _rows(payload, "native_pra_plus_apc", concurrency)
        outputs = _outputs(rows)
        cached = sorted({int(output["num_cached_tokens"]) for output in outputs})
        table_rows.append(
            (
                f"Native PRA + APC, $C={concurrency}$",
                len(outputs),
                _rate([bool(output["exact_recovery"]) for output in outputs]),
                statistics.mean(row["requests_per_second"] for row in rows),
                f"{cached[0]}--{cached[-1]}",
            )
        )

    mixed_rows = _rows(payload, "mixed_selected_and_ordinary")
    mixed_outputs = _outputs(mixed_rows)
    mixed_success = [
        bool(output["exact_recovery"])
        if output["selected_registered"]
        else not bool(output["exact_recovery"])
        for output in mixed_outputs
    ]
    table_rows.append(
        (
            "Mixed selected / ordinary",
            len(mixed_outputs),
            _rate(mixed_success),
            statistics.mean(row["requests_per_second"] for row in mixed_rows),
            "144--144",
        )
    )

    disabled_rows = _rows(payload, "disabled_after_native")
    disabled_outputs = _outputs(disabled_rows)
    table_rows.append(
        (
            "Post-native disabled isolation",
            len(disabled_outputs),
            _rate([not bool(output["exact_recovery"]) for output in disabled_outputs]),
            statistics.mean(row["requests_per_second"] for row in disabled_rows),
            "144--144",
        )
    )

    wrong_rows = _rows(payload, "wrong_memory_plus_apc")
    wrong_outputs = _outputs(wrong_rows)
    table_rows.append(
        (
            "Wrong-memory redirection",
            len(wrong_outputs),
            _rate(
                [
                    bool(output["wrong_memory_follows_wrong_code"])
                    for output in wrong_outputs
                ]
            ),
            statistics.mean(row["requests_per_second"] for row in wrong_rows),
            "144--144",
        )
    )

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Condition & $n$ & Success & Requests/s & APC tokens \\",
        r"\midrule",
    ]
    lines.extend(
        f"{name} & {count} & {success} & {throughput:.2f} & {cached} \\\\"
        for name, count, success, throughput, cached in table_rows
    )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib.pyplot as plt

    concurrency = payload["concurrency_levels"]
    means = [
        statistics.mean(
            row["requests_per_second"]
            for row in _rows(payload, "native_pra_plus_apc", value)
        )
        for value in concurrency
    ]
    lows = [
        min(
            row["requests_per_second"]
            for row in _rows(payload, "native_pra_plus_apc", value)
        )
        for value in concurrency
    ]
    highs = [
        max(
            row["requests_per_second"]
            for row in _rows(payload, "native_pra_plus_apc", value)
        )
        for value in concurrency
    ]
    figure, axis = plt.subplots(figsize=(5.6, 3.2))
    axis.plot(concurrency, means, marker="o", color="#176b87", linewidth=2)
    axis.fill_between(concurrency, lows, highs, color="#64a6bd", alpha=0.25)
    axis.set_xlabel("Concurrent requests")
    axis.set_ylabel("Completed requests / second")
    axis.set_xticks(concurrency)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.plot)
    plt.close(figure)


if __name__ == "__main__":
    main()
