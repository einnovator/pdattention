"""Generate Paper 6.3 tables and plots from the distractor ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


DATASET_NAMES = {
    "qasper": "QASPER",
    "hotpotqa": "HotpotQA",
    "2wikimultihopqa": "2Wiki",
}


def summarize(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Retain quality, latency, and context growth as separate measurements."""

    rows = []
    for aggregate in payload["aggregates"]:
        condition = str(aggregate["condition"])
        if condition == "evidence_only":
            mode, count = "evidence_only", 0
        else:
            mode, count_text = condition.split("_distractors_k", 1)
            count = int(count_text)
        rows.append(
            {
                "dataset": str(aggregate["dataset"]),
                "mode": mode,
                "distractor_count": count,
                "sample_count": int(aggregate["sample_count"]),
                "token_f1": float(aggregate["token_f1"]),
                "exact_match": float(aggregate["exact_match"]),
                "answer_containment": float(aggregate["answer_containment"]),
                "evidence_recall_at_4": float(aggregate["evidence_recall_at_4"]),
                "mean_source_tokens": float(aggregate["mean_source_tokens"]),
                "mean_distractor_tokens": float(aggregate["mean_distractor_tokens"]),
                "ttft_p50_ms": float(aggregate["ttft_ms"]["p50"]),
                "ttft_p95_ms": float(aggregate["ttft_ms"]["p95"]),
                "completion_p50_ms": float(
                    aggregate["completion_latency_ms"]["p50"]
                ),
                "successful_requests_per_second": float(
                    aggregate["successful_requests_per_second"]
                ),
            }
        )
    return {
        "schema_version": "paper6.3-openvino-distractor-summary-v1",
        "source_schema_version": payload.get("schema_version"),
        "evidence_tier": payload.get("evidence_tier"),
        "engine_version": payload.get("engine_version"),
        "model_id": payload.get("model_id"),
        "device": payload.get("device"),
        "selector_frozen": bool(payload.get("selector_frozen")),
        "request_count": len(payload.get("rows") or ()),
        "example_count": len(
            {
                (str(row["dataset"]), str(row["example_id"]))
                for row in payload.get("rows") or ()
            }
        ),
        "rows": rows,
    }


def render_table(summary: Mapping[str, Any]) -> str:
    """Render the quality surface compactly; serving metrics stay separate."""

    indexed = {
        (str(row["dataset"]), str(row["mode"]), int(row["distractor_count"])): row
        for row in summary["rows"]
    }
    counts = (1, 2, 4, 8)
    lines = [
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Dataset & Evidence & Rel. 1 & Rel. 2 & Rel. 4 & Rel. 8 & Irr. 1 & Irr. 2 & Irr. 4 & Irr. 8 \\",
        r"\midrule",
    ]
    for dataset in ("qasper", "hotpotqa", "2wikimultihopqa"):
        evidence = indexed[(dataset, "evidence_only", 0)]["token_f1"]
        relevant = [indexed[(dataset, "relevant", count)]["token_f1"] for count in counts]
        irrelevant = [indexed[(dataset, "irrelevant", count)]["token_f1"] for count in counts]
        values = [evidence, *relevant, *irrelevant]
        lines.append(
            f"{DATASET_NAMES[dataset]} & "
            + " & ".join(f"{float(value):.3f}" for value in values)
            + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def plot(summary: Mapping[str, Any], output: Path) -> None:
    """Plot quality and TTFT against distractor count without merging axes."""

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharex="col")
    colors = {"relevant": "#b13f46", "irrelevant": "#286f9e"}
    for column, dataset in enumerate(("qasper", "hotpotqa", "2wikimultihopqa")):
        dataset_rows = [row for row in summary["rows"] if row["dataset"] == dataset]
        evidence = next(row for row in dataset_rows if row["mode"] == "evidence_only")
        for mode in ("relevant", "irrelevant"):
            selected = sorted(
                (row for row in dataset_rows if row["mode"] == mode),
                key=lambda row: row["distractor_count"],
            )
            x = [0, *(row["distractor_count"] for row in selected)]
            axes[0, column].plot(
                x,
                [evidence["token_f1"], *(row["token_f1"] for row in selected)],
                marker="o",
                color=colors[mode],
                label=mode,
            )
            axes[1, column].plot(
                x,
                [evidence["ttft_p50_ms"], *(row["ttft_p50_ms"] for row in selected)],
                marker="o",
                color=colors[mode],
            )
        axes[0, column].set_title(DATASET_NAMES[dataset])
        axes[0, column].grid(alpha=0.25)
        axes[1, column].grid(alpha=0.25)
        axes[1, column].set_xlabel("distractor documents")
    axes[0, 0].set_ylabel("token F1")
    axes[1, 0].set_ylabel("TTFT p50 (ms)")
    axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--plot", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    summary = summarize(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(render_table(summary), encoding="utf-8")
    plot(summary, args.plot)


if __name__ == "__main__":
    main()
