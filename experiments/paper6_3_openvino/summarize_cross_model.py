"""Compare selector-frozen OpenVINO E0 evidence across model sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


_DATASET_NAMES = {
    "qasper": "QASPER",
    "hotpotqa": "HotpotQA",
    "2wikimultihopqa": "2Wiki",
}


def _model_label(model_id: str) -> str:
    """Return a compact, stable label without relying on input ordering."""

    name = model_id.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    lowered = name.lower()
    if "tinyllama" in lowered:
        return "TinyLlama 1.1B"
    if "qwen2.5-1.5b" in lowered:
        return "Qwen2.5 1.5B"
    if "qwen2-0.5b" in lowered:
        return "Qwen2 0.5B"
    return name.replace("-Instruct", "").replace("-int4-ov", "")


def _p50(value: Any) -> float:
    return float(value["p50"] if isinstance(value, Mapping) else value)


def summarize(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build paired selected/full rows for every model and dataset."""

    rows: list[dict[str, Any]] = []
    for payload in payloads:
        model_id = str(payload["model_id"])
        indexed = {
            (str(row["dataset"]), str(row["condition"])): row
            for row in payload["aggregates"]
        }
        datasets = sorted({dataset for dataset, _ in indexed})
        for dataset in datasets:
            selected = indexed[(dataset, "selected_context")]
            full = indexed[(dataset, "full_context")]
            selected_ttft = _p50(selected["ttft_ms"])
            full_ttft = _p50(full["ttft_ms"])
            rows.append(
                {
                    "model_id": model_id,
                    "dataset": dataset,
                    "samples_per_condition": min(
                        int(selected["sample_count"]), int(full["sample_count"])
                    ),
                    "selected_f1": float(selected["token_f1"]),
                    "full_f1": float(full["token_f1"]),
                    "selected_answer_containment": float(
                        selected["answer_containment"]
                    ),
                    "full_answer_containment": float(full["answer_containment"]),
                    "selected_minus_full_f1": float(selected["token_f1"])
                    - float(full["token_f1"]),
                    "selected_ttft_p50_ms": selected_ttft,
                    "full_ttft_p50_ms": full_ttft,
                    "selected_ttft_p95_ms": float(selected["ttft_ms"]["p95"]),
                    "full_ttft_p95_ms": float(full["ttft_ms"]["p95"]),
                    "full_over_selected_ttft": full_ttft
                    / max(selected_ttft, 1e-12),
                    "selected_prompt_tokens": float(selected["mean_prompt_tokens"]),
                    "full_prompt_tokens": float(full["mean_prompt_tokens"]),
                }
            )
    return {
        "schema_version": "paper6.3-openvino-cross-model-v1",
        "integration_level": "E0_SELECTED_TEXT",
        "selector_frozen": True,
        "engine": "openvino_genai",
        "engine_version": payloads[0].get("engine_version"),
        "device": payloads[0].get("device"),
        "evidence_tier": payloads[0].get("evidence_tier"),
        "models": [str(payload["model_id"]) for payload in payloads],
        "rows": rows,
    }


def render_table(summary: Mapping[str, Any]) -> str:
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Model & Dataset & Sel. F1 & Full F1 & Sel. TTFT & Full TTFT & TTFT ratio \\",
        r"\midrule",
    ]
    for model_id in summary["models"]:
        label = _model_label(str(model_id))
        for row in summary["rows"]:
            if row["model_id"] != model_id:
                continue
            lines.append(
                f"{label} & {_DATASET_NAMES.get(row['dataset'], row['dataset'])} & "
                f"{row['selected_f1']:.3f} & {row['full_f1']:.3f} & "
                f"{row['selected_ttft_p50_ms']:.0f} & {row['full_ttft_p50_ms']:.0f} & "
                f"{row['full_over_selected_ttft']:.2f}$\\times$ \\\\"
            )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def render_plot(summary: Mapping[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    datasets = ("qasper", "hotpotqa", "2wikimultihopqa")
    models = list(summary["models"])
    indexed = {
        (str(row["model_id"]), str(row["dataset"])): row
        for row in summary["rows"]
    }
    x = np.arange(len(datasets), dtype=float)
    bar_count = max(1, len(models) * 2)
    width = min(0.18, 0.82 / bar_count)
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    for model_index, model in enumerate(models):
        model_label = _model_label(str(model))
        for condition_index, condition in enumerate(("selected", "full")):
            offset_index = model_index * 2 + condition_index
            offset = (offset_index - (bar_count - 1) / 2) * width
            rows = [indexed[(model, dataset)] for dataset in datasets]
            axes[0].bar(
                x + offset,
                [row[f"{condition}_f1"] for row in rows],
                width,
                color=colors[offset_index % len(colors)],
                label=f"{model_label} {condition}",
            )
            axes[1].bar(
                x + offset,
                [row[f"{condition}_ttft_p50_ms"] for row in rows],
                width,
                color=colors[offset_index % len(colors)],
            )
    axes[0].set_ylabel("Token F1")
    axes[1].set_ylabel("Median TTFT (ms)")
    for axis in axes:
        axis.set_xticks(x, [_DATASET_NAMES[name] for name in datasets])
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--plot", required=True, type=Path)
    args = parser.parse_args()
    summary = summarize(
        [json.loads(path.read_text(encoding="utf-8-sig")) for path in args.inputs]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.table.write_text(render_table(summary), encoding="utf-8")
    render_plot(summary, args.plot)


if __name__ == "__main__":
    main()
