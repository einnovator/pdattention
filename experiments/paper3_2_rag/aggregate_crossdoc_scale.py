"""Aggregate reduced Paper 3.2 cross-document model/family replications."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
from pathlib import Path
from typing import Mapping, Sequence


CONDITIONS = (
    "A_FULL_CAUSAL_RAG",
    "C_PRA_PRE_ROPE_EXACT_PACKED_OFFSETS",
    "D_GIST_SA_APPEND",
    "E_GIST_SA_BOUNDARY_8",
    "F_GIST_SA_BOUNDARY_32",
)


def _mean(rows: Sequence[Mapping[str, object]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.fmean(values) if values else None


def aggregate(paths: Sequence[Path]) -> dict[str, object]:
    if not paths:
        raise ValueError("scale aggregation requires at least one manifest")
    summaries = []
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        with gzip.open(
            path.parent / "condition_results.jsonl.gz", "rt", encoding="utf-8"
        ) as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        for condition in CONDITIONS:
            selected = [row for row in rows if row["condition"] == condition]
            if not selected:
                continue
            summaries.append(
                {
                    "model": manifest["model"],
                    "model_revision": manifest["model_revision"],
                    "precision_mode": manifest["precision"]["precision_mode"],
                    "seed": manifest["seed"],
                    "condition": condition,
                    "examples": len(selected),
                    "token_f1": _mean(selected, "token_f1"),
                    "official_score": _mean(
                        selected, "official_multihop_rag_score"
                    ),
                    "gold_answer_nll": _mean(selected, "gold_answer_mean_nll"),
                    "first_step_js": _mean(selected, "first_step_js_divergence"),
                    "request_composition_ms": _mean(
                        selected, "request_composition_ms"
                    ),
                    "request_local_native_tokens": _mean(
                        selected, "request_local_native_tokens"
                    ),
                    "request_composition_flops_estimate": _mean(
                        selected, "request_composition_flops_estimate"
                    ),
                }
            )
    return {
        "schema_version": "paper3.2-crossdoc-scale-aggregate-v1",
        "experiment": "crossdoc_composition_reduced_scale",
        "conditions": summaries,
        "source_manifests": [str(path) for path in paths],
    }


def _short_model(model: str) -> str:
    return (
        model.rsplit("/", 1)[-1]
        .replace("-Instruct", "")
        .replace("-4bit", "")
    )


def _condition_label(condition: str) -> str:
    return {
        "A_FULL_CAUSAL_RAG": "A full",
        "C_PRA_PRE_ROPE_EXACT_PACKED_OFFSETS": "C pre-RoPE",
        "D_GIST_SA_APPEND": "D gist append",
        "E_GIST_SA_BOUNDARY_8": "E boundary-8",
        "F_GIST_SA_BOUNDARY_32": "F boundary-32",
    }.get(condition, condition)


def _write(result: Mapping[str, object], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = list(result["conditions"])  # type: ignore[arg-type]
    fields = tuple(rows[0]) if rows else ()
    with (output / "scale_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    table = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Model & Condition & $n$ & Token F1 & Official & Gold NLL \\\\",
        "\\midrule",
    ]
    for row in rows:
        table.append(
            f"{_short_model(str(row['model']))} & "
            f"{_condition_label(str(row['condition']))} & "
            f"{int(row['examples'])} & "
            f"{float(row['token_f1']):.3f} & "
            f"{float(row['official_score']):.3f} & "
            f"{float(row['gold_answer_nll']):.3f} \\\\"
        )
    table.extend(("\\bottomrule", "\\end{tabular}"))
    (output / "generated_scale_table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )

    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(str(row["model"]) for row in rows))
    conditions = list(dict.fromkeys(str(row["condition"]) for row in rows))
    width = 0.8 / max(len(conditions), 1)
    figure, axis = plt.subplots(figsize=(10.5, 4.2))
    for index, condition in enumerate(conditions):
        values = [
            next(
                float(row["token_f1"])
                for row in rows
                if row["model"] == model and row["condition"] == condition
            )
            for model in models
        ]
        positions = [item + (index - (len(conditions) - 1) / 2) * width for item in range(len(models))]
        axis.bar(
            positions,
            values,
            width=width,
            label=_condition_label(condition),
        )
    axis.set_xticks(range(len(models)), [_short_model(model) for model in models])
    axis.set_ylabel("Token F1")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=len(conditions))
    figure.tight_layout()
    figure.savefig(output / "crossdoc_scale_quality.pdf", bbox_inches="tight")
    figure.savefig(output / "crossdoc_scale_quality.png", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _write(aggregate(args.manifest), args.output)


if __name__ == "__main__":
    main()
