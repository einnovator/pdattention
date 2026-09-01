"""Summarize the shared MLX context-dilution and positional-control campaign."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean


def _mean(rows: list[dict], field: str) -> float:
    return fmean(float(row[field]) for row in rows)


def summarize(paths: list[Path]) -> dict[str, object]:
    rows = []
    models = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        models[str(payload["model_id"])] = payload.get("runtime", {})
        rows.extend(row for row in payload["rows"] if row.get("status") == "MEASURED")

    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["model_id"]), int(row["context_target_tokens"]), str(row["condition"]))
        ].append(row)
    aggregate = []
    for (model, target, condition), values in sorted(grouped.items()):
        aggregate.append(
            {
                "model_id": model,
                "context_target_tokens": target,
                "condition": condition,
                "samples": len(values),
                "datasets": len({str(row["dataset"]) for row in values}),
                "token_f1": _mean(values, "token_f1"),
                "gold_answer_logprob": _mean(values, "gold_answer_logprob"),
                "completion_latency_ms": _mean(values, "completion_latency_ms"),
                "first_token_agreement_vs_full": _mean(
                    values, "first_token_agreement_vs_full"
                ),
                "sequence_agreement_vs_full": _mean(
                    values, "sequence_agreement_vs_full"
                ),
                "first_logit_rmse_vs_full": _mean(values, "first_logit_rmse_vs_full"),
                "active_detail_bytes": _mean(values, "active_detail_bytes"),
                "visible_prompt_tokens": _mean(values, "visible_prompt_tokens")
                if all("visible_prompt_tokens" in row for row in values)
                else None,
                "peak_unified_memory_bytes": max(
                    int(row["peak_unified_memory_bytes"]) for row in values
                ),
            }
        )

    by_key = {
        (row["model_id"], row["context_target_tokens"], row["condition"]): row
        for row in aggregate
    }
    comparisons = []
    raw_by_key = {
        (
            str(row["model_id"]),
            int(row["context_target_tokens"]),
            str(row["dataset"]),
            str(row["example_id"]),
            str(row["condition"]),
        ): row
        for row in rows
        if "example_id" in row
    }
    for model in sorted(models):
        targets = sorted(
            target
            for candidate, target, condition in by_key
            if candidate == model and condition == "FULL_VISIBLE"
        )
        for target in targets:
            full = by_key[(model, target, "FULL_VISIBLE")]
            selected = by_key.get((model, target, "E0_SELECTED"))
            selected_native = by_key.get((model, target, "E2_SELECTED"))
            source = by_key.get((model, target, "E2_SOURCE_RELATIVE"))
            restart = by_key.get((model, target, "E2_QUERY_RESTART"))
            selected_pair_keys = {
                (dataset, example_id)
                for candidate, candidate_target, dataset, example_id, condition
                in raw_by_key
                if candidate == model
                and candidate_target == target
                and condition == "E0_SELECTED"
            }
            selected_pair_agreement = [
                float(
                    raw_by_key[(model, target, dataset, example_id, "E0_SELECTED")][
                        "output_token_ids"
                    ]
                    == raw_by_key[(model, target, dataset, example_id, "E2_SELECTED")][
                        "output_token_ids"
                    ]
                )
                for dataset, example_id in selected_pair_keys
                if (model, target, dataset, example_id, "E2_SELECTED") in raw_by_key
            ]
            comparisons.append(
                {
                    "model_id": model,
                    "context_target_tokens": target,
                    "samples": full["samples"],
                    "full_minus_selected_gold_logprob": (
                        full["gold_answer_logprob"] - selected["gold_answer_logprob"]
                        if selected else None
                    ),
                    "full_minus_selected_f1": (
                        full["token_f1"] - selected["token_f1"] if selected else None
                    ),
                    "selected_native_minus_text_gold_logprob": (
                        selected_native["gold_answer_logprob"]
                        - selected["gold_answer_logprob"]
                        if selected and selected_native else None
                    ),
                    "selected_native_minus_text_f1": (
                        selected_native["token_f1"] - selected["token_f1"]
                        if selected and selected_native else None
                    ),
                    "selected_native_sequence_agreement": (
                        fmean(selected_pair_agreement)
                        if selected_pair_agreement else None
                    ),
                    "selected_native_active_detail_bytes": (
                        selected_native["active_detail_bytes"]
                        if selected_native else None
                    ),
                    "source_relative_logit_rmse": (
                        source["first_logit_rmse_vs_full"] if source else None
                    ),
                    "query_restart_logit_rmse": (
                        restart["first_logit_rmse_vs_full"] if restart else None
                    ),
                    "source_relative_sequence_agreement": (
                        source["sequence_agreement_vs_full"] if source else None
                    ),
                    "query_restart_sequence_agreement": (
                        restart["sequence_agreement_vs_full"] if restart else None
                    ),
                }
            )
    return {
        "schema_version": "pra-mac-long-context-summary-v1",
        "models": models,
        "aggregate": aggregate,
        "comparisons": comparisons,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _short_model(model: str) -> str:
    return model.rsplit("/", 1)[-1].replace("-4bit", "")


def _write_table(path: Path, comparisons: list[dict]) -> None:
    rows = [row for row in comparisons if row["context_target_tokens"] in {8192, 32768}]
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Model & Context & $\Delta$LP full--selected & source RMSE & restart RMSE & source agr. & restart agr. \\",
        r"\midrule",
    ]
    for row in rows:
        value = row["full_minus_selected_gold_logprob"]
        lines.append(
            f"{_short_model(str(row['model_id']))} & "
            f"{int(row['context_target_tokens']) // 1024}K & "
            f"{value:+.3f} & "
            f"{row['source_relative_logit_rmse']:.3f} & "
            f"{row['query_restart_logit_rmse']:.3f} & "
            f"{row['source_relative_sequence_agreement']:.3f} & "
            f"{row['query_restart_sequence_agreement']:.3f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(path: Path, comparisons: list[dict]) -> None:
    import matplotlib.pyplot as plt

    models = sorted({str(row["model_id"]) for row in comparisons})
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for model in models:
        values = sorted(
            (row for row in comparisons if row["model_id"] == model),
            key=lambda row: row["context_target_tokens"],
        )
        x = [row["context_target_tokens"] / 1024 for row in values]
        axes[0].plot(
            x,
            [row["full_minus_selected_gold_logprob"] for row in values],
            marker="o",
            label=_short_model(model),
        )
        axes[1].plot(
            x,
            [row["source_relative_logit_rmse"] for row in values],
            marker="o",
            label=f"{_short_model(model)} source",
        )
        axes[1].plot(
            x,
            [row["query_restart_logit_rmse"] for row in values],
            marker="x",
            linestyle="--",
            label=f"{_short_model(model)} restart",
        )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Visible full-context tokens (K)")
    axes[0].set_ylabel("Gold logP: full minus selected")
    axes[0].set_xscale("log", base=2)
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("Transport-control source tokens (K)")
    axes[1].set_ylabel("First-logit RMSE vs ordinary prefix")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=6, ncol=2)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = summarize(args.inputs)
    (args.output_dir / "mlx_long_context_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir / "mlx_long_context_aggregate.csv", payload["aggregate"])
    _write_csv(args.output_dir / "mlx_long_context_comparisons.csv", payload["comparisons"])
    _write_table(args.output_dir / "generated_mlx_long_context_table.tex", payload["comparisons"])
    _plot(args.output_dir / "mlx_long_context", payload["comparisons"])


if __name__ == "__main__":
    main()
