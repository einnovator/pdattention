"""Build required cross-evaluation tables, statistics, figures, and TeX macros."""

from __future__ import annotations

import csv
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/papers/shared/results/paper7_records/headroom_cross_eval"
FIGURES = OUTPUT / "figures"


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    fields = sorted({key for row in values for key in row}) if values else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _number(row: Mapping[str, str], field: str) -> float | None:
    value = row.get(field, "")
    try:
        return float(value) if value != "" else None
    except (TypeError, ValueError):
        return None


def _mean(rows: Sequence[Mapping[str, str]], field: str) -> float:
    values = [value for row in rows if (value := _number(row, field)) is not None]
    return statistics.fmean(values) if values else math.nan


def _bootstrap_case_ci(
    rows: Sequence[Mapping[str, str]], field: str, seed: int = 20260828
) -> tuple[float, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _number(row, field)
        if value is not None:
            grouped[row["case_id"]].append(value)
    case_means = [statistics.fmean(values) for values in grouped.values()]
    if not case_means:
        return math.nan, math.nan
    rng = random.Random(seed)
    draws = []
    for _ in range(10_000):
        sample = [rng.choice(case_means) for _ in case_means]
        draws.append(statistics.fmean(sample))
    draws.sort()
    return draws[249], draws[9749]


def _exact_sign_flip(differences: Sequence[float]) -> float:
    nonzero = [value for value in differences if abs(value) > 1e-12]
    if not nonzero:
        return 1.0
    observed = abs(statistics.fmean(nonzero))
    if len(nonzero) <= 20:
        extreme = 0
        total = 0
        for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
            total += 1
            value = abs(statistics.fmean(sign * delta for sign, delta in zip(signs, nonzero)))
            extreme += int(value >= observed - 1e-12)
        return extreme / total
    rng = random.Random(20260828)
    extreme = 0
    for _ in range(100_000):
        value = abs(statistics.fmean(rng.choice((-1.0, 1.0)) * delta for delta in nonzero))
        extreme += int(value >= observed - 1e-12)
    return (extreme + 1) / 100_001


def _summaries(paper7: list[dict[str, str]], external: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in paper7:
        grouped[("paper7", "paper7_c0_c5", row["condition"])].append(row)
    for row in external:
        if row.get("status") == "supported" and row.get("evidence_eligible", "1") != "0":
            grouped[("headroom_eval", row["dataset"], row["condition"])].append(row)
    for (workload, dataset, condition), values in sorted(grouped.items()):
        success_lo, success_hi = _bootstrap_case_ci(values, "task_success")
        rows.append({
            "workload": workload,
            "dataset": dataset,
            "condition": condition,
            "n": len(values),
            "task_success": _mean(values, "task_success"),
            "evidence_recall_at_8": _mean(values, "evidence_recall_at_8"),
            "task_success_ci_low": success_lo,
            "task_success_ci_high": success_hi,
            "active_tokens": _mean(values, "active_tokens"),
            "initial_visible_tokens": _mean(values, "initial_visible_tokens"),
            "retrieved_tokens": _mean(values, "retrieved_tokens"),
            "ingestion_seconds": _mean(values, "ingestion_seconds"),
            "retrieval_seconds": _mean(values, "retrieval_seconds"),
            "total_latency_seconds": _mean(values, "total_latency_seconds"),
            "status": "supported",
        })
    manifest = json.loads((OUTPUT / "headroom_official_manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["compatibility"]:
        if entry["status"] not in {"supported", "supported_smoke"}:
            rows.append({
                "workload": "compatibility",
                "dataset": entry["path"],
                "condition": "HEADROOM_OFFICIAL",
                "n": 0,
                "task_success": "",
                "evidence_recall_at_8": "",
                "task_success_ci_low": "",
                "task_success_ci_high": "",
                "active_tokens": "",
                "initial_visible_tokens": "",
                "retrieved_tokens": "",
                "ingestion_seconds": "",
                "retrieval_seconds": "",
                "total_latency_seconds": "",
                "status": entry["status"],
                "notes": entry["details"],
            })
    return rows


def _trigger(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["condition"] in {
            "HEADROOM_OFFICIAL_DEFAULT",
            "HEADROOM_OFFICIAL_TUNED",
            "CCR_STYLE",
        }:
            grouped[(row["condition"], row["case_class"])].append(row)
    output = []
    for (condition, case_class), values in sorted(grouped.items()):
        needed = sum(int(row["trigger_needed"]) for row in values)
        called = sum(int(row["trigger_called"]) for row in values)
        true_positive = sum(
            int(row["trigger_needed"]) * int(row["trigger_correct"]) for row in values
        )
        output.append({
            "condition": condition,
            "case_class": case_class,
            "n": len(values),
            "trigger_needed": needed,
            "trigger_called": called,
            "trigger_correct": sum(int(row["trigger_correct"]) for row in values),
            "trigger_precision": true_positive / called if called else (1.0 if not needed else 0.0),
            "trigger_recall": true_positive / needed if needed else 1.0,
            "trigger_accuracy": _mean(values, "trigger_correct"),
            "final_task_success": _mean(values, "task_success"),
            "status": "supported",
        })
    return output


def _faithfulness(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    indexed = {(row["condition"], row["case_id"], row["seed"]): row for row in rows}
    output = []
    for profile in ("HEADROOM_OFFICIAL_DEFAULT", "HEADROOM_OFFICIAL_TUNED"):
        for key, official in indexed.items():
            if key[0] != profile:
                continue
            style = indexed.get(("CCR_STYLE", key[1], key[2]))
            if style is None:
                continue
            output.append({
                "official_condition": profile,
                "case_id": key[1],
                "case_class": official["case_class"],
                "seed": key[2],
                "same_trigger_action": int(official["predicted_action"] == style["predicted_action"]),
                "official_success": official["task_success"],
                "ccr_style_success": style["task_success"],
                "success_delta": float(official["task_success"]) - float(style["task_success"]),
                "official_active_tokens": official["active_tokens"],
                "ccr_style_active_tokens": style["active_tokens"],
                "active_token_delta": float(official["active_tokens"]) - float(style["active_tokens"]),
                "status": "paired",
            })
    return output


def _cost(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "active_tokens",
        "initial_visible_tokens",
        "retrieved_tokens",
        "ingestion_seconds",
        "retrieval_seconds",
        "total_latency_seconds",
    )
    rows = []
    for row in summary:
        if row["status"] != "supported":
            rows.append({
                "workload": row["workload"],
                "dataset": row["dataset"],
                "condition": row["condition"],
                "metric": "unsupported",
                "mean": "",
                "status": row["status"],
                "notes": row.get("notes", ""),
            })
            continue
        for field in fields:
            rows.append({
                "workload": row["workload"],
                "dataset": row["dataset"],
                "condition": row["condition"],
                "metric": field,
                "mean": row[field],
                "status": "supported",
            })
    return rows


def _stats(paper7: list[dict[str, str]], external: list[dict[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {"paper7": {}, "headroom_eval": {}}
    by_condition = defaultdict(list)
    for row in paper7:
        by_condition[row["condition"]].append(row)
    for condition, values in by_condition.items():
        lo, hi = _bootstrap_case_ci(values, "task_success")
        result["paper7"][condition] = {
            "n": len(values),
            "task_success": _mean(values, "task_success"),
            "task_success_bootstrap_ci": [lo, hi],
            "active_tokens": _mean(values, "active_tokens"),
        }
    pairs = {}
    for other in ("PRA_FROZEN", "FULL_BACKING", "CCR_STYLE"):
        tuned = {(row["case_id"], row["seed"]): row for row in by_condition["HEADROOM_OFFICIAL_TUNED"]}
        comparison = {(row["case_id"], row["seed"]): row for row in by_condition[other]}
        keys = sorted(set(tuned) & set(comparison))
        success_by_case: dict[str, list[float]] = defaultdict(list)
        active_by_case: dict[str, list[float]] = defaultdict(list)
        for key in keys:
            success_by_case[key[0]].append(
                float(tuned[key]["task_success"]) - float(comparison[key]["task_success"])
            )
            active_by_case[key[0]].append(
                float(tuned[key]["active_tokens"]) - float(comparison[key]["active_tokens"])
            )
        success_delta = [statistics.fmean(values) for values in success_by_case.values()]
        active_delta = [statistics.fmean(values) for values in active_by_case.values()]
        pairs[other] = {
            "paired_cases": len(success_delta),
            "controller_rows": len(keys),
            "success_delta": statistics.fmean(success_delta),
            "success_exact_sign_flip_p": _exact_sign_flip(success_delta),
            "active_token_delta": statistics.fmean(active_delta),
            "active_token_exact_sign_flip_p": _exact_sign_flip(active_delta),
        }
    result["paper7"]["paired_tuned_comparisons"] = pairs

    for dataset in sorted({row.get("dataset", "") for row in external}):
        if not dataset:
            continue
        result["headroom_eval"][dataset] = {}
        for condition in sorted({row["condition"] for row in external if row.get("dataset") == dataset}):
            values = [
                row for row in external
                if row.get("dataset") == dataset
                and row["condition"] == condition
                and row.get("status") == "supported"
                and row.get("evidence_eligible", "1") != "0"
            ]
            if values:
                result["headroom_eval"][dataset][condition] = {
                    "n": len(values),
                    "task_success": _mean(values, "task_success"),
                    "active_tokens": _mean(values, "active_tokens"),
                }
    return result


def _plots(
    paper7: list[dict[str, str]],
    external: list[dict[str, str]],
    trigger: list[dict[str, Any]],
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {
        "PRA_FROZEN": "#1f6f54",
        "HEADROOM_OFFICIAL_DEFAULT": "#2f6690",
        "HEADROOM_OFFICIAL_TUNED": "#d97706",
        "CCR_STYLE": "#777777",
        "FULL_BACKING": "#202020",
        "COMPACT_ONLY": "#9b2226",
    }
    grouped = defaultdict(list)
    for row in paper7:
        grouped[row["condition"]].append(row)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for condition, values in grouped.items():
        if condition not in colors:
            continue
        ax.scatter(
            _mean(values, "active_tokens"),
            _mean(values, "task_success"),
            s=75,
            color=colors[condition],
            label=condition.replace("_", " "),
        )
    ax.set_xlabel("Mean active or model-visible tokens")
    ax.set_ylabel("Exact evidence success")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"paper7_headroom_quality_cost.{suffix}", dpi=220)
    plt.close(fig)

    classes = ["C0_CONTINUE", "C1_FULL", "C2_MORE", "C3_CURSOR", "C4_SEARCH", "C5_TOOL"]
    conditions = ["HEADROOM_OFFICIAL_TUNED", "CCR_STYLE"]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    width = 0.36
    for offset, condition in enumerate(conditions):
        values = []
        for case_class in classes:
            row = next(
                item for item in trigger
                if item["condition"] == condition and item["case_class"] == case_class
            )
            values.append(row["trigger_accuracy"])
        x = [index + (offset - 0.5) * width for index in range(len(classes))]
        ax.bar(x, values, width=width, color=colors[condition], label=condition.replace("_", " "))
    ax.set_xticks(range(len(classes)), [value.split("_")[0] for value in classes])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Trigger/action accuracy")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"paper7_headroom_trigger_accuracy.{suffix}", dpi=220)
    plt.close(fig)

    supported = [
        row for row in external
        if row.get("status") == "supported" and row.get("evidence_eligible", "1") != "0"
    ]
    if supported:
        datasets = sorted({row["dataset"] for row in supported})
        conditions = ["PRA_FROZEN_R4", "PRA_FROZEN_R8", "HEADROOM_OFFICIAL_TUNED"]
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        width = 0.24
        for offset, condition in enumerate(conditions):
            source_condition = "PRA_FROZEN" if condition.startswith("PRA_FROZEN") else condition
            metric = "evidence_recall_at_8" if condition.endswith("R8") else "task_success"
            values = [
                _mean(
                    [
                        row for row in supported
                        if row["dataset"] == dataset and row["condition"] == source_condition
                    ],
                    metric,
                )
                for dataset in datasets
            ]
            x = [index + (offset - 1) * width for index in range(len(datasets))]
            color = {
                "PRA_FROZEN_R4": "#1f6f54",
                "PRA_FROZEN_R8": "#7aa98f",
                "HEADROOM_OFFICIAL_TUNED": "#d97706",
            }[condition]
            ax.bar(x, values, width=width, color=color, label=condition.replace("_", " "))
        ax.set_xticks(range(len(datasets)), datasets)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Exact evidence recall")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(FIGURES / f"pra_on_headroom_cross_dataset.{suffix}", dpi=220)
        plt.close(fig)


def _macros(summary: list[dict[str, Any]], trigger: list[dict[str, Any]]) -> None:
    paper = {
        row["condition"]: row
        for row in summary
        if row["workload"] == "paper7" and row["status"] == "supported"
    }
    c5 = {
        row["condition"]: row
        for row in trigger
        if row["case_class"] == "C5_TOOL"
    }
    pct = lambda value: f"{100 * float(value):.1f}"
    lines = [
        "% Generated by experiments/paper7_headroom/summarize_cross_eval.py",
        f"\\newcommand{{\\PaperSevenHeadroomDefaultSuccess}}{{{pct(paper['HEADROOM_OFFICIAL_DEFAULT']['task_success'])}\\%}}",
        f"\\newcommand{{\\PaperSevenHeadroomTunedSuccess}}{{{pct(paper['HEADROOM_OFFICIAL_TUNED']['task_success'])}\\%}}",
        f"\\newcommand{{\\PaperSevenHeadroomDefaultTokens}}{{{paper['HEADROOM_OFFICIAL_DEFAULT']['active_tokens']:.1f}}}",
        f"\\newcommand{{\\PaperSevenHeadroomTunedTokens}}{{{paper['HEADROOM_OFFICIAL_TUNED']['active_tokens']:.1f}}}",
        f"\\newcommand{{\\PaperSevenHeadroomCfiveTrigger}}{{{pct(c5['HEADROOM_OFFICIAL_TUNED']['trigger_accuracy'])}\\%}}",
    ]
    external_rows = [row for row in summary if row["workload"] == "headroom_eval"]
    for dataset in ("tool_outputs", "ccr_needle", "hotpotqa", "msmarco"):
        label = dataset.replace("_", "").title()
        for condition, short in (("PRA_FROZEN", "PRA"), ("HEADROOM_OFFICIAL_TUNED", "Headroom")):
            match = next(
                (row for row in external_rows if row["dataset"] == dataset and row["condition"] == condition),
                None,
            )
            value = pct(match["task_success"]) + "\\%" if match else "n/a"
            lines.append(f"\\newcommand{{\\PaperSevenCross{label}{short}}}{{{value}}}")
            recall_eight = pct(match["evidence_recall_at_8"]) + "\\%" if match else "n/a"
            lines.append(f"\\newcommand{{\\PaperSevenCross{label}{short}REight}}{{{recall_eight}}}")
            sample_count = str(match["n"]) if match else "0"
            lines.append(f"\\newcommand{{\\PaperSevenCross{label}{short}N}}{{{sample_count}}}")
    (OUTPUT / "generated_headroom_cross_eval_results.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    paper7 = _read(OUTPUT / "headroom_on_paper7_results.csv")
    external = _read(OUTPUT / "pra_on_headroom_results.csv")
    if not paper7:
        raise SystemExit("run_headroom_cross_eval.py first")
    trigger = _trigger(paper7)
    faithfulness = _faithfulness(paper7)
    summary = _summaries(paper7, external)
    _write(OUTPUT / "headroom_trigger_analysis.csv", trigger)
    _write(OUTPUT / "headroom_ccr_style_faithfulness.csv", faithfulness)
    _write(OUTPUT / "headroom_cross_eval_summary.csv", summary)
    _write(OUTPUT / "headroom_cost_accounting.csv", _cost(summary))
    stats = _stats(paper7, external)
    (OUTPUT / "headroom_cross_eval_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8"
    )
    _plots(paper7, external, trigger)
    _macros(summary, trigger)
    print(f"summarized {len(paper7)} Paper 7 rows and {len(external)} cross-dataset rows")


if __name__ == "__main__":
    main()
