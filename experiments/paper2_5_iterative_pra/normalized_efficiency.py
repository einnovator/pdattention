"""Normalize Paper 2.5 retrieval quality by candidate and evidence set size."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.pra_efficiency import normalized_metrics, parse_ids, summarize


ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _primary(row: dict[str, str]) -> bool:
    return row["method"] == "one_shot" or (
        row["method"] == "iterative"
        and int(row["depth"]) == 2
        and float(row["alpha"]) == 0.25
        and row["frontier"] == "direct"
        and row["path_score"] == "product"
    )


def _rows(source: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, hops = [], []
    oracle_keys: set[tuple[str, ...]] = set()
    for item in source:
        if not _primary(item):
            continue
        evidence = parse_ids(item["evidence_chunk_ids"])
        selected = parse_ids(item["selected_chunk_ids"])
        base = {
            "example_id": item["example_id"], "dataset": item["dataset"],
            "seed": int(item["seed"]), "policy": item["method"],
            "depth": int(item["depth"]), "budget_fraction": float(item["fraction"]),
        }
        row = {**base, **normalized_metrics(int(item["candidate_chunks"]), evidence, selected)}
        rows.append(row)

        cumulative: list[str] = []
        for hop, identities in enumerate(json.loads(item["selected_chunk_ids_by_hop"]), 1):
            previous = len(set(cumulative))
            cumulative.extend(identities)
            hop_row = {
                **base, "hop": hop, "incremental_K": len(set(cumulative)) - previous,
                "cumulative_K": len(set(cumulative)),
                **normalized_metrics(int(item["candidate_chunks"]), evidence, cumulative),
            }
            hops.append(hop_row)

        oracle_key = tuple(str(base[key]) for key in ("dataset", "example_id", "seed", "depth", "budget_fraction"))
        if item["method"] == "one_shot" and oracle_key not in oracle_keys:
            oracle_keys.add(oracle_key)
            budget = int(item["budget"])
            oracle_selected = evidence[: min(len(evidence), budget)]
            rows.append({**base, "policy": "oracle", **normalized_metrics(int(item["candidate_chunks"]), evidence, oracle_selected)})
    return rows, hops


def _frontiers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = summarize(rows, ("dataset", "policy", "budget_fraction"))
    output = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for policy in ("one_shot", "iterative", "oracle"):
            points = [row for row in summary if row["dataset"] == dataset and row["policy"] == policy]
            for target in (0.80, 0.90, 0.95):
                feasible = [row for row in points if row["R_E"] >= target]
                if feasible:
                    best = min(feasible, key=lambda row: row["K_over_N"])
                    output.append({"dataset": dataset, "policy": policy, "target_R_E": target,
                                   "phi": best["K_over_N"], "K_star": best["K"], "R_E": best["R_E"], "C_E": best["C_E"]})
    return output


def _plots(output: Path, summary: list[dict[str, Any]]) -> None:
    specs = (
        ("K_over_N", "R_E", "recall_vs_active_fraction", "Evidence recall", "Selected fraction K/N"),
        ("K_over_N", "C_E", "complete_recovery_vs_active_fraction", "Complete recovery", "Selected fraction K/N"),
        ("R_E", "P_E", "precision_vs_recall", "Evidence precision", "Evidence recall"),
        ("R_E", "K_over_E", "working_set_overhead_vs_recall", "Working-set overhead K/E", "Evidence recall"),
    )
    for x_name, y_name, stem, y_label, x_label in specs:
        figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        for (dataset, policy) in sorted({(r["dataset"], r["policy"]) for r in summary}):
            values = sorted((r for r in summary if r["dataset"] == dataset and r["policy"] == policy), key=lambda r: r[x_name])
            axis.plot([r[x_name] for r in values], [r[y_name] for r in values], marker="o", label=f"{dataset}: {policy}")
        axis.set(xlabel=x_label, ylabel=y_label)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
        figure.savefig(output / f"{stem}.png", dpi=180)
        figure.savefig(output / f"{stem}.pdf")
        plt.close(figure)


def run(input_path: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    rows, hops = _rows(_read(input_path))
    summary = summarize(rows, ("dataset", "policy", "budget_fraction"))
    frontiers = _frontiers(rows)
    complete = summarize(rows, ("dataset", "policy"))
    overhead = [{k: row[k] for k in ("dataset", "policy", "budget_fraction", "examples", "R_E", "K_over_N", "K_over_E", "rho", "eta")} for row in summary]
    by_e = summarize(rows, ("dataset", "policy", "E"))
    by_n = summarize(rows, ("dataset", "policy", "N"))
    _write(output / "pra_efficiency_rows.csv", rows)
    _write(output / "pra_efficiency_summary.csv", summary)
    _write(output / "pra_budget_frontiers.csv", frontiers)
    _write(output / "pra_complete_recovery.csv", complete)
    _write(output / "pra_working_set_overhead.csv", overhead)
    _write(output / "pra_efficiency_by_evidence_count.csv", by_e)
    _write(output / "pra_efficiency_by_chunk_count.csv", by_n)
    _write(output / "pra_efficiency_per_hop.csv", hops)
    _plots(output, summary)
    findings = {
        "schema_version": "1.0", "rows": len(rows), "per_hop_rows": len(hops),
        "definitions": {"R_E": "recovered/E", "P_E": "recovered/K", "C_E": "all evidence recovered", "phi": "K/N", "omega": "K/E", "rho": "1-K/N", "eta": "R_E/phi"},
        "macro_summary": summary, "reachable_recall_frontiers": frontiers,
        "scope": "Frozen closure replay; oracle is a matched-budget evidence-aware upper bound.",
    }
    (output / "pra_efficiency_findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    canonical_path = output.parent / "final_metrics" / "final_metrics_results.json"
    if canonical_path.exists():
        canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
        canonical["normalized_efficiency"] = findings
        canonical_path.write_text(json.dumps(canonical, indent=2), encoding="utf-8")
    (output / "pra_efficiency_claim_audit.md").write_text(
        "# PRA efficiency claim audit\n\n"
        "- K is the actual deduplicated selected set, not the requested budget.\n"
        "- E is counted from evaluator-side evidence identities after routing.\n"
        "- The oracle is a matched-budget recovery ceiling and not a deployable policy.\n"
        "- Discovery efficiency is not a physical K/V latency or materialization claim.\n",
        encoding="utf-8",
    )
    return findings


def parse_args() -> argparse.Namespace:
    root = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=root / "iterative_closure_rows.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "normalized_efficiency")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.input, args.output_dir), indent=2))
