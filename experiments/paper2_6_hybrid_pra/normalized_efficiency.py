"""Build normalized root/successor efficiency artifacts for Paper 2.6."""

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
ROOT_CHANNELS = ("gist", "exact", "bm25", "approx", "hybrid")
SUCCESSOR_CHANNELS = ("native_semantic", "exact_new_address", "bm25_state", "approx_new_address", "hybrid_state")


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
        writer.writeheader(); writer.writerows(rows)


def _identity(row: dict[str, str]) -> tuple[str, str, str]:
    return row["split"], row["dataset"], row["example_id"]


def build_rows(root_rows: list[dict[str, str]], successor_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    roots = {( *_identity(row), row["channel"]): row for row in root_rows if row["channel"] in ROOT_CHANNELS}
    candidate_counts = {_identity(row): int(row["comparisons"]) for row in root_rows if row["channel"] == "gist"}
    output, increments = [], []
    for key, item in roots.items():
        identity, channel = key[:3], key[3]
        selected, evidence = parse_ids(item["selected_chunk_ids"]), parse_ids(item["gold_chunk_ids"])
        output.append({"stage": "root", "split": identity[0], "dataset": identity[1], "example_id": identity[2],
                       "policy": channel, "root_method": channel, "successor_method": "", "root_channel": channel, "successor_channel": "",
                       "K_root": len(set(selected)), "K_total": len(set(selected)),
                       "search_cost": int(item["comparisons"]), "search_comparisons": int(item["comparisons"]), "token_span_operations": int(item["token_span_operations"]),
                       **normalized_metrics(candidate_counts[identity], evidence, selected)})
    for item in successor_rows:
        root_channel, successor_channel = item["root_channel"], item["successor_channel"]
        if root_channel not in ROOT_CHANNELS or successor_channel not in SUCCESSOR_CHANNELS:
            continue
        identity = _identity(item)
        root = roots[(*identity, root_channel)]
        root_selected = parse_ids(item["selected_root_ids"])
        successor_selected = parse_ids(item["selected_chunk_ids"])
        root_evidence = parse_ids(root["gold_chunk_ids"])
        successor_evidence = parse_ids(item["gold_chunk_ids"])
        selected = list(dict.fromkeys([*root_selected, *successor_selected]))
        evidence = list(dict.fromkeys([*root_evidence, *successor_evidence]))
        root_metrics = normalized_metrics(candidate_counts[identity], evidence, root_selected)
        full_metrics = normalized_metrics(candidate_counts[identity], evidence, selected)
        delta_phi = full_metrics["K_over_N"] - root_metrics["K_over_N"]
        delta_recall = full_metrics["R_E"] - root_metrics["R_E"]
        policy = f"{root_channel}->{successor_channel}"
        output.append({"stage": "root_successor", "split": identity[0], "dataset": identity[1], "example_id": identity[2],
                       "policy": policy, "root_method": root_channel, "successor_method": successor_channel, "root_channel": root_channel, "successor_channel": successor_channel,
                       "K_root": len(set(root_selected)), "K_total": len(set(selected)),
                       "root_R_E": root_metrics["R_E"], "root_P_E": root_metrics["P_E"], "root_C_E": root_metrics["C_E"],
                       "root_K_over_N": root_metrics["K_over_N"],
                       "search_cost": int(root["comparisons"]) + int(item["comparisons"]),
                       "search_comparisons": int(root["comparisons"]) + int(item["comparisons"]),
                       "token_span_operations": int(root["token_span_operations"]) + int(item["token_span_operations"]),
                       **full_metrics})
        increments.append({"split": identity[0], "dataset": identity[1], "example_id": identity[2], "root_channel": root_channel,
                           "successor_channel": successor_channel, "K_root": len(set(root_selected)), "K_total": len(set(selected)),
                           "incremental_K": len(set(selected)) - len(set(root_selected)), "root_R_E": root_metrics["R_E"],
                           "final_R_E": full_metrics["R_E"], "delta_R_E": delta_recall, "delta_K_over_N": delta_phi,
                           "delta_R_E_per_delta_K_over_N": delta_recall / delta_phi if delta_phi else 0.0})
    return output, increments


def _pareto(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for dataset in sorted({row["dataset"] for row in summary}):
        rows = [row for row in summary if row["dataset"] == dataset]
        for row in rows:
            dominated = any(other["R_E"] >= row["R_E"] and other["K_over_N"] <= row["K_over_N"] and
                            (other["R_E"] > row["R_E"] or other["K_over_N"] < row["K_over_N"]) for other in rows)
            output.append({**row, "pareto": int(not dominated)})
    return output


def _plots(output: Path, summary: list[dict[str, Any]], increments: list[dict[str, Any]], postmortem: list[dict[str, str]]) -> None:
    specs = (("K_over_N", "R_E", "normalized_channel_recall_vs_active_fraction"), ("K_over_N", "C_E", "normalized_channel_complete_recovery"),
             ("R_E", "P_E", "normalized_channel_precision_recall"), ("R_E", "K_over_E", "normalized_channel_working_set_overhead"))
    for x, y, stem in specs:
        fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        for policy in sorted({r["policy"] for r in summary}):
            values = [r for r in summary if r["policy"] == policy]
            ax.scatter([r[x] for r in values], [r[y] for r in values], s=28, label=policy)
        ax.set(xlabel=x.replace("_", " "), ylabel=y.replace("_", " ")); ax.grid(alpha=.25)
        ax.legend(fontsize=6, ncol=3)
        fig.savefig(output / f"{stem}.png", dpi=180); fig.savefig(output / f"{stem}.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    ax.scatter([r["delta_K_over_N"] for r in increments], [r["delta_R_E"] for r in increments], alpha=.3, s=14)
    ax.axhline(0, color="black", lw=.8); ax.set(xlabel="Incremental selected fraction", ylabel="Incremental evidence recall"); ax.grid(alpha=.25)
    fig.savefig(output / "root_successor_incremental_efficiency.png", dpi=180); fig.savefig(output / "root_successor_incremental_efficiency.pdf"); plt.close(fig)
    pairs = [row for row in summary if row["stage"] == "root_successor"]
    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    ax.scatter([r["K_over_N"] for r in pairs], [r["R_E"] for r in pairs], c=[r["search_cost"] for r in pairs], cmap="viridis", s=42)
    ax.set(xlabel="Selected fraction K/N", ylabel="Evidence recall"); ax.grid(alpha=.25)
    fig.colorbar(ax.collections[0], ax=ax, label="Search comparisons")
    fig.savefig(output / "search_cost_vs_selected_fraction.png", dpi=180); fig.savefig(output / "search_cost_vs_selected_fraction.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    held = [r for r in postmortem if r["split"] == "test"]
    ax.scatter([float(r["unique_distractors_from_lexical"]) for r in held], [float(r["unique_gold_from_lexical"]) for r in held], alpha=.4, s=18)
    ax.set(xlabel="Lexical-only distractors", ylabel="Lexical-only evidence"); ax.grid(alpha=.25)
    fig.savefig(output / "static_fusion_incremental_evidence_distractors.png", dpi=180); fig.savefig(output / "static_fusion_incremental_evidence_distractors.pdf"); plt.close(fig)
    pareto = [r for r in _pareto(pairs) if r["pareto"]]
    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    for dataset in sorted({r["dataset"] for r in pareto}):
        vals = [r for r in pareto if r["dataset"] == dataset]
        ax.scatter([r["K_over_N"] for r in vals], [r["R_E"] for r in vals], s=48, label=dataset)
    ax.set(xlabel="Selected fraction K/N", ylabel="Evidence recall"); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(output / "channel_oracle_pareto.png", dpi=180); fig.savefig(output / "channel_oracle_pareto.pdf"); plt.close(fig)


def run(input_dir: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    rows, increments = build_rows(_read(input_dir / "root_channel_results.csv"), _read(input_dir / "successor_channel_results.csv"))
    heldout = [row for row in rows if row["split"] == "test"]
    summary = summarize(heldout, ("stage", "dataset", "policy", "root_channel", "successor_channel"))
    for record in summary:
        values = [row for row in heldout if all(row[key] == record[key] for key in ("stage", "dataset", "policy", "root_channel", "successor_channel"))]
        for field in ("K_root", "K_total", "search_cost", "search_comparisons", "token_span_operations"):
            record[field] = statistics.fmean(float(row[field]) for row in values)
    pairs = [row for row in summary if row["stage"] == "root_successor"]
    frontier = _pareto(pairs)
    complete = [{key: row[key] for key in ("dataset", "policy", "examples", "C_E", "R_E", "K_over_N")} for row in pairs]
    by_e = summarize([row for row in heldout if row["stage"] == "root_successor"], ("dataset", "policy", "E"))
    _write(output / "pra_search_efficiency_rows.csv", rows)
    _write(output / "pra_search_efficiency_summary.csv", summary)
    _write(output / "root_successor_efficiency.csv", increments)
    _write(output / "channel_efficiency_frontiers.csv", frontier)
    _write(output / "channel_complete_recovery.csv", complete)
    _write(output / "channel_oracle_pareto.csv", [row for row in frontier if row["pareto"]])
    _write(output / "channel_efficiency_by_E.csv", by_e)
    _plots(output, summary, [row for row in increments if row["split"] == "test"], _read(input_dir / "static_hybrid_postmortem.csv"))
    findings = {"schema_version": "1.0", "rows": len(rows), "heldout_rows": len(heldout), "root_successor_pairs": 25,
                "definitions": {"R_E": "recovered/E", "P_E": "recovered/K_total", "C_E": "complete evidence recovery", "phi": "K_total/N", "omega": "K_total/E"},
                "summary": summary, "materialization_performed": False,
                "scope": "Frozen channel replay; search cost and selected identity fraction are reported separately."}
    (output / "paper2_6_efficiency_findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    canonical_path = input_dir.parent / "channel_geometry" / "paper2_6_findings.json"
    if canonical_path.exists():
        canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
        canonical["normalized_efficiency"] = findings
        canonical_path.write_text(json.dumps(canonical, indent=2), encoding="utf-8")
    (output / "pra_efficiency_claim_audit.md").write_text(
        "# Normalized search-efficiency claim audit\n\n- K_total is deduplicated across root and successor selections.\n"
        "- N is the per-example candidate chunk count.\n- Evidence identities are evaluator-only labels.\n"
        "- Search comparisons are not K/V materialization or generation cost.\n", encoding="utf-8")
    metric_contract = {"candidate_count": "N", "candidate_chunks": "N", "evidence_count": "E",
        "root_unique_selected": "K_root", "selected_unique_chunks": "K_total",
        "cumulative_unique_selected": "K_total", "evidence_recall": "R_E", "evidence_precision": "P_E",
        "complete_recovery": "C_E", "selected_fraction": "K_total/N", "working_set_overhead": "K_total/E",
        "search_cost": "search_cost", "search_cost_separate": ["comparisons", "index_lookups", "token_span_operations", "latency_ms"]}
    for spec_path in (input_dir / "search_method_action_spec.json", input_dir.parent / "final_iteration" / "search_method_action_spec.json"):
        if spec_path.exists():
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            spec["normalized_efficiency_metrics"] = metric_contract
            spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
    return findings


def parse_args() -> argparse.Namespace:
    base = ROOT / "docs/papers/shared/results/paper2_6_hybrid_pra/channel_selection"
    parser = argparse.ArgumentParser(); parser.add_argument("--input-dir", type=Path, default=base)
    parser.add_argument("--output-dir", type=Path, default=base / "normalized_efficiency")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args(); print(json.dumps(run(args.input_dir, args.output_dir), indent=2))
