"""Normalize Paper 3.5 search, admission, and retry working sets."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.pra_efficiency import normalized_metrics, parse_ids, summarize


ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream: return list(csv.DictReader(stream))


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: path.write_text("", encoding="utf-8"); return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _surface_row(item: dict[str, str], policy: str, *, prefix: str = "") -> dict[str, Any]:
    get = lambda name: item[f"{prefix}{name}"]
    evidence = parse_ids(get("evidence_parent_ids"))
    visited, admitted = parse_ids(get("visited_ids")), parse_ids(get("materialized_ids"))
    search = normalized_metrics(int(float(get("parent_count"))), evidence, visited)
    admit = normalized_metrics(int(float(get("parent_count"))), evidence, admitted)
    return {"partition": item.get("partition", "test"), "dataset": item["dataset"], "example_id": item["example_id"],
            "model_seed": int(item.get("seed", item.get("model_seed", 0))), "policy": policy,
            "N": search["N"], "E": search["E"], "K": search["K"], "recovered": search["recovered"],
            "R_E": search["R_E"], "P_E": search["P_E"], "C_E": search["C_E"],
            "K_over_N": search["K_over_N"], "K_over_E": search["K_over_E"],
            "rho": search["rho"], "eta": search["eta"],
            "K_search": search["K"], "K_admit": admit["K"],
            "KV_active": float(get("materialized_kv_tokens")),
            "search_cost": float(get("root_comparisons")) + float(get("transition_comparisons")),
            "output_quality": item.get(f"{prefix}output_quality", ""),
            "search_R_E": search["R_E"], "search_P_E": search["P_E"], "search_C_E": search["C_E"],
            "search_K_over_N": search["K_over_N"], "search_K_over_E": search["K_over_E"],
            "admit_recovered": admit["recovered"], "admit_R_E": admit["R_E"],
            "admit_P_E": admit["P_E"], "admit_C_E": admit["C_E"],
            "admit_K_over_N": admit["K_over_N"], "admit_K_over_E": admit["K_over_E"]}


def _controller_rows(base: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    for name, filename in (("fixed_E1", "fixed_policy_rows.csv"), ("factorized_oracle", "factorized_oracle_rows.csv")):
        rows.extend(_surface_row(item, name) for item in _read(base / filename) if item["partition"] == "test")
    for item in _read(base / "router_under_over_allocation.csv"):
        rows.append(_surface_row(item, item["variant"], prefix="selected_"))
    retry_rows = []
    for item in _read(base / "targeted_retry_results.csv"):
        initial = _surface_row(item, "retry_initial", prefix="initial_")
        final_item = dict(item)
        for name in ("parent_count", "evidence_parent_count", "evidence_parent_ids"):
            final_item[f"final_{name}"] = item[f"initial_{name}"]
        final = _surface_row(final_item, "targeted_retry", prefix="final_")
        rows.append(final)
        retry_rows.append({"dataset": item["dataset"], "example_id": item["example_id"], "model_seed": int(item["model_seed"]),
                           "retry_action": item["retry_action"], "initial_K_search": initial["K_search"], "final_K_search": final["K_search"],
                           "incremental_K_search": final["K_search"] - initial["K_search"], "initial_K_admit": initial["K_admit"],
                           "final_K_admit": final["K_admit"], "incremental_K_admit": final["K_admit"] - initial["K_admit"],
                           "initial_R_E": initial["R_E"], "final_R_E": final["R_E"], "delta_R_E": final["R_E"] - initial["R_E"],
                           "initial_K_over_N": initial["K_over_N"], "final_K_over_N": final["K_over_N"],
                           "incremental_K_over_N": final["K_over_N"] - initial["K_over_N"]})
    return rows, retry_rows


def _pareto(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for dataset in sorted({r["dataset"] for r in summary}):
        values = [r for r in summary if r["dataset"] == dataset]
        for row in values:
            dominated = any(other["R_E"] >= row["R_E"] and other["K_over_N"] <= row["K_over_N"] and
                            (other["R_E"] > row["R_E"] or other["K_over_N"] < row["K_over_N"]) for other in values)
            output.append({**row, "pareto": int(not dominated)})
    return output


def _plots(output: Path, summary: list[dict[str, Any]], rows: list[dict[str, Any]], retry: list[dict[str, Any]]) -> None:
    specs = (("admit_K_over_N", "admit_R_E", "controller_recall_vs_admission_fraction"), ("search_K_over_N", "search_R_E", "controller_recall_vs_search_fraction"),
             ("R_E", "P_E", "controller_precision_recall"), ("R_E", "K_over_E", "controller_working_set_overhead"))
    for x, y, stem in specs:
        fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
        for policy in sorted({r["policy"] for r in summary}):
            vals = [r for r in summary if r["policy"] == policy]
            ax.scatter([r[x] for r in vals], [r[y] for r in vals], label=policy, s=34)
        ax.set(xlabel=x.replace("_", " "), ylabel=y.replace("_", " ")); ax.grid(alpha=.25); ax.legend(fontsize=6, ncol=2)
        fig.savefig(output / f"{stem}.png", dpi=180); fig.savefig(output / f"{stem}.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.6, 4.3), constrained_layout=True)
    ax.scatter([r["search_K_over_N"] for r in rows], [r["admit_K_over_N"] for r in rows], alpha=.12, s=10)
    ax.plot([0, 1], [0, 1], color="black", lw=.8); ax.set(xlabel="Search fraction", ylabel="Admission fraction"); ax.grid(alpha=.25)
    fig.savefig(output / "search_more_admit_less.png", dpi=180); fig.savefig(output / "search_more_admit_less.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.6, 4.3), constrained_layout=True)
    ax.scatter([r["incremental_K_admit"] for r in retry], [r["delta_R_E"] for r in retry], alpha=.35, s=18)
    ax.set(xlabel="Retry incremental admitted parents", ylabel="Retry evidence-recall gain"); ax.grid(alpha=.25)
    fig.savefig(output / "retry_incremental_efficiency.png", dpi=180); fig.savefig(output / "retry_incremental_efficiency.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.6, 4.3), constrained_layout=True)
    for policy in sorted({r["policy"] for r in summary}):
        vals = [r for r in summary if r["policy"] == policy]
        ax.scatter([r["K_over_N"] for r in vals], [r["C_E"] for r in vals], s=34, label=policy)
    ax.set(xlabel="Admission fraction K/N", ylabel="Complete evidence recovery"); ax.grid(alpha=.25); ax.legend(fontsize=6, ncol=2)
    fig.savefig(output / "controller_complete_recovery_vs_selected_fraction.png", dpi=180); fig.savefig(output / "controller_complete_recovery_vs_selected_fraction.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.6, 3.6), constrained_layout=True)
    ax.axis("off"); ax.text(.5, .55, "Output quality was not measured in the\nfrozen factorized-control replay.", ha="center", va="center", fontsize=12)
    ax.text(.5, .3, "No generation value is imputed from retrieval recovery.", ha="center", va="center", fontsize=9)
    fig.savefig(output / "output_quality_vs_selected_fraction.png", dpi=180); fig.savefig(output / "output_quality_vs_selected_fraction.pdf"); plt.close(fig)
    pareto = [row for row in _pareto(summary) if row["pareto"]]
    fig, ax = plt.subplots(figsize=(6.6, 4.3), constrained_layout=True)
    for policy in sorted({r["policy"] for r in pareto}):
        vals = [r for r in pareto if r["policy"] == policy]
        ax.scatter([r["K_over_N"] for r in vals], [r["R_E"] for r in vals], s=45, label=policy)
    ax.set(xlabel="Admission fraction K/N", ylabel="Evidence recall"); ax.grid(alpha=.25); ax.legend(fontsize=6, ncol=2)
    fig.savefig(output / "controller_oracle_learned_pareto.png", dpi=180); fig.savefig(output / "controller_oracle_learned_pareto.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.6, 4.3), constrained_layout=True)
    misses = [r for r in rows if r.get("allocation_outcome_physical") == "under_allocation"]
    ax.hist([r["K_over_N"] for r in misses], bins=12, color="#c0392b", alpha=.75)
    ax.set(xlabel="Admission fraction K/N", ylabel="Under-allocated rows"); ax.grid(alpha=.25)
    fig.savefig(output / "under_allocation_vs_selected_fraction.png", dpi=180); fig.savefig(output / "under_allocation_vs_selected_fraction.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    for dataset in sorted({r["dataset"] for r in summary}):
        vals = [r for r in summary if r["dataset"] == dataset]
        ax.scatter([r["K_over_N"] for r in vals], [r["R_E"] for r in vals], s=42, label=dataset)
    ax.set(xlabel="Admission fraction K/N", ylabel="Evidence recall"); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(output / "dataset_efficiency_frontiers.png", dpi=180); fig.savefig(output / "dataset_efficiency_frontiers.pdf"); plt.close(fig)


def run(base: Path) -> dict[str, Any]:
    rows, retry = _controller_rows(base)
    oracle = {(r["dataset"], r["example_id"], r["model_seed"]): r for r in rows if r["policy"] == "factorized_oracle"}
    for row in rows:
        target = oracle[(row["dataset"], row["example_id"], row["model_seed"])]
        if row["C_E"] < target["C_E"] and row["K_over_N"] < target["K_over_N"]:
            outcome = "under_allocation"
        elif row["C_E"] >= target["C_E"] and row["K_over_N"] > target["K_over_N"]:
            outcome = "over_allocation"
        elif row["C_E"] >= target["C_E"]:
            outcome = "efficient"
        else:
            outcome = "action_space_failure"
        row["allocation_outcome_physical"] = outcome
    summary = summarize(rows, ("dataset", "policy"))
    for record in summary:
        values = [row for row in rows if row["dataset"] == record["dataset"] and row["policy"] == record["policy"]]
        for field in ("K_search", "K_admit", "KV_active", "search_cost", "search_R_E", "search_P_E",
                      "search_C_E", "search_K_over_N", "search_K_over_E", "admit_recovered",
                      "admit_R_E", "admit_P_E", "admit_C_E", "admit_K_over_N", "admit_K_over_E"):
            record[field] = statistics.fmean(float(row[field]) for row in values)
        record["median_N"] = statistics.median(float(row["N"]) for row in values)
        record["median_E"] = statistics.median(float(row["E"]) for row in values)
    pareto = _pareto(summary)
    complete = [{k: r[k] for k in ("dataset", "policy", "examples", "C_E", "R_E", "K_over_N")} for r in summary]
    by_e = summarize(rows, ("dataset", "policy", "E"))
    allocation = [{"dataset": r["dataset"], "example_id": r["example_id"], "policy": r["policy"],
                   "allocation_outcome_physical": r["allocation_outcome_physical"], "under_allocated": int(r["allocation_outcome_physical"] == "under_allocation"), "overhead_K_over_E": r["admit_K_over_E"],
                   "KV_active": r["KV_active"]} for r in rows]
    search_admit = [{k: r[k] for k in ("dataset", "example_id", "policy", "N", "E", "K_search", "K_admit",
                                                  "search_K_over_N", "admit_K_over_N", "search_R_E", "admit_R_E",
                                                  "KV_active")} for r in rows]
    _write(base / "controller_efficiency_rows.csv", rows); _write(base / "controller_efficiency_summary.csv", summary)
    _write(base / "controller_pareto_frontier.csv", pareto); _write(base / "controller_complete_recovery.csv", complete)
    _write(base / "controller_retry_efficiency.csv", retry); _write(base / "controller_under_over_allocation_physical.csv", allocation)
    _write(base / "search_admission_efficiency.csv", search_admit); _write(base / "controller_efficiency_by_E.csv", by_e)
    _plots(base, summary, rows, retry)
    findings = {"schema_version": "1.0", "rows": len(rows), "definitions": {"N": "candidate parents", "E": "evidence parents",
                "K_search": "unique visited parents", "K_admit": "unique materialized parents", "KV_active": "materialized native-K/V tokens",
                "R_E": "searched evidence-parent recall", "phi_search": "K_search/N", "phi_admit": "K_admit/N",
                "admit_R_E": "admitted evidence-parent recall"},
                "summary": summary, "scope": "Frozen native-score replay; output generation is not imputed."}
    (base / "controller_efficiency_findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    canonical_path = base / "paper3_5_next_findings.json"
    if canonical_path.exists():
        canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
        canonical["normalized_efficiency"] = findings
        canonical_path.write_text(json.dumps(canonical, indent=2), encoding="utf-8")
    (base / "controller_efficiency_claim_audit.md").write_text(
        "# Controller efficiency claim audit\n\n- Search, admission, and physical native-K/V tokens remain separate.\n"
        "- Controller and retry rows use the exact selected factorized configuration.\n"
        "- Evidence identities are evaluator-only and never controller inputs.\n- Generation quality is not inferred from retrieval recovery.\n", encoding="utf-8")
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path,
        default=ROOT / "docs/papers/shared/results/paper3_5_adaptive_pra"); return parser.parse_args()


if __name__ == "__main__":
    args = parse_args(); print(json.dumps(run(args.output_dir), indent=2))
