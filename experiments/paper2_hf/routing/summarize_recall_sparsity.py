"""Reconstruct Paper 2 recall-sparsity curves from stored ranking artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from common import DEFAULT_FRACTIONS, recall_sparsity_curve
from pra_torch.hf import load_hf_routing_projection

ROUTING_DIR = ROOT / "docs" / "papers" / "shared" / "results" / "paper2_hf" / "routing"
OUTPUT_DIR = ROOT / "docs" / "papers" / "shared" / "results" / "recall_sparsity" / "paper2_hf"
LEARNED_DIR = ROUTING_DIR / "learned_adapter"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _legacy_curve(rows: list[dict]) -> dict:
    """Recover the historical any-evidence convention from candidate count and best rank."""
    rankings = []
    evidence = []
    for row in rows:
        size = int(row["candidate_chunks"])
        rankings.append(list(range(size)))
        rank = row.get("best_evidence_rank")
        evidence.append({int(rank) - 1} if rank is not None else {size})
    result = recall_sparsity_curve(rankings, evidence, require_complete_endpoint=False)
    result["evidence_semantics"] = "any annotated evidence chunk; reconstructed from stored best rank"
    return result


def _deduplicate(rows: list[dict]) -> list[dict]:
    """Remove repeated top-k evaluations that share one complete ranking."""
    output = {}
    for row in rows:
        key = (row.get("seed"), row["dataset"], row["example_id"])
        output.setdefault(key, row)
    return list(output.values())


def _mechanisms() -> list[dict]:
    representation = _load(ROUTING_DIR / "qwen_routing_representation.json")["rows"]
    centered = _load(ROUTING_DIR / "centered_rope" / "centered_rope_segment_mean.json")["rows"]
    queries = _load(ROUTING_DIR / "query_strategies" / "query_strategy_confirmation.json")["rows"]
    segment = _load(ROUTING_DIR / "qwen_routing_segment_mean_confirmation.json")["rows"]
    token = _load(ROUTING_DIR / "qwen_routing_hidden_token_max.json")["rows"]
    margin = _load(LEARNED_DIR / "margin_objective_results.json")

    definitions = []
    for value, label in (
        ("post_rope_key", "Post-RoPE K mean"),
        ("pre_rope_key", "Pre-RoPE K mean"),
        ("hidden_state", "Hidden-state mean"),
    ):
        definitions.append(
            {
                "mechanism": label,
                "cohort": "representation-stage1-16",
                "rows": _deduplicate(
                    [row for row in representation if row["routing_representation"] == value]
                ),
                "major": True,
            }
        )
    for count in (1, 2, 4, 8):
        definitions.append(
            {
                "mechanism": f"Centered-RoPE segment mean G={count}",
                "cohort": "centered-rope-stage1-16",
                "rows": _deduplicate(
                    [
                        row
                        for row in centered
                        if row["routing_representation"] == "centered_rope_key"
                        and int(row["gist_count"]) == count
                    ]
                ),
                "major": count == 8,
            }
        )
    for count in (2, 4, 8):
        definitions.append(
            {
                "mechanism": f"Hidden-state segment mean G={count}",
                "cohort": "segment-confirmation-64",
                "rows": _deduplicate([row for row in segment if int(row["gist_count"]) == count]),
                "major": False,
            }
        )
    definitions.append(
        {
            "mechanism": "Hidden-state token maximum",
            "cohort": "token-diagnostic-16",
            "rows": _deduplicate([row for row in token if int(row["gist_count"]) == 32]),
            "major": False,
        }
    )
    for strategy, label in (
        ("last", "Last-state query"),
        ("uniform_w32", "Uniform query W=32"),
        ("question_exp_h2.0", "Question decay H=2"),
    ):
        definitions.append(
            {
                "mechanism": label,
                "cohort": "query-confirmation-32",
                "rows": _deduplicate([row for row in queries if row["query_strategy"] == strategy]),
                "major": strategy == "last",
            }
        )
    selected_runs = [
        run
        for run in margin["runs"]
        if run["architecture"] == "asymmetric_linear"
        and int(run["routing_width"]) == 128
        and run["objective"] == "margin"
    ]
    definitions.append(
        {
            "mechanism": "Learned margin adapter",
            "cohort": "learned-test-32x5-seeds",
            "rows": [row for run in selected_runs for row in run["test"]["rows"]],
            "major": True,
        }
    )
    return definitions


def _exact_frozen_curves() -> dict[str, dict]:
    features = torch.load(LEARNED_DIR / "router_features_test.pt", weights_only=False)
    projection = load_hf_routing_projection(
        LEARNED_DIR
        / "checkpoints"
        / "asymmetric_linear_d128_last_joint_seed53_margin_exhaustive.pt"
    )
    output = {}
    for name, learned in (("Last-state cosine", False), ("Learned margin seed 53", True)):
        rankings = []
        evidence = []
        lengths = []
        for feature in features:
            query = feature["queries"]["last"].float().unsqueeze(0)
            memory = feature["memory_gists"].float()
            scores = (
                projection.scores(query, memory)[0]
                if learned
                else (F.normalize(query, dim=-1) @ F.normalize(memory, dim=-1).T)[0]
            )
            order = torch.argsort(scores, descending=True).tolist()
            rankings.append(order)
            evidence.append(
                set(torch.nonzero(feature["positive_mask"], as_tuple=False).flatten().tolist())
            )
            token_lengths = [int(end) - int(start) for start, end in feature["chunk_spans"]]
            lengths.append([token_lengths[index] for index in order])
        output[name] = recall_sparsity_curve(
            rankings,
            evidence,
            candidate_token_lengths=lengths,
            require_complete_endpoint=True,
        )
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary_row(mechanism: dict, dataset: str, result: dict) -> dict:
    curve = {float(row["fraction"]): row for row in result["curve"]}
    return {
        "mechanism": mechanism["mechanism"],
        "dataset": dataset,
        "cohort": mechanism["cohort"],
        "examples": result["examples"],
        "r_at_5pct": curve[0.05]["recall"],
        "r_at_10pct": curve[0.10]["recall"],
        "r_at_20pct": curve[0.20]["recall"],
        "r_at_30pct": curve[0.30]["recall"],
        "f70": result["inverse"]["f70"],
        "f80": result["inverse"]["f80"],
        "f90": result["inverse"]["f90"],
        "f95": result["inverse"]["f95"],
        "auc_0_30": result["auc_0_30"],
        "recall_at_1": result["fixed_k"]["1"]["recall"],
        "recall_at_3": result["fixed_k"]["3"]["recall"],
        "recall_at_8": result["fixed_k"]["8"]["recall"],
        "recall_at_16": result["fixed_k"]["16"]["recall"],
        "endpoint_complete": result["endpoint_complete"],
        "kv_fraction_exact": result["kv_fraction_exact"],
    }


def run() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mechanisms = _mechanisms()
    summaries = []
    curves = {dataset: [] for dataset in ("hotpotqa", "qasper", "combined")}
    results = {}
    for mechanism in mechanisms:
        results[mechanism["mechanism"]] = {}
        for dataset in ("hotpotqa", "qasper", "combined"):
            rows = (
                mechanism["rows"]
                if dataset == "combined"
                else [row for row in mechanism["rows"] if row["dataset"] == dataset]
            )
            result = _legacy_curve(rows)
            results[mechanism["mechanism"]][dataset] = result
            summaries.append(_summary_row(mechanism, dataset, result))
            for row in result["curve"]:
                curves[dataset].append(
                    {
                        "mechanism": mechanism["mechanism"],
                        "cohort": mechanism["cohort"],
                        **row,
                    }
                )
    _write_csv(OUTPUT_DIR / "mechanism_summary.csv", summaries)
    for dataset, rows in curves.items():
        _write_csv(OUTPUT_DIR / f"{dataset}_curves.csv", rows)
    _write_csv(
        OUTPUT_DIR / "inverse_metrics.csv",
        [
            {
                key: row[key]
                for key in ("mechanism", "dataset", "cohort", "examples", "f70", "f80", "f90", "f95", "auc_0_30")
            }
            for row in summaries
        ],
    )

    exact = _exact_frozen_curves()
    exact_rows = []
    for mechanism, result in exact.items():
        for row in result["curve"]:
            exact_rows.append({"mechanism": mechanism, **row})
    _write_csv(OUTPUT_DIR / "exact_kv_curves.csv", exact_rows)

    major = {item["mechanism"] for item in mechanisms if item["major"]}
    colors = ("#a5a5a5", "#70ad47", "#ed7d31", "#8064a2", "#4472c4", "#c0504d")
    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    for color, name in zip(colors, sorted(major)):
        selected = [row for row in curves["combined"] if row["mechanism"] == name and row["fraction"] <= 0.30]
        axis.plot(
            [100 * row["selected_chunk_fraction"] for row in selected],
            [row["recall"] for row in selected],
            marker="o",
            label=name,
            color=color,
        )
    axis.set_xlabel("Selected parent chunks (%)")
    axis.set_ylabel("Any annotated evidence recall")
    axis.set_xlim(0, 32)
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(OUTPUT_DIR / f"major_sparse_curve.{suffix}", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    for color, name in zip(colors, sorted(major)):
        selected = [row for row in curves["combined"] if row["mechanism"] == name]
        axis.plot(
            [100 * row["selected_chunk_fraction"] for row in selected],
            [row["recall"] for row in selected],
            marker="o",
            label=name,
            color=color,
        )
    axis.set_xlabel("Selected parent chunks (%)")
    axis.set_ylabel("Any annotated evidence recall")
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(OUTPUT_DIR / f"major_full_curve.{suffix}", dpi=180)
    plt.close(figure)

    best = "Learned margin adapter"
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for dataset, color in (("hotpotqa", "#70ad47"), ("qasper", "#ed7d31")):
        selected = [row for row in curves[dataset] if row["mechanism"] == best and row["fraction"] <= 0.30]
        axis.plot(
            [100 * row["selected_chunk_fraction"] for row in selected],
            [row["recall"] for row in selected],
            marker="o",
            label=dataset,
            color=color,
        )
    axis.set_xlabel("Selected parent chunks (%)")
    axis.set_ylabel("Any annotated evidence recall")
    axis.set_xlim(0, 32)
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(OUTPUT_DIR / f"learned_dataset_sparse_curve.{suffix}", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for name, color in (("Last-state cosine", "#a5a5a5"), ("Learned margin seed 53", "#4472c4")):
        selected = [row for row in exact[name]["curve"] if row["fraction"] <= 0.30]
        axis.plot(
            [100 * row["selected_kv_token_fraction"] for row in selected],
            [row["recall"] for row in selected],
            marker="o",
            label=name,
            color=color,
        )
    axis.set_xlabel("Materialized native-KV tokens (%)")
    axis.set_ylabel("Annotated evidence coverage")
    axis.set_xlim(0, 32)
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(OUTPUT_DIR / f"exact_kv_sparse_curve.{suffix}", dpi=180)
    plt.close(figure)

    report = {
        "fractions": list(DEFAULT_FRACTIONS),
        "historical_semantics": "Any-evidence recall reconstructed from stored best evidence rank.",
        "evidence_mapping": "Any token-span overlap with a 32-token parent chunk, unchanged from the original runners.",
        "endpoint_failures": [
            {"mechanism": mechanism, "dataset": dataset}
            for mechanism, values in results.items()
            for dataset, result in values.items()
            if not result["endpoint_complete"]
        ],
        "exact_kv_mechanisms": list(exact),
        "cohort_warning": "Representation/centered curves use 16 examples, query curves 32, segment confirmation 64 evaluations, and learned curves 32 examples x 5 seeds; cross-mechanism differences are not all paired.",
        "results": results,
    }
    (OUTPUT_DIR / "recall_sparsity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps({"mechanisms": len(report["results"]), "endpoint_failures": report["endpoint_failures"]}, indent=2))
