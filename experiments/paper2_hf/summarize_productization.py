"""Create Paper 2 cross-family productization tables and plots."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs" / "papers" / "shared" / "results" / "paper2_hf" / "productization"


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _curve(run: dict, dataset: str) -> dict[float, float]:
    return {
        float(row["fraction"]): float(row["recall"])
        for row in run["test"][dataset]["curve"]
    }


def run() -> dict:
    routers = {
        "Qwen3-0.6B": _load("qwen_router_five_seed.json"),
        "SmolLM2-135M (Llama)": _load("llama_router_five_seed.json"),
    }
    demos = {
        "Qwen3-0.6B": _load("qwen_product_demo.json"),
        "SmolLM2-135M (Llama)": _load("llama_product_demo.json"),
    }
    colors = {"Qwen3-0.6B": "#2f6b9a", "SmolLM2-135M (Llama)": "#d9782d"}
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for family, artifact in routers.items():
        curves = [_curve(record, "qasper") for record in artifact["runs"]]
        fractions = sorted(curves[0])
        means = [statistics.fmean(curve[value] for curve in curves) for value in fractions]
        stds = [statistics.stdev(curve[value] for curve in curves) for value in fractions]
        axis.plot(
            [100 * value for value in fractions],
            means,
            marker="o",
            linewidth=2,
            label=family,
            color=colors[family],
        )
        axis.fill_between(
            [100 * value for value in fractions],
            [max(0.0, mean - std) for mean, std in zip(means, stds)],
            [min(1.0, mean + std) for mean, std in zip(means, stds)],
            alpha=0.16,
            color=colors[family],
        )
    axis.set_xlim(0, 30)
    axis.set_ylim(0, 0.65)
    axis.set_xlabel("Requested parent chunks (percent)")
    axis.set_ylabel("QASPER evidence-identity recall")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(RESULTS / f"cross_family_qasper_recall.{suffix}", dpi=180)
    plt.close(figure)

    rows = []
    for family, artifact in routers.items():
        qasper = artifact["aggregates"]["qasper"]
        demo = demos[family]["aggregates"]["combined"]
        rows.append(
            {
                "model": family,
                "router_parameters": (
                    262144 if family.startswith("Qwen") else 147456
                ),
                "qasper_r5_mean": qasper["R@5%"]["mean"],
                "qasper_r10_mean": qasper["R@10%"]["mean"],
                "qasper_r20_mean": qasper["R@20%"]["mean"],
                "qasper_r20_std": qasper["R@20%"]["std"],
                "qasper_r30_mean": qasper["R@30%"]["mean"],
                "qasper_auc_0_30_mean": qasper["AUC0-30"]["mean"],
                "demo_evidence_recall": demo["evidence_recall"],
                "demo_routed_f1": demo["routed_f1"],
                "demo_requested_chunk_fraction": demo["requested_chunk_fraction"],
                "demo_materialized_kv_fraction": demo["materialized_kv_token_fraction"],
                "demo_routing_seconds": demo["routing_seconds"],
                "demo_wall_seconds": demo["routed_wall_seconds"],
                "demo_peak_gpu_bytes": demo["peak_gpu_bytes"],
                "demo_routing_index_bytes": demo["routing_index_bytes"],
                "demo_resident_detail_kv_bytes": demo["resident_detail_kv_bytes"],
            }
        )
    with (RESULTS / "cross_family_productization.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": "five-seed QASPER-trained router frontiers plus four-example API systems demo",
        "rows": rows,
        "interpretation": (
            "Both family adapters preserve bounded native-KV execution. Qwen is more stable "
            "across seeds; the Llama-family selected seed is competitive, but neither frozen "
            "model converts retrieval into dependable answer quality."
        ),
    }
    (RESULTS / "cross_family_productization.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
