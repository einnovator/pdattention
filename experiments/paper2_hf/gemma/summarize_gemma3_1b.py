"""Consolidate completed Gemma 3 artifacts and render publication figures."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = (
    ROOT / "docs" / "papers" / "shared" / "results" / "paper2_hf" / "gemma3_1b"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate_row(rows: list[dict], condition: str, dataset: str = "combined") -> dict:
    return next(
        row for row in rows if row["condition"] == condition and row["dataset"] == dataset
    )


def _plot_recall(router: dict, output_dir: Path) -> None:
    colors = {"qasper": "#2878B5", "hotpotqa": "#D95F02"}
    figure, axis = plt.subplots(figsize=(6.6, 4.1))
    for dataset, label in (("qasper", "QASPER"), ("hotpotqa", "HotpotQA transfer")):
        curves = [run["test"][dataset]["curve"] for run in router["runs"]]
        fractions = [float(row["fraction"]) for row in curves[0] if row["fraction"] <= 0.30]
        means = []
        deviations = []
        for index in range(len(fractions)):
            values = [float(curve[index]["recall"]) for curve in curves]
            means.append(statistics.fmean(values))
            deviations.append(statistics.stdev(values))
        axis.plot(fractions, means, marker="o", linewidth=2, label=label, color=colors[dataset])
        axis.fill_between(
            fractions,
            [max(0.0, mean - std) for mean, std in zip(means, deviations)],
            [min(1.0, mean + std) for mean, std in zip(means, deviations)],
            alpha=0.17,
            color=colors[dataset],
        )
    axis.plot([0.0, 0.30], [0.0, 0.30], linestyle="--", color="#666666", label="Random ranking")
    axis.set(xlabel="Selected parent fraction", ylabel="Evidence-identity recall", xlim=(0, 0.30), ylim=(0, 0.58))
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"gemma3_1b_recall_sparsity.{suffix}", dpi=220)
    plt.close(figure)


def summarize(results_dir: Path) -> dict:
    parity = _read(results_dir / "parity_native_kv.json")
    router = _read(results_dir / "router_five_seed.json")
    product = _read(results_dir / "product_demo.json")
    oracle = _read(results_dir / "oracle" / "oracle_memory_use.json")
    config = _read(ROOT / router["router_directory"] / "config.json")
    combined_product = product["aggregates"]["combined"]
    combined_oracle = [row for row in oracle["aggregates"] if row["dataset"] == "combined"]
    learned = _aggregate_row(combined_oracle, "learned_router")
    forced = _aggregate_row(combined_oracle, "oracle_last_1")
    direct = _aggregate_row(combined_oracle, "direct_text_oracle")
    router_parameters = 2 * int(config["input_dim"]) * int(config["routing_dim"])
    summary = {
        "status": "completed",
        "model_id": parity["model_id"],
        "model_revision": parity["model_revision"],
        "tokenizer_revision": parity["tokenizer_revision"],
        "base_parameters": parity["base_parameters"],
        "disabled_parity": parity["disabled_parity"],
        "native_limit_violations": {
            "parity": parity["long_prompt"]["native_limit_violations"],
            "product_demo": max(row["native_limit_violations"] for row in product["rows"]),
            "causal_probe": oracle["native_limit_violations"],
        },
        "global_attention_layers": parity["architecture_audit"]["global_attention_layers"],
        "routing_layer": router["routing_layer"],
        "consumption_layers": product["resolved_consumption_layers"],
        "router_parameters": router_parameters,
        "router_percent_of_model": 100.0 * router_parameters / parity["base_parameters"],
        "five_seed_identity_recall": router["aggregates"],
        "routing_index_bytes": combined_product["routing_index_bytes"],
        "detail_kv_bytes": combined_product["resident_detail_kv_bytes"],
        "routing_index_percent_of_detail_kv": 100.0
        * combined_product["routing_index_bytes"]
        / combined_product["resident_detail_kv_bytes"],
        "requested_chunk_fraction": combined_product["requested_chunk_fraction"],
        "materialized_kv_token_fraction": combined_product["materialized_kv_token_fraction"],
        "peak_gpu_bytes": combined_product["peak_gpu_bytes"],
        "transfer_bytes_across_layers": combined_product["transfer_bytes_across_layers"],
        "ttft_seconds": combined_product["ttft_seconds"],
        "tpot_seconds": combined_product["tpot_seconds"],
        "routing_seconds": combined_product["routing_seconds"],
        "causal_probe_examples": learned["examples"],
        "learned_gold_mean_logprob_delta": learned["gold_mean_logprob_delta_vs_none"],
        "forced_gold_mean_logprob_delta": forced["gold_mean_logprob_delta_vs_none"],
        "direct_text_gold_mean_logprob_delta": direct["gold_mean_logprob_delta_vs_none"],
        "learned_f1": learned["f1"],
        "no_memory_f1": _aggregate_row(combined_oracle, "no_memory")["f1"],
        "direct_text_f1": direct["f1"],
    }
    (results_dir / "gemma3_1b_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    table_rows = []
    for dataset in ("qasper", "hotpotqa", "combined"):
        row = {"dataset": dataset}
        for metric, value in router["aggregates"][dataset].items():
            row[f"{metric}_mean"] = value["mean"]
            row[f"{metric}_std"] = value["std"]
        table_rows.append(row)
    keys = sorted({key for row in table_rows for key in row})
    with (results_dir / "gemma3_1b_router_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(table_rows)
    _plot_recall(router, results_dir)
    suite = {
        "status": "completed",
        "model_id": parity["model_id"],
        "model_revision": parity["model_revision"],
        "tokenizer_revision": parity["tokenizer_revision"],
        "seeds": router["seeds"],
        "routing_layer": router["routing_layer"],
        "consumption_layers": product["resolved_consumption_layers"],
        "parity": str((results_dir / "parity_native_kv.json").relative_to(ROOT)),
        "features": str((results_dir / "features").relative_to(ROOT)),
        "router": router["router_directory"],
        "router_metrics": str((results_dir / "router_five_seed.json").relative_to(ROOT)),
        "product_demo": str((results_dir / "product_demo.json").relative_to(ROOT)),
        "oracle": str((results_dir / "oracle").relative_to(ROOT)),
        "summary": str((results_dir / "gemma3_1b_summary.json").relative_to(ROOT)),
    }
    clean_smoke = results_dir / "clean_install_smoke.json"
    if clean_smoke.exists():
        suite["clean_install_smoke"] = str(clean_smoke.relative_to(ROOT))
    (results_dir / "suite_status.json").write_text(
        json.dumps(suite, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(summarize(parse_args().results_dir), indent=2, sort_keys=True))
