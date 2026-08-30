"""Summarize Apple-Silicon cross-family, promotion, and serving extensions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean


def _nearest(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot summarize an empty metric cohort.")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tex(value: object) -> str:
    return str(value).replace("_", r"\_")


def _async_rows(payloads: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for payload in payloads:
        rows = list(payload["rows"])
        for lead_ms in sorted({int(row["lead_ms"]) for row in rows}):
            selected = [row for row in rows if int(row["lead_ms"]) == lead_ms]
            result.append(
                {
                    "engine": payload["engine"],
                    "lead_ms": lead_ms,
                    "examples": len(selected),
                    "ready_rate": fmean(bool(row["ready_at_demand"]) for row in selected),
                    "exact_rate": fmean(bool(row["output_exact"]) for row in selected),
                    "demand_stall_p50_ms": _nearest(
                        [float(row["demand_stall_ms"]) for row in selected], 0.50
                    ),
                    "demand_stall_p95_ms": _nearest(
                        [float(row["demand_stall_ms"]) for row in selected], 0.95
                    ),
                    "demand_to_hot_p50": _nearest(
                        [float(row["demand_to_hot_ratio"]) for row in selected], 0.50
                    ),
                    "total_p50_ms": _nearest(
                        [float(row["prefetch_to_completion_ms"]) for row in selected],
                        0.50,
                    ),
                }
            )
    return result


def _write_async_table(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Engine & Lead & Ready & Stall p50 & Stall p95 & Demand/HOT \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['engine']} & {row['lead_ms']} ms & "
            f"{100 * float(row['ready_rate']):.0f}\\% & "
            f"{float(row['demand_stall_p50_ms']):.1f} & "
            f"{float(row['demand_stall_p95_ms']):.1f} & "
            f"{float(row['demand_to_hot_p50']):.3f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_quant_table(path: Path, summary: dict[str, object]) -> None:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Profile & Exact & First & Prefix & $\Delta$F1 & $\Delta\log p$ \\",
        r"\midrule",
    ]
    for profile, row in summary.items():
        examples = max(1, int(row["examples"]))
        lines.append(
            f"{_tex(profile)} & "
            f"{100 * int(row['exact_outputs']) / examples:.1f}\\% & "
            f"{100 * int(row['first_token_equal']) / examples:.1f}\\% & "
            f"{float(row['mean_common_prefix_tokens']):.1f} & "
            f"{float(row['mean_f1_delta']):+.4f} & "
            f"{float(row['mean_logprob_delta']):+.4f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_concurrency_table(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"Workload & Tier & $c$ & req/s & p50 & p95 & p99 & queue p95 & exact/HOT \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_tex(row['workload'])} & "
            f"{_tex(row['tier'])} & {row['concurrency']} & "
            f"{float(row['requests_per_second']):.2f} & "
            f"{float(row['request_p50_ms']):.0f} & "
            f"{float(row['request_p95_ms']):.0f} & "
            f"{float(row['request_p99_ms']):.0f} & "
            f"{float(row.get('model_queue_p95_ms', 0.0)):.0f} & "
            f"{100 * float(row['exact_vs_hot_rate']):.1f}\\% \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_concurrency(rows: list[dict[str, object]]) -> None:
    """Reject contaminated lossless controls before generating paper artifacts."""

    invalid = [
        row
        for row in rows
        if row["tier"] in {"hot", "warm"}
        and float(row["exact_vs_hot_rate"]) != 1.0
    ]
    if invalid:
        cells = ", ".join(
            f"{row['workload']}/{row['tier']}/c={row['concurrency']}"
            for row in invalid
        )
        raise ValueError(
            "Lossless storage concurrency rows diverge from HOT baseline: " + cells
        )


def _plot_concurrency(path: Path, rows: list[dict[str, object]]) -> None:
    """Plot throughput and tail latency without conflating the two workloads."""

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for workload in sorted({str(row["workload"]) for row in rows}):
        for tier in sorted({str(row["tier"]) for row in rows}):
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["workload"] == workload and row["tier"] == tier
                ),
                key=lambda row: int(row["concurrency"]),
            )
            if not selected:
                continue
            label = f"{workload.replace('_', ' ')} / {tier.upper()}"
            concurrency = [int(row["concurrency"]) for row in selected]
            axes[0].plot(
                concurrency,
                [float(row["requests_per_second"]) for row in selected],
                marker="o",
                label=label,
            )
            axes[1].plot(
                concurrency,
                [float(row["request_p95_ms"]) for row in selected],
                marker="o",
                label=label,
            )
    axes[0].set_ylabel("Requests/s")
    axes[1].set_ylabel("Request p95 (ms)")
    for axis in axes:
        axis.set_xlabel("Concurrent sessions")
        axis.set_xscale("log", base=2)
        axis.set_xticks((1, 2, 4, 8, 16))
        axis.set_xticklabels(("1", "2", "4", "8", "16"))
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0.16, 1, 1))
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=180)
    plt.close(figure)


def _write_online_table(path: Path, payloads: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Engine & $c$ & req/s & TTFT p50 & TTFT p95 & request p50 & request p95 \\",
        r"\midrule",
    ]
    for payload in payloads:
        for row in payload["concurrency_rows"]:
            lines.append(
                f"{payload['engine']} & {row['concurrency']} & "
                f"{float(row['requests_per_second']):.2f} & "
                f"{float(row['ttft_p50_ms']):.0f} & "
                f"{float(row['ttft_p95_ms']):.0f} & "
                f"{float(row['request_p50_ms']):.0f} & "
                f"{float(row['request_p95_ms']):.0f} \\\\"
            )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pressure_rows(payloads: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate sustained-session residency by dataset and HOT capacity."""

    result = []
    for payload in payloads:
        for budget in payload["resident_resource_budgets"]:
            rows = [
                row
                for row in payload["rows"]
                if int(row["resident_resource_budget"]) == int(budget)
            ]
            summaries = [
                row
                for row in payload["seed_summaries"]
                if int(row["resident_resource_budget"]) == int(budget)
            ]
            final = [row for row in rows if row["final_revisit"]]
            result.append(
                {
                    "dataset": payload["dataset"],
                    "resident_resource_budget": int(budget),
                    "resources_per_seed": int(payload["resources_per_seed"]),
                    "session_rounds": int(payload["session_rounds"]),
                    "requests": len(rows),
                    "reloads_mean": fmean(float(row["reloads"]) for row in summaries),
                    "evictions_mean": fmean(
                        float(row["evictions"]) for row in summaries
                    ),
                    "final_revisit_reload_rate": fmean(
                        bool(row["reload_on_request"]) for row in final
                    ),
                    "token_f1": fmean(float(row["token_f1"]) for row in rows),
                    "gold_answer_logprob": fmean(
                        float(row["gold_answer_logprob"]) for row in rows
                    ),
                    "resolve_p95_ms": _nearest(
                        [float(row["resolve_ms"]) for row in rows], 0.95
                    ),
                    "completion_p95_ms": _nearest(
                        [float(row["completion_latency_ms"]) for row in rows], 0.95
                    ),
                }
            )
    return result


def _write_pressure_table(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Dataset & HOT resources & Requests & Reloads & Evictions & Final reload & F1 & p95 ms \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_tex(row['dataset'])} & {row['resident_resource_budget']} & "
            f"{row['requests']} & {float(row['reloads_mean']):.1f} & "
            f"{float(row['evictions_mean']):.1f} & "
            f"{100 * float(row['final_revisit_reload_rate']):.0f}\\% & "
            f"{float(row['token_f1']):.3f} & "
            f"{float(row['completion_p95_ms']):.0f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_pressure(path: Path, rows: list[dict[str, object]]) -> None:
    """Plot reload pressure separately for each natural workload."""

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        selected = sorted(
            (row for row in rows if row["dataset"] == dataset),
            key=lambda row: int(row["resident_resource_budget"]),
        )
        budgets = [int(row["resident_resource_budget"]) for row in selected]
        axes[0].plot(
            budgets,
            [float(row["reloads_mean"]) for row in selected],
            marker="o",
            label=dataset,
        )
        axes[1].plot(
            budgets,
            [100 * float(row["final_revisit_reload_rate"]) for row in selected],
            marker="o",
            label=dataset,
        )
    axes[0].set_ylabel("Mean reloads / seed")
    axes[1].set_ylabel("Final revisit reload (%)")
    for axis in axes:
        axis.set_xlabel("HOT resource budget")
        axis.set_xticks((2, 4, 8))
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=180)
    plt.close(figure)


def _tier_window_rows(payload: dict[str, object] | None) -> list[dict[str, object]]:
    if payload is None:
        return []
    result = []
    rows = list(payload["rows"])
    for hot in payload["hot_resource_budgets"]:
        for warm in payload["warm_resource_budgets"]:
            for window in payload["local_kv_sizes"]:
                selected = [
                    row
                    for row in rows
                    if int(row["hot_resource_budget"]) == int(hot)
                    and int(row["warm_resource_budget"]) == int(warm)
                    and int(row["local_kv_size"]) == int(window)
                ]
                result.append(
                    {
                        "hot_resource_budget": int(hot),
                        "warm_resource_budget": int(warm),
                        "local_kv_size": int(window),
                        "requests": len(selected),
                        "hot_start_rate": fmean(
                            row["tier_before"] == "hot" for row in selected
                        ),
                        "warm_start_rate": fmean(
                            row["tier_before"] == "warm" for row in selected
                        ),
                        "source_start_rate": fmean(
                            row["tier_before"] == "source" for row in selected
                        ),
                        "token_f1": fmean(float(row["token_f1"]) for row in selected),
                        "resolve_p50_ms": _nearest(
                            [float(row["resolve_ms"]) for row in selected], 0.50
                        ),
                        "resolve_p95_ms": _nearest(
                            [float(row["resolve_ms"]) for row in selected], 0.95
                        ),
                        "completion_p95_ms": _nearest(
                            [float(row["completion_latency_ms"]) for row in selected],
                            0.95,
                        ),
                    }
                )
    return result


def _write_tier_window_table(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{tabular}{rrrrrrrrrr}",
        r"\toprule",
        r"HOT & WARM & Window & Requests & HOT start & WARM start & SOURCE start & F1 & Resolve p95 & Total p95 \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['hot_resource_budget']} & {row['warm_resource_budget']} & "
            f"{row['local_kv_size']} & {row['requests']} & "
            f"{100 * float(row['hot_start_rate']):.0f}\\% & "
            f"{100 * float(row['warm_start_rate']):.0f}\\% & "
            f"{100 * float(row['source_start_rate']):.0f}\\% & "
            f"{float(row['token_f1']):.3f} & "
            f"{float(row['resolve_p95_ms']):.1f} & "
            f"{float(row['completion_p95_ms']):.0f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_tier_window(path: Path, rows: list[dict[str, object]]) -> None:
    """Plot local-window effects at each physical HOT/WARM budget pair."""

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    budget_pairs = sorted(
        {
            (
                int(row["hot_resource_budget"]),
                int(row["warm_resource_budget"]),
            )
            for row in rows
        }
    )
    for hot, warm in budget_pairs:
        selected = sorted(
            (
                row
                for row in rows
                if int(row["hot_resource_budget"]) == hot
                and int(row["warm_resource_budget"]) == warm
            ),
            key=lambda row: int(row["local_kv_size"]),
        )
        label = f"HOT {hot} / WARM {warm}"
        windows = [int(row["local_kv_size"]) for row in selected]
        axes[0].plot(
            windows,
            [float(row["resolve_p95_ms"]) for row in selected],
            marker="o",
            label=label,
        )
        axes[1].plot(
            windows,
            [float(row["token_f1"]) for row in selected],
            marker="o",
            label=label,
        )
    axes[0].set_ylabel("Resolve p95 (ms)")
    axes[1].set_ylabel("Token F1")
    for axis in axes:
        axis.set_xlabel("Local rotating K/V window")
        axis.set_xscale("log", base=2)
        axis.set_xticks((64, 256))
        axis.set_xticklabels(("64", "256"))
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("docs/papers/shared/results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/papers/shared/results/mac_engine_extension"),
    )
    args = parser.parse_args()
    root = args.results_root
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    async_payloads = [
        _load(root / "paper6_2_mlx" / "async_warm_promotion_qasper.json"),
        _load(root / "paper6_1_sglang" / "async_warm_promotion_qasper.json"),
    ]
    quantization = _load(
        root / "paper6_2_mlx" / "selective_kv_quantization_qasper.json"
    )
    concurrency = _load(
        root / "paper6_2_mlx" / "live_storage_concurrency_qasper.json"
    )
    _validate_concurrency(concurrency["rows"])
    online_paths = (
        root / "paper6_2_mlx" / "online_native_gateway_qasper.json",
        root / "paper6_1_sglang" / "online_native_gateway_qasper.json",
    )
    online = [_load(path) for path in online_paths if path.exists()]
    pressure = [
        _load(path)
        for path in (
            root / "paper6_2_mlx" / "long_session_pressure_qasper.json",
            root / "paper6_2_mlx" / "long_session_pressure_hotpotqa.json",
            root / "paper6_2_mlx" / "long_session_pressure_2wikimultihopqa.json",
        )
        if path.exists()
    ]
    tier_window_path = root / "paper6_2_mlx" / "tier_window_pressure_qasper.json"
    tier_window = _tier_window_rows(
        _load(tier_window_path) if tier_window_path.exists() else None
    )
    async_summary = _async_rows(async_payloads)
    pressure_summary = _pressure_rows(pressure)
    summary = {
        "schema_version": "1.0",
        "experiment": "paper6_mac_engine_extension_summary_v1",
        "async_warm": async_summary,
        "selective_quantization": quantization["summary"],
        "storage_concurrency": concurrency["rows"],
        "online_gateway": online,
        "long_session_pressure": pressure_summary,
        "tier_window_pressure": tier_window,
    }
    (output / "mac_engine_extension_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_async_table(output / "generated_async_warm_table.tex", async_summary)
    _write_quant_table(
        output / "generated_selective_quantization_table.tex",
        quantization["summary"],
    )
    _write_concurrency_table(
        output / "generated_storage_concurrency_table.tex", concurrency["rows"]
    )
    if online:
        _write_online_table(output / "generated_online_gateway_table.tex", online)
    if pressure_summary:
        _write_pressure_table(
            output / "generated_long_session_pressure_table.tex", pressure_summary
        )
    if tier_window:
        _write_tier_window_table(
            output / "generated_tier_window_pressure_table.tex", tier_window
        )

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    for engine in sorted({str(row["engine"]) for row in async_summary}):
        rows = [row for row in async_summary if row["engine"] == engine]
        axis.plot(
            [row["lead_ms"] for row in rows],
            [row["demand_to_hot_p50"] for row in rows],
            marker="o",
            label=engine,
        )
    axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axis.set_xlabel("Prefetch lead (ms)")
    axis.set_ylabel("Demand-path latency / HOT")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "async_warm_frontier.pdf")
    figure.savefig(output / "async_warm_frontier.png", dpi=180)
    plt.close(figure)
    _plot_concurrency(output / "storage_concurrency", concurrency["rows"])
    if pressure_summary:
        _plot_pressure(output / "long_session_pressure", pressure_summary)
    if tier_window:
        _plot_tier_window(output / "tier_window_pressure", tier_window)


if __name__ == "__main__":
    main()
