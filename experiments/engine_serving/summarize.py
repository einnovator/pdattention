"""Generate the shared engine registry, TeX table, and smoke plots."""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs" / "papers" / "shared" / "results"
ENGINE_DIRS = {
    "vllm": RESULTS / "paper6_vllm",
    "sglang": RESULTS / "paper6_1_sglang",
    "mlx": RESULTS / "paper6_2_mlx",
}
CONDITION_LABELS = {
    "no_prefix_no_pra": "None",
    "prefix_only": "Prefix",
    "pra_only": "Selected",
    "prefix_plus_pra": "Prefix + selected",
    "full_context": "Full context",
}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sglang_cache_rows(path: Path, repeats: int) -> dict[str, list[int]]:
    pattern = re.compile(r"#cached-token: (\d+)")
    values = [int(match.group(1)) for match in map(pattern.search, path.read_text().splitlines()) if match]
    conditions = list(CONDITION_LABELS)
    if len(values) != len(conditions) * repeats:
        raise ValueError(f"Expected {len(conditions) * repeats} SGLang cache rows, got {len(values)}.")
    return {
        condition: values[index * repeats : (index + 1) * repeats]
        for index, condition in enumerate(conditions)
    }


def _vllm_global_hit_rates(path: Path) -> list[float]:
    pattern = re.compile(r"Prefix cache hit rate: ([0-9.]+)%")
    return [float(match.group(1)) for match in map(pattern.search, path.read_text().splitlines()) if match]


def build_registry() -> dict:
    rows = []
    metadata = {}
    sglang_cache = None
    for engine, directory in ENGINE_DIRS.items():
        result = _load(directory / "serving_smoke.json")
        environment = _load(directory / "environment.json")
        metadata[engine] = environment
        if engine == "sglang":
            sglang_cache = _sglang_cache_rows(
                directory / "engine_log_extract.txt", result["repeats"]
            )
        aggregates = {row["condition"]: row for row in result["aggregates"]}
        samples = result["samples"]
        reference = aggregates["full_context"]["quality_success_rate"]
        for condition, aggregate in aggregates.items():
            condition_samples = [row for row in samples if row["condition"] == condition]
            cached_values = [
                row["cached_tokens"]
                for row in condition_samples
                if row["cached_tokens"] is not None
            ]
            cache_source = "response_usage" if cached_values else "NOT_MEASURED"
            warm_cached = None
            if engine == "sglang" and sglang_cache is not None:
                cached_values = sglang_cache[condition]
                warm_cached = fmean(cached_values[1:])
                cache_source = "scheduler_log"
            elif len(cached_values) > 1:
                warm_cached = fmean(cached_values[1:])
            rows.append({
                "model_family": "Qwen3",
                "model_id": environment["model_id"],
                "model_revision": environment["model_revision"],
                "parameter_count": 600_000_000,
                "num_layers": 28,
                "workload": "prefix_pra_complementarity_smoke",
                "dataset": "synthetic_codeword_memory_v1",
                "split": "fixed",
                "sample_count": aggregate["sample_count"],
                "seed_count": 0,
                "profile": "CANDIDATE",
                "profile_registry_version": "2026-08-product-profile-v2",
                "condition": condition,
                "engine": engine,
                "engine_version": environment["engine_version"],
                "hardware": environment["hardware"],
                "precision": environment["precision"],
                "quality_metric_name": "exact_codeword_recovery",
                "quality_absolute": aggregate["quality_success_rate"],
                "quality_reference": reference,
                "quality_delta": aggregate["quality_success_rate"] - reference,
                "quality_retention": (
                    aggregate["quality_success_rate"] / reference if reference else None
                ),
                "visible_initial_tokens": aggregate["mean_prompt_tokens"],
                "visible_recovered_tokens": 0,
                "materialized_tokens": "NOT_MEASURED_NATIVE_KV",
                "active_native_kv_tokens": "NOT_MEASURED",
                "active_native_kv_bytes": "NOT_MEASURED",
                "detail_kv_bytes": "NOT_MEASURED",
                "address_index_bytes": "NOT_MEASURED",
                "backing_bytes": "NOT_MEASURED",
                "cold_ttft_ms": aggregate["cold_ttft_ms"],
                "warm_ttft_ms_mean": aggregate["warm_ttft_ms_mean"],
                "ttft_ms_p50": aggregate["ttft_ms_p50"],
                "completion_latency_ms_p50": aggregate["completion_latency_ms_p50"],
                "tail_latency_status": aggregate["tail_latency_status"],
                "warm_cached_tokens_mean": warm_cached,
                "cache_metric_source": cache_source,
                "evidence_tier": "SMOKE",
                "measurement_status": "MEASURED",
                "native_pra_status": environment["native_pra_status"],
            })
    vllm_rates = _vllm_global_hit_rates(
        ENGINE_DIRS["vllm"] / "engine_log_extract.txt"
    )
    return {
        "schema_version": "1.0",
        "registry_version": "2026-08-paper6-engine-smoke-v1",
        "description": "Cross-engine E0/G10 prefix and selected-text smoke; no row claims native PRA K/V.",
        "environment": metadata,
        "vllm_global_prefix_cache_hit_rates_percent": vllm_rates,
        "rows": rows,
    }


def _tex_escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def write_table(registry: dict) -> None:
    selected = [
        row for row in registry["rows"]
        if row["condition"] in {"no_prefix_no_pra", "prefix_plus_pra", "full_context"}
    ]
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Engine & Condition & Exact & Prompt tok. & Cold TTFT & Warm TTFT \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(
            f"{_tex_escape(row['engine'])} & {_tex_escape(CONDITION_LABELS[row['condition']])} & "
            f"{100 * row['quality_absolute']:.0f}\\% & {row['visible_initial_tokens']:.0f} & "
            f"{row['cold_ttft_ms']:.1f} & {row['warm_ttft_ms_mean']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (RESULTS / "generated_engine_smoke_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_plots(registry: dict) -> None:
    engines = list(ENGINE_DIRS)
    colors = {"vllm": "#2f6f9f", "sglang": "#2f855a", "mlx": "#b35c1e"}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    width = 0.24
    conditions = ["prefix_plus_pra", "full_context"]
    for engine_index, engine in enumerate(engines):
        engine_rows = {row["condition"]: row for row in registry["rows"] if row["engine"] == engine}
        offset = (engine_index - 1) * width
        axes[0].bar(
            [index + offset for index in range(len(conditions))],
            [engine_rows[condition]["visible_initial_tokens"] for condition in conditions],
            width=width,
            label=engine,
            color=colors[engine],
        )
        axes[1].bar(
            [index + offset for index in range(len(conditions))],
            [engine_rows[condition]["warm_ttft_ms_mean"] for condition in conditions],
            width=width,
            label=engine,
            color=colors[engine],
        )
    labels = [CONDITION_LABELS[value] for value in conditions]
    axes[0].set_xticks(range(len(conditions)), labels)
    axes[0].set_ylabel("Prompt tokens")
    axes[0].set_title("Selected text versus full context")
    axes[1].set_xticks(range(len(conditions)), labels)
    axes[1].set_ylabel("Warm TTFT (ms)")
    axes[1].set_title("Warm serving smoke")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULTS / "engine_smoke_frontier.png", dpi=180)
    fig.savefig(RESULTS / "engine_smoke_frontier.pdf")
    plt.close(fig)


def main() -> None:
    registry = build_registry()
    (RESULTS / "pra_engine_benchmarks.json").write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8"
    )
    write_table(registry)
    write_plots(registry)


if __name__ == "__main__":
    main()

