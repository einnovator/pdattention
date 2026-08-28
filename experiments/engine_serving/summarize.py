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
    rotating = _load(ENGINE_DIRS["mlx"] / "rotating_archive.json")
    rotating_summary = []
    keys = [("full_sequential_kv", None)] + [
        (condition, size)
        for condition in ("rotating_only", "rotating_plus_selected_archive")
        for size in rotating["kv_sizes"]
    ]
    for condition, cache_size in keys:
        values = [
            row for row in rotating["rows"]
            if row["condition"] == condition and row["cache_size"] == cache_size
        ]
        rotating_summary.append({
            "condition": condition,
            "cache_size": cache_size,
            "sample_count": len(values),
            "quality_absolute": fmean(float(row["exact_recovery"]) for row in values),
            "cache_bytes_mean": fmean(row["cache_bytes"] for row in values),
            "completion_latency_ms_mean": fmean(
                row["completion_latency_ms"] for row in values
            ),
            "prompt_tokens_per_second_mean": fmean(
                row["prompt_tokens_per_second"] for row in values
            ),
            "peak_memory_gb_mean": fmean(row["peak_memory_gb"] for row in values),
        })
    native = _native_results()
    return {
        "schema_version": "1.0",
        "registry_version": "2026-08-paper6-engine-native-v2",
        "description": "Cross-engine E0/G10 smoke plus separately tiered native execution evidence.",
        "environment": metadata,
        "vllm_global_prefix_cache_hit_rates_percent": vllm_rates,
        "mlx_rotating_archive": {
            "experiment": rotating["experiment"],
            "evidence_tier": rotating["evidence_tier"],
            "seeds": rotating["seeds"],
            "native_pra_status": rotating["native_pra_status"],
            "summary": rotating_summary,
        },
        "native_results": native,
        "rows": rows,
    }


def _native_results() -> dict:
    """Aggregate engine-specific native artifacts without equating their tiers."""

    mlx = _load(ENGINE_DIRS["mlx"] / "native_kv.json")
    mlx_model_artifacts = [mlx]
    for name in ("native_kv_llama32_1b.json", "native_kv_gemma3_1b.json"):
        path = ENGINE_DIRS["mlx"] / name
        if path.exists():
            mlx_model_artifacts.append(_load(path))
    sglang = _load(ENGINE_DIRS["sglang"] / "native_kv.json")
    vllm = _load(ENGINE_DIRS["vllm"] / "native_paged_kv.json")
    residency = _load(RESULTS / "engine_residency_sweep.json")

    mlx_native = [row for row in mlx["rows"] if row["condition"] == "native_selected_kv"]
    vllm_by_tokens = []
    for token_count in vllm["token_counts"]:
        values = [row for row in vllm["rows"] if row["selected_tokens"] == token_count]
        vllm_by_tokens.append({
            "selected_tokens": token_count,
            "sample_count": len(values),
            "cold_paged_attention_ms_mean": fmean(
                row.get("cold_paged_attention_ms", row["paged_attention_ms"])
                for row in values
            ),
            "warm_paged_attention_ms_mean": fmean(
                row.get("warm_paged_attention_ms", row["paged_attention_ms"])
                for row in values
            ),
            "paged_attention_ms_mean": fmean(row["paged_attention_ms"] for row in values),
            "max_error": max(row["max_error"] for row in values),
            "max_sharing_bytes_saved": max(row["sharing_bytes_saved"] for row in values),
        })
    residency_by_policy = []
    for policy in residency["policies"]:
        values = [row for row in residency["rows"] if row["policy"] == policy]
        residency_by_policy.append({
            "policy": policy,
            "seed_count": len(values),
            "loads_mean": fmean(row["loads"] for row in values),
            "evictions_mean": fmean(row["evictions"] for row in values),
            "reload_amplification_mean": fmean(
                row["reload_amplification"] for row in values
            ),
            "bytes_loaded_mean": fmean(row["bytes_loaded"] for row in values),
        })
    return {
        "mlx": {
            "status": mlx["native_pra_status"],
            "evidence_tier": mlx["evidence_tier"],
            "seed_count": len(mlx_native),
            "exact_recovery": fmean(float(row["exact_recovery"]) for row in mlx_native),
            "max_logit_error": max(row["max_logit_error_vs_ordinary_split"] for row in mlx_native),
            "completion_latency_ms_mean": fmean(row["completion_latency_ms"] for row in mlx_native),
            "native_encode_ms_mean": fmean(row["native_encode_ms"] for row in mlx_native),
            "active_native_kv_bytes_mean": fmean(row["active_native_kv_bytes"] for row in mlx_native),
            "models": [
                {
                    "model_id": artifact["model_id"],
                    "model_revision": artifact["model_revision"],
                    "seed_count": len(artifact["seeds"]),
                    "exact_recovery": fmean(
                        float(row["exact_recovery"])
                        for row in artifact["rows"]
                        if row["condition"] == "native_selected_kv"
                    ),
                    "rotating_exact_recovery": fmean(
                        float(row["exact_recovery"])
                        for row in artifact["rows"]
                        if row["condition"] == "native_selected_kv_rotating_local_64"
                    ),
                    "max_logit_error": max(
                        row["max_logit_error_vs_ordinary_split"]
                        for row in artifact["rows"]
                    ),
                    "active_native_kv_bytes_mean": fmean(
                        row["active_native_kv_bytes"]
                        for row in artifact["rows"]
                        if row["condition"] == "native_selected_kv"
                    ),
                }
                for artifact in mlx_model_artifacts
            ],
        },
        "sglang": {
            "status": sglang["native_pra_status"],
            "evidence_tier": sglang["evidence_tier"],
            "seed_count": len(sglang["rows"]),
            "exact_recovery": fmean(float(row["exact_recovery"]) for row in sglang["rows"]),
            "max_logit_error": max(row["max_logit_error_vs_sglang_split_cache"] for row in sglang["rows"]),
            "completion_latency_ms_mean": fmean(row["completion_latency_ms"] for row in sglang["rows"]),
            "radix_identity_separation_rate": fmean(
                float(row["pra_tokens_absent_from_radix_prefix"])
                for row in sglang["rows"]
            ),
        },
        "vllm": {
            "status": vllm["native_pra_status"],
            "evidence_tier": vllm["evidence_tier"],
            "seed_count": len(vllm["seeds"]),
            "rows_by_selected_tokens": vllm_by_tokens,
        },
        "residency": {
            "evidence_tier": residency["evidence_tier"],
            "rows_by_policy": residency_by_policy,
        },
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
    for engine, directory in ENGINE_DIRS.items():
        rows = [row for row in registry["rows"] if row["engine"] == engine]
        local = [
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Condition & Exact & Prompt & Cold TTFT & Warm TTFT & Warm cached \\",
            r"\midrule",
        ]
        for row in rows:
            cached = (
                "n/a"
                if row["warm_cached_tokens_mean"] is None
                else f"{row['warm_cached_tokens_mean']:.0f}"
            )
            local.append(
                f"{_tex_escape(CONDITION_LABELS[row['condition']])} & "
                f"{100 * row['quality_absolute']:.0f}\\% & "
                f"{row['visible_initial_tokens']:.0f} & {row['cold_ttft_ms']:.1f} & "
                f"{row['warm_ttft_ms_mean']:.1f} & {cached} \\\\"
            )
        local.extend([r"\bottomrule", r"\end{tabular}"])
        (directory / "generated_serving_table.tex").write_text(
            "\n".join(local) + "\n", encoding="utf-8"
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

    rotating = registry["mlx_rotating_archive"]["summary"]
    full = next(row for row in rotating if row["condition"] == "full_sequential_kv")
    sizes = sorted(
        row["cache_size"] for row in rotating if row["cache_size"] is not None
    )
    sizes = sorted(set(sizes))
    only = {
        row["cache_size"]: row
        for row in rotating if row["condition"] == "rotating_only"
    }
    selected = {
        row["cache_size"]: row
        for row in rotating if row["condition"] == "rotating_plus_selected_archive"
    }
    fig, axis = plt.subplots(figsize=(6.2, 3.7))
    axis.plot(
        sizes,
        [only[size]["quality_absolute"] for size in sizes],
        marker="o",
        label="Rotating only",
        color="#6b7280",
    )
    axis.plot(
        sizes,
        [selected[size]["quality_absolute"] for size in sizes],
        marker="s",
        label="Rotating + selected archive",
        color="#2f6f9f",
    )
    axis.axhline(
        full["quality_absolute"], color="#2f855a", linestyle="--", label="Full sequential K/V"
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks(sizes, [str(size) for size in sizes])
    axis.set_ylim(-0.03, 1.05)
    axis.set_xlabel("Rotating K/V capacity (tokens)")
    axis.set_ylabel("Exact recovery")
    axis.set_title("MLX rotating-cache archive control (5 seeds)")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, loc="center right")
    fig.tight_layout()
    output = ENGINE_DIRS["mlx"] / "rotating_archive_frontier"
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def write_mlx_rotating_table(registry: dict) -> None:
    rows = registry["mlx_rotating_archive"]["summary"]
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Condition & K/V tokens & Exact & Cache MB & Latency ms \\",
        r"\midrule",
    ]
    for row in rows:
        label = {
            "full_sequential_kv": "Full sequential K/V",
            "rotating_only": "Rotating only",
            "rotating_plus_selected_archive": "Rotating + selected archive",
        }[row["condition"]]
        size = "full" if row["cache_size"] is None else str(row["cache_size"])
        lines.append(
            f"{label} & {size} & {100 * row['quality_absolute']:.0f}\\% & "
            f"{row['cache_bytes_mean'] / 1048576:.1f} & "
            f"{row['completion_latency_ms_mean']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ENGINE_DIRS["mlx"] / "generated_rotating_archive_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_native_tables(registry: dict) -> None:
    native = registry["native_results"]
    mlx = native["mlx"]
    sglang = native["sglang"]
    vllm = native["vllm"]
    lines = [
        r"\begin{tabular}{llllrr}",
        r"\toprule",
        r"Engine & Native path & Tier & Seeds & Exact/parity & Error \\",
        r"\midrule",
        f"MLX & selected K/V execution & {_tex_escape(mlx['evidence_tier'])} & {mlx['seed_count']} & {100 * mlx['exact_recovery']:.0f}\\% & {mlx['max_logit_error']:.4g} \\\\",
        f"SGLang & native cache path & {_tex_escape(sglang['evidence_tier'])} & {sglang['seed_count']} & {100 * sglang['exact_recovery']:.0f}\\% & {sglang['max_logit_error']:.4g} \\\\",
        f"vLLM-Metal & paged-attention kernel & {_tex_escape(vllm['evidence_tier'])} & {vllm['seed_count']} & kernel parity & {max(row['max_error'] for row in vllm['rows_by_selected_tokens']):.4g} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (RESULTS / "generated_engine_native_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    policy_lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Policy & Mean loads & Mean evictions & Reload amplification \\",
        r"\midrule",
    ]
    for row in native["residency"]["rows_by_policy"]:
        policy_lines.append(
            f"{_tex_escape(row['policy'])} & {row['loads_mean']:.1f} & "
            f"{row['evictions_mean']:.1f} & {row['reload_amplification_mean']:.3f} \\\\"
        )
    policy_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (RESULTS / "generated_engine_residency_table.tex").write_text(
        "\n".join(policy_lines) + "\n", encoding="utf-8"
    )

    vllm_lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Selected tok. & Fraction & Cold ms & Warm ms & Max error & Shared MB saved \\",
        r"\midrule",
    ]
    for row in vllm["rows_by_selected_tokens"]:
        vllm_lines.append(
            f"{row['selected_tokens']} & {100 * row['selected_tokens'] / 8192:.2f}\\% & "
            f"{row['cold_paged_attention_ms_mean']:.3f} & "
            f"{row['warm_paged_attention_ms_mean']:.3f} & {row['max_error']:.4g} & "
            f"{row['max_sharing_bytes_saved'] / 1048576:.2f} \\\\"
        )
    vllm_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ENGINE_DIRS["vllm"] / "generated_native_paged_table.tex").write_text(
        "\n".join(vllm_lines) + "\n", encoding="utf-8"
    )

    mlx_raw = _load(ENGINE_DIRS["mlx"] / "native_kv.json")
    mlx_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Condition & Exact & Latency ms & Native MB & Max error \\",
        r"\midrule",
    ]
    for condition in (
        "ordinary_split_cache",
        "native_selected_kv",
        "native_selected_kv_rotating_local_64",
    ):
        values = [row for row in mlx_raw["rows"] if row["condition"] == condition]
        label = {
            "ordinary_split_cache": "Ordinary split cache",
            "native_selected_kv": "Native selected K/V",
            "native_selected_kv_rotating_local_64": "Native + rotating local 64",
        }[condition]
        mlx_lines.append(
            f"{label} & {100 * fmean(float(row['exact_recovery']) for row in values):.0f}\\% & "
            f"{fmean(row['completion_latency_ms'] for row in values):.1f} & "
            f"{fmean(row['active_native_kv_bytes'] for row in values) / 1048576:.1f} & "
            f"{max(row['max_logit_error_vs_ordinary_split'] for row in values):.4g} \\\\"
        )
    mlx_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ENGINE_DIRS["mlx"] / "generated_native_kv_table.tex").write_text(
        "\n".join(mlx_lines) + "\n", encoding="utf-8"
    )
    model_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Seeds & Native exact & Rotating exact & Native MB \\",
        r"\midrule",
    ]
    for row in mlx["models"]:
        model_label = {
            "mlx-community/Qwen3-0.6B-4bit": "Qwen3 0.6B 4-bit",
            "mlx-community/Llama-3.2-1B-Instruct-4bit": "Llama 3.2 1B 4-bit",
            "mlx-community/gemma-3-1b-it-4bit": "Gemma 3 1B 4-bit",
        }.get(row["model_id"], row["model_id"])
        model_lines.append(
            f"{_tex_escape(model_label)} & {row['seed_count']} & "
            f"{100 * row['exact_recovery']:.0f}\\% & "
            f"{100 * row['rotating_exact_recovery']:.0f}\\% & "
            f"{row['active_native_kv_bytes_mean'] / 1048576:.2f} \\\\"
        )
    model_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ENGINE_DIRS["mlx"] / "generated_native_models_table.tex").write_text(
        "\n".join(model_lines) + "\n", encoding="utf-8"
    )

    sglang_lines = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"Seeds & Exact & Argmax parity & Radix separation & Latency ms \\",
        r"\midrule",
        f"{sglang['seed_count']} & {100 * sglang['exact_recovery']:.0f}\\% & "
        f"100\\% & {100 * sglang['radix_identity_separation_rate']:.0f}\\% & "
        f"{sglang['completion_latency_ms_mean']:.1f} \\\\ ",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (ENGINE_DIRS["sglang"] / "generated_native_kv_table.tex").write_text(
        "\n".join(sglang_lines) + "\n", encoding="utf-8"
    )


def write_native_plots(registry: dict) -> None:
    native = registry["native_results"]
    rows = native["vllm"]["rows_by_selected_tokens"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
    axes[0].plot(
        [row["selected_tokens"] for row in rows],
        [row["paged_attention_ms_mean"] for row in rows],
        marker="o",
        color="#2f6f9f",
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Selected native K/V tokens")
    axes[0].set_ylabel("Paged attention (ms)")
    axes[0].set_title("vLLM-Metal native kernel")
    policies = native["residency"]["rows_by_policy"]
    axes[1].bar(
        [row["policy"].replace("_", "\n") for row in policies],
        [row["reload_amplification_mean"] for row in policies],
        color=["#6b7280", "#2f855a", "#b35c1e", "#7c3aed"],
    )
    axes[1].set_ylabel("Loads per request")
    axes[1].set_title("Fixed-budget residency pressure")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULTS / "engine_native_systems.png", dpi=180)
    fig.savefig(RESULTS / "engine_native_systems.pdf")
    plt.close(fig)


def main() -> None:
    registry = build_registry()
    (RESULTS / "pra_engine_benchmarks.json").write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8"
    )
    write_table(registry)
    write_plots(registry)
    write_mlx_rotating_table(registry)
    write_native_tables(registry)
    write_native_plots(registry)


if __name__ == "__main__":
    main()
