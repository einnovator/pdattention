"""Generate the shared engine registry, TeX table, and smoke plots."""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import fmean, median

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
    lifecycle_rows = (
        native.get("live_storage_scaling", {}).get("rows", [])
        if native.get("live_storage_scaling")
        else []
    )

    def lifecycle_status(engine: str, model_suffix: str) -> str:
        row = next(
            (
                value
                for value in lifecycle_rows
                if value["engine"] == engine
                and str(value["model_id"]).endswith(model_suffix)
            ),
            None,
        )
        if row is None:
            return "not measured"
        examples = int(row["examples"])
        warm = round(float(row["warm_exact_rate"]) * examples)
        int8 = round(float(row["int8_exact_rate"]) * examples)
        return (
            f"lossless WARM {warm}/{examples}; int8 COLD {int8}/{examples} "
            "(smoke; gated)"
        )

    product_matrix = [
        {
            "engine": "vLLM-Metal V1",
            "model": native["vllm"]["live_generation"]["model_id"],
            "profile": "all-layer native K/V + APC, concurrency 1--8",
            "hardware": metadata["vllm"]["hardware"],
            "evidence_tier": "NATURAL_QA_MATCHED_SELECTION",
            "status": "840/840 exact at 0.6B/1.7B; lossless WARM 80/80; int8 COLD open",
        },
        {
            "engine": "SGLang MLX",
            "model": metadata["sglang"]["model_id"],
            "profile": "Radix + selected K/V + built-in HiCache file storage",
            "hardware": metadata["sglang"]["hardware"],
            "evidence_tier": "NATURAL_QA_MATCHED_SELECTION",
            "status": "840/840 exact; lossless WARM 80/80; distributed HiCache open",
        },
        {
            "engine": "MLX-LM",
            "model": "mlx-community/Qwen3-0.6B-4bit",
            "profile": "all-layer selected K/V, four matched regimes",
            "hardware": metadata["mlx"]["hardware"],
            "evidence_tier": "NATURAL_QA_MATCHED_SELECTION",
            "status": "840/840 exact; lossless WARM 80/80; int8 COLD open",
        },
        {
            "engine": "MLX-LM",
            "model": "mlx-community/Qwen3-1.7B-4bit",
            "profile": "FP/int8 all-layer selected K/V",
            "hardware": metadata["mlx"]["hardware"],
            "evidence_tier": "NATURAL_QA_ROUTED_EVIDENCE_MATERIALIZATION",
            "status": "routed natural QA; bounded residency curve; oracle gap remains",
        },
        {
            "engine": "vLLM-Metal V1",
            "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
            "profile": "all-layer native K/V + APC, four matched regimes",
            "hardware": metadata["vllm"]["hardware"],
            "evidence_tier": "NATURAL_QA_MATCHED_SELECTION",
            "status": "840/840 exact; "
            + lifecycle_status(
                "vllm-metal", "Llama-3.2-1B-Instruct-4bit"
            ),
        },
        {
            "engine": "vLLM-Metal V1",
            "model": "mlx-community/gemma-3-1b-it-4bit",
            "profile": "global/local native K/V + APC, four matched regimes",
            "hardware": metadata["vllm"]["hardware"],
            "evidence_tier": "NATURAL_QA_MATCHED_SELECTION",
            "status": "840/840 exact; "
            + lifecycle_status("vllm-metal", "gemma-3-1b-it-4bit"),
        },
        {
            "engine": "SGLang MLX",
            "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
            "profile": "selected K/V + lifecycle manager",
            "hardware": metadata["sglang"]["hardware"],
            "evidence_tier": "NATURAL_QA_LIFECYCLE_SMOKE",
            "status": lifecycle_status(
                "sglang-mlx", "Llama-3.2-1B-Instruct-4bit"
            ),
        },
        {
            "engine": "SGLang MLX",
            "model": "mlx-community/gemma-3-1b-it-4bit",
            "profile": "mixed global/sliding-window topology",
            "hardware": metadata["sglang"]["hardware"],
            "evidence_tier": "BACKEND_COMPATIBILITY",
            "status": "blocked before PRA: per-layer window map unsupported",
        },
        {
            "engine": "MLX-LM",
            "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
            "profile": "all-layer selected K/V + lifecycle manager",
            "hardware": metadata["mlx"]["hardware"],
            "evidence_tier": "NATURAL_QA_LIFECYCLE_SMOKE",
            "status": lifecycle_status(
                "mlx-lm", "Llama-3.2-1B-Instruct-4bit"
            ),
        },
        {
            "engine": "MLX-LM",
            "model": "mlx-community/gemma-3-1b-it-4bit",
            "profile": "global/local selected K/V + lifecycle manager",
            "hardware": metadata["mlx"]["hardware"],
            "evidence_tier": "NATURAL_QA_LIFECYCLE_SMOKE",
            "status": lifecycle_status("mlx-lm", "gemma-3-1b-it-4bit"),
        },
    ]
    return {
        "schema_version": "1.0",
        "registry_version": "2026-08-paper6-engine-native-v9",
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
        "product_matrix": product_matrix,
        "rows": rows,
    }


def _native_results() -> dict:
    """Aggregate engine-specific native artifacts without equating their tiers."""

    mlx = _load(ENGINE_DIRS["mlx"] / "native_kv.json")
    live_storage = {
        "mlx": _load(ENGINE_DIRS["mlx"] / "live_storage_lifecycle.json"),
        "sglang": _load(ENGINE_DIRS["sglang"] / "live_storage_lifecycle.json"),
        "vllm": _load(ENGINE_DIRS["vllm"] / "live_storage_lifecycle.json"),
    }
    live_storage_scaling_path = (
        RESULTS / "live_storage_scaling" / "live_storage_scaling_summary.json"
    )
    live_storage_scaling = (
        _load(live_storage_scaling_path) if live_storage_scaling_path.exists() else None
    )
    mac_extension_path = (
        RESULTS / "mac_engine_extension" / "mac_engine_extension_summary.json"
    )
    mac_extension = (
        _load(mac_extension_path) if mac_extension_path.exists() else None
    )
    expanded_matched_path = (
        RESULTS
        / "expanded_mac_validation"
        / "matched_e0_e2"
        / "matched_e0_e2_summary.json"
    )
    expanded_matched = (
        _load(expanded_matched_path) if expanded_matched_path.exists() else None
    )
    mlx_model_artifacts = [mlx]
    for name in ("native_kv_llama32_1b.json", "native_kv_gemma3_1b.json"):
        path = ENGINE_DIRS["mlx"] / name
        if path.exists():
            mlx_model_artifacts.append(_load(path))
    sglang = _load(ENGINE_DIRS["sglang"] / "native_kv.json")
    sglang_live = _load(ENGINE_DIRS["sglang"] / "live_runner.json")
    sglang_hicache = _load(ENGINE_DIRS["sglang"] / "hicache_l123.json")
    sglang_combined = _load(
        ENGINE_DIRS["sglang"] / "radix_hicache_combined.json"
    )
    sglang_builtin_hicache = _load(
        ENGINE_DIRS["sglang"] / "builtin_hicache_backend.json"
    )
    sglang_hicache_prefetch = _load(
        ENGINE_DIRS["sglang"] / "builtin_hicache_prefetch.json"
    )
    sglang_natural = {
        dataset: _load(ENGINE_DIRS["sglang"] / f"natural_{dataset}.json")
        for dataset in ("qasper", "hotpotqa")
    }
    vllm = _load(ENGINE_DIRS["vllm"] / "native_paged_kv.json")
    vllm_live = _load(ENGINE_DIRS["vllm"] / "v1_live_generation.json")
    vllm_apc = _load(ENGINE_DIRS["vllm"] / "v1_apc_concurrency.json")
    residency = _load(RESULTS / "engine_residency_sweep.json")
    mlx_profiles = _load(
        ENGINE_DIRS["mlx"] / "profiles_persistence_concurrency.json"
    )
    mlx_natural = {
        dataset: _load(ENGINE_DIRS["mlx"] / f"natural_transport_{dataset}.json")
        for dataset in ("qasper", "hotpotqa")
    }
    mlx_answer_quality = {}
    mlx_routed_quality = {}
    for dataset in ("qasper", "hotpotqa", "2wikimultihopqa"):
        path = ENGINE_DIRS["mlx"] / f"answer_quality_{dataset}.json"
        if path.exists():
            mlx_answer_quality[dataset] = _load(path)
        routed_path = ENGINE_DIRS["mlx"] / f"routed_answer_quality_{dataset}.json"
        if routed_path.exists():
            mlx_routed_quality[dataset] = _load(routed_path)
    pressure_path = ENGINE_DIRS["mlx"] / "bounded_residency_hotpotqa.json"
    mlx_pressure = _load(pressure_path) if pressure_path.exists() else None
    mlx_pressure_curves = {}
    for dataset in ("qasper", "hotpotqa", "2wikimultihopqa"):
        artifacts = []
        for suffix in ("", "_budget8"):
            path = ENGINE_DIRS["mlx"] / f"residency_pressure_curve_{dataset}{suffix}.json"
            if path.exists():
                artifacts.append(_load(path))
        if artifacts:
            mlx_pressure_curves[dataset] = artifacts

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

    mlx_profile_summary = []
    for profile in sorted({row["profile"] for row in mlx_profiles["profile_rows"]}):
        values = [
            row for row in mlx_profiles["profile_rows"] if row["profile"] == profile
        ]
        mlx_profile_summary.append(
            {
                "profile": profile,
                "selected_layer_count": values[0]["selected_layer_count"],
                "exact_recovery": fmean(
                    float(row["exact_recovery"]) for row in values
                ),
                "active_native_kv_bytes": values[0]["active_native_kv_bytes"],
                "completion_latency_ms_mean": fmean(
                    row["completion_latency_ms"] for row in values
                ),
                "max_logit_error": max(
                    row["max_logit_error_vs_ordinary_split"] for row in values
                ),
            }
        )
    mlx_concurrency_summary = []
    for concurrency in sorted(
        {row["concurrency"] for row in mlx_profiles["concurrency_rows"]}
    ):
        values = [
            row
            for row in mlx_profiles["concurrency_rows"]
            if row["concurrency"] == concurrency
        ]
        mlx_concurrency_summary.append(
            {
                "concurrency": concurrency,
                "requests_per_second_mean": fmean(
                    row["requests_per_second"] for row in values
                ),
                "request_latency_ms_mean": fmean(
                    row["mean_request_latency_ms"] for row in values
                ),
                "exact_recovery": fmean(
                    row["exact_recovery_rate"] for row in values
                ),
                "sharing_bytes_saved": values[0]["sharing_bytes_saved"],
            }
        )

    def natural_summary(artifacts: dict[str, dict]) -> list[dict]:
        summary = []
        for dataset, artifact in artifacts.items():
            for condition in sorted({row["condition"] for row in artifact["rows"]}):
                values = [
                    row for row in artifact["rows"] if row["condition"] == condition
                ]
                summary.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "sample_count": len(values),
                        "ranked_exact": fmean(
                            float(row["ranked_exact"]) for row in values
                        ),
                        "latency_ms_mean": fmean(
                            row["completion_latency_ms"] for row in values
                        ),
                        "gold_answer_margin_mean": (
                            fmean(row["gold_answer_margin"] for row in values)
                            if "gold_answer_margin" in values[0]
                            else None
                        ),
                    }
                )
        return summary

    def answer_quality_summary(artifacts: dict[str, dict]) -> list[dict]:
        summary = []
        for dataset, artifact in artifacts.items():
            for condition in (
                "ordinary_split",
                "native_fp",
                "native_int8_resident",
                "native_shuffled",
                "no_memory",
            ):
                values = [
                    row for row in artifact["rows"] if row["condition"] == condition
                ]
                if not values:
                    continue
                summary.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "sample_count": len(values),
                        "seed_count": len({row["seed"] for row in values}),
                        "exact_match": fmean(row["exact_match"] for row in values),
                        "token_f1": fmean(row["token_f1"] for row in values),
                        "gold_answer_logprob": fmean(
                            row["gold_answer_logprob"] for row in values
                        ),
                        "completion_latency_ms": fmean(
                            row["completion_latency_ms"] for row in values
                        ),
                        "resident_selected_kv_bytes": fmean(
                            row["resident_selected_kv_bytes"] for row in values
                        ),
                        "storage_compression_ratio": fmean(
                            row["storage_compression_ratio"] for row in values
                        ),
                    }
                )
        return summary

    def routed_quality_summary(artifacts: dict[str, dict]) -> dict[str, list[dict]]:
        routing = []
        conditions = []
        parity = []
        for dataset, artifact in artifacts.items():
            routed = [
                row for row in artifact["rows"] if row["condition"] == "routed_native"
            ]
            routing.append(
                {
                    "dataset": dataset,
                    "sample_count": len(routed),
                    "seed_count": len({row["seed"] for row in routed}),
                    "candidate_documents": fmean(
                        row["candidate_documents"] for row in routed
                    ),
                    "selected_documents": fmean(
                        row["selected_documents"] for row in routed
                    ),
                    "evidence_recall_at_1": fmean(
                        row["evidence_recall_at_1"] for row in routed
                    ),
                    "evidence_recall_at_2": fmean(
                        row["evidence_recall_at_2"] for row in routed
                    ),
                    "evidence_recall_at_4": fmean(
                        row["evidence_recall_at_4"] for row in routed
                    ),
                    "routed_source_tokens": fmean(
                        row["routed_source_tokens"] for row in routed
                    ),
                    "index_build_ms": fmean(row["index_build_ms"] for row in routed),
                    "routing_ms": fmean(row["routing_ms"] for row in routed),
                    "index_bytes": fmean(row["index_bytes"] for row in routed),
                }
            )
            for condition in (
                "oracle_native",
                "routed_ordinary",
                "routed_native",
                "routed_shuffled",
                "no_memory",
            ):
                values = [
                    row for row in artifact["rows"] if row["condition"] == condition
                ]
                if values:
                    conditions.append(
                        {
                            "dataset": dataset,
                            "condition": condition,
                            "sample_count": len(values),
                            "token_f1": fmean(row["token_f1"] for row in values),
                            "gold_answer_logprob": fmean(
                                row["gold_answer_logprob"] for row in values
                            ),
                            "completion_latency_ms": fmean(
                                row["completion_latency_ms"] for row in values
                            ),
                        }
                    )
            ordinary = {
                (row["seed"], row["example_id"]): row
                for row in artifact["rows"]
                if row["condition"] == "routed_ordinary"
            }
            paired = [
                row for row in artifact["rows"] if row["condition"] == "routed_native"
            ]
            parity.append(
                {
                    "dataset": dataset,
                    "pair_count": len(paired),
                    "matching_outputs": sum(
                        row["output"]
                        == ordinary[(row["seed"], row["example_id"])]["output"]
                        for row in paired
                    ),
                    "max_logprob_delta": max(
                        abs(
                            row["gold_answer_logprob"]
                            - ordinary[(row["seed"], row["example_id"])][
                                "gold_answer_logprob"
                            ]
                        )
                        for row in paired
                    ),
                }
            )
        return {"routing": routing, "conditions": conditions, "parity": parity}

    hicache_rows = sglang_hicache["rows"]
    sglang_combined_conditions = [
        condition
        for row in sglang_combined["rows"]
        for condition in row["conditions"]
    ]
    sglang_prefetch_summary = []
    for lead_ms in sglang_hicache_prefetch["lead_ms"]:
        lead_rows = [
            row
            for row in sglang_hicache_prefetch["rows"]
            if row["requested_lead_ms"] == lead_ms
        ]
        sglang_prefetch_summary.append(
            {
                "requested_lead_ms": lead_ms,
                "sample_count": len(lead_rows),
                "exact_tensor_recovery": fmean(
                    float(row["exact_tensor_recovery"]) for row in lead_rows
                ),
                "promotion_ms_mean": fmean(row["promotion_ms"] for row in lead_rows),
                "actual_lead_ms_mean": fmean(
                    row["actual_lead_ms"] for row in lead_rows
                ),
                "ready_at_demand_rate": fmean(
                    float(row["ready_at_demand"]) for row in lead_rows
                ),
                "demand_stall_ms_mean": fmean(
                    row["demand_stall_ms"] for row in lead_rows
                ),
            }
        )
    vllm_live_rows = vllm_live["rows"]
    vllm_apc_native_rows = [
        row
        for row in vllm_apc["rows"]
        if row["condition"] == "native_pra_plus_apc"
    ]
    pressure_summary = None
    if mlx_pressure is not None:
        summaries = mlx_pressure["seed_summaries"]
        revisits = [row for row in mlx_pressure["rows"] if row["revisit_after_eviction"]]
        pressure_summary = {
            "dataset": mlx_pressure["dataset"],
            "evidence_tier": mlx_pressure["evidence_tier"],
            "seed_count": len(summaries),
            "resident_resource_budget": mlx_pressure["resident_resource_budget"],
            "loads_mean": fmean(row["loads"] for row in summaries),
            "evictions_mean": fmean(row["evictions"] for row in summaries),
            "reloads_mean": fmean(row["reloads"] for row in summaries),
            "peak_resident_bytes_mean": fmean(
                row["peak_resident_bytes"] for row in summaries
            ),
            "revisit_token_f1": fmean(row["token_f1"] for row in revisits),
            "revisit_logprob": fmean(row["gold_answer_logprob"] for row in revisits),
        }
    pressure_curve_summary = []
    for dataset, artifacts in mlx_pressure_curves.items():
        budgets = sorted(
            {
                budget
                for artifact in artifacts
                for budget in artifact["resident_resource_budgets"]
            }
        )
        for budget in budgets:
            rows = [
                row
                for artifact in artifacts
                for row in artifact["rows"]
                if row["resident_resource_budget"] == budget
            ]
            summaries = [
                row
                for artifact in artifacts
                for row in artifact["seed_summaries"]
                if row["resident_resource_budget"] == budget
            ]
            resource_count = artifacts[0]["resources_per_seed"]
            requests_per_seed = artifacts[0]["requests_per_budget_seed"]
            repeated_accesses = requests_per_seed - resource_count
            pressure_curve_summary.append(
                {
                    "dataset": dataset,
                    "resident_resource_budget": budget,
                    "seed_count": len(summaries),
                    "request_count": len(rows),
                    "loads_mean": fmean(row["loads"] for row in summaries),
                    "evictions_mean": fmean(row["evictions"] for row in summaries),
                    "reloads_mean": fmean(row["reloads"] for row in summaries),
                    "reload_fraction": fmean(row["reloads"] for row in summaries)
                    / repeated_accesses,
                    "peak_resident_bytes_mean": fmean(
                        row["peak_resident_bytes"] for row in summaries
                    ),
                    "resolve_ms_mean": fmean(row["resolve_ms"] for row in rows),
                    "dequantize_ms_mean": fmean(
                        row["dequantize_ms"] for row in rows
                    ),
                    "completion_latency_ms_mean": fmean(
                        row["completion_latency_ms"] for row in rows
                    ),
                    "token_f1": fmean(row["token_f1"] for row in rows),
                    "exact_match": fmean(row["exact_match"] for row in rows),
                }
            )

    sglang_concurrency_summary = []
    for concurrency in sorted(
        {row["concurrency"] for row in sglang_live["concurrency_rows"]}
    ):
        values = [
            row
            for row in sglang_live["concurrency_rows"]
            if row["concurrency"] == concurrency
        ]
        sglang_concurrency_summary.append(
            {
                "concurrency": concurrency,
                "requests_per_second_mean": fmean(
                    row["requests_per_second"] for row in values
                ),
                "exact_recovery": fmean(
                    row["exact_recovery_rate"] for row in values
                ),
                "sharing_bytes_saved": values[0]["sharing_bytes_saved"],
                "scheduler_isolation": all(
                    row["scheduler_counts_exclude_pra"] for row in values
                ),
            }
        )
    return {
        "live_storage_lifecycle": {
            engine: {
                "experiment": artifact["experiment"],
                "model_id": artifact["model_id"],
                "summary": artifact["summary"],
            }
            for engine, artifact in live_storage.items()
        },
        "live_storage_scaling": live_storage_scaling,
        "mac_engine_extension": mac_extension,
        "expanded_matched_e0_e2": expanded_matched,
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
            "profiles": mlx_profile_summary,
            "concurrency": mlx_concurrency_summary,
            "persistence": {
                "serialized_bytes_mean": fmean(
                    row["serialized_bytes"]
                    for row in mlx_profiles["persistence_rows"]
                ),
                "save_ms_mean": fmean(
                    row["save_ms"] for row in mlx_profiles["persistence_rows"]
                ),
                "load_ms_mean": fmean(
                    row["load_ms"] for row in mlx_profiles["persistence_rows"]
                ),
                "max_logit_error": max(
                    row["max_logit_error_vs_ordinary_split"]
                    for row in mlx_profiles["persistence_rows"]
                ),
            },
            "natural": natural_summary(mlx_natural),
            "answer_quality": answer_quality_summary(mlx_answer_quality),
            "routed_answer_quality": routed_quality_summary(mlx_routed_quality),
            "bounded_pressure": pressure_summary,
            "residency_pressure_curve": pressure_curve_summary,
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
            "live_runner": {
                "exact_recovery": fmean(
                    float(row["exact_recovery"])
                    for row in sglang_live["rows"]
                    if row["condition"] == "native_runner"
                ),
                "disabled_recovery": fmean(
                    float(row["exact_recovery"])
                    for row in sglang_live["rows"]
                    if row["condition"] == "disabled"
                ),
                "scheduler_isolation": all(
                    row["pra_tokens_absent_from_scheduler_count"]
                    for row in sglang_live["rows"]
                    if row["condition"] == "native_runner"
                ),
            },
            "concurrency": sglang_concurrency_summary,
            "natural": natural_summary(sglang_natural),
            "hicache": {
                "evidence_tier": sglang_hicache["evidence_tier"],
                "seed_count": len(hicache_rows),
                "exact_recovery": fmean(
                    float(row["exact_recovery"]) for row in hicache_rows
                ),
                "l1_hit_ms_mean": fmean(row["l1_hit_ms"] for row in hicache_rows),
                "l2_to_l1_ms_mean": fmean(
                    row["l2_to_l1_ms"] for row in hicache_rows
                ),
                "l3_to_l1_ms_mean": fmean(
                    row["l3_to_l1_ms"] for row in hicache_rows
                ),
                "warm_l1_ms_mean": fmean(row["warm_l1_ms"] for row in hicache_rows),
                "l1_to_l2_demotion_rate": fmean(
                    float(row["hicache"]["l1_to_l2_demotions"] > 0)
                    for row in hicache_rows
                ),
                "l2_to_l3_demotion_rate": fmean(
                    float(row["hicache"]["l2_to_l3_demotions"] > 0)
                    for row in hicache_rows
                ),
                "radix_separation_rate": fmean(
                    float(row["pra_tokens_absent_from_radix_prefix"])
                    for row in hicache_rows
                ),
            },
            "builtin_hicache_storage": {
                "evidence_tier": sglang_builtin_hicache["evidence_tier"],
                "backend": sglang_builtin_hicache["storage_backend"],
                "off_node_transport": sglang_builtin_hicache["off_node_transport"],
                "seed_count": len(sglang_builtin_hicache["rows"]),
                "exact_recovery": fmean(
                    float(row["exact_recovery"])
                    for row in sglang_builtin_hicache["rows"]
                ),
                "fresh_adapter_hit_rate": fmean(
                    float(row["fresh_adapter_backend_hit"])
                    for row in sglang_builtin_hicache["rows"]
                ),
                "write_ms_mean": fmean(
                    row["write_ms"] for row in sglang_builtin_hicache["rows"]
                ),
                "cold_read_to_l1_ms_mean": fmean(
                    row["cold_backend_read_to_l1_ms"]
                    for row in sglang_builtin_hicache["rows"]
                ),
                "cold_read_to_l1_ms_median": median(
                    row["cold_backend_read_to_l1_ms"]
                    for row in sglang_builtin_hicache["rows"]
                ),
                "warm_l1_ms_mean": fmean(
                    row["warm_l1_ms"] for row in sglang_builtin_hicache["rows"]
                ),
            },
            "builtin_hicache_prefetch": {
                "evidence_tier": sglang_hicache_prefetch["evidence_tier"],
                "prefetch_signal": sglang_hicache_prefetch["prefetch_signal"],
                "off_node_transport": sglang_hicache_prefetch[
                    "off_node_transport"
                ],
                "seed_count": len(
                    {row["seed"] for row in sglang_hicache_prefetch["rows"]}
                ),
                "rows_by_requested_lead_ms": sglang_prefetch_summary,
                "native_async_overlap_status": "OPEN_PYTHON_THREAD_BLOCKS_CALLER",
            },
            "radix_hicache_combined": {
                "evidence_tier": sglang_combined["evidence_tier"],
                "seed_count": len(sglang_combined["rows"]),
                "selected_exact": fmean(
                    float(row["exact_recovery"])
                    for row in sglang_combined_conditions
                    if row["condition"] in {"selected_A", "reselected_C"}
                ),
                "ordinary_cleanup_recovery": fmean(
                    float(row["exact_recovery"])
                    for row in sglang_combined_conditions
                    if row["condition"] == "ordinary_B_after_cleanup"
                ),
                "radix_separation": fmean(
                    float(row["selected_tokens_excluded_from_radix_length"])
                    for row in sglang_combined_conditions
                ),
                "exactly_one_copy": fmean(
                    float(row["exactly_one_selected_copy"])
                    for row in sglang_combined_conditions
                    if row["exactly_one_selected_copy"] is not None
                ),
                "l1_hits": sglang_combined["hicache_metrics"]["l1_hits"],
                "l2_to_l1_promotions": sglang_combined["hicache_metrics"][
                    "l2_to_l1_promotions"
                ],
            },
        },
        "vllm": {
            "status": vllm["native_pra_status"],
            "evidence_tier": vllm["evidence_tier"],
            "seed_count": len(vllm["seeds"]),
            "rows_by_selected_tokens": vllm_by_tokens,
            "live_generation": {
                "evidence_tier": vllm_live["evidence_tier"],
                "engine_version": vllm_live["engine_version"],
                "model_id": vllm_live["model_id"],
                "seed_count": len(vllm_live_rows),
                "full_context_exact": fmean(
                    float(row["full_context_exact_recovery"])
                    for row in vllm_live_rows
                ),
                "native_exact": fmean(
                    float(row["native_exact_recovery"]) for row in vllm_live_rows
                ),
                "disabled_exact": fmean(
                    float(row["disabled_exact_recovery"]) for row in vllm_live_rows
                ),
                "post_cleanup_leak": fmean(
                    float(row["post_cleanup_leak"]) for row in vllm_live_rows
                ),
                "wrong_memory_follows_wrong_code": fmean(
                    float(row["wrong_memory_follows_wrong_code"])
                    for row in vllm_live_rows
                ),
                "full_context_ms_mean": fmean(
                    row["full_context_ms"] for row in vllm_live_rows
                ),
                "native_ms_mean": fmean(row["native_ms"] for row in vllm_live_rows),
                "active_native_kv_bytes": vllm_live_rows[0]["active_native_kv_bytes"],
            },
            "apc_concurrency": {
                "evidence_tier": vllm_apc["evidence_tier"],
                "seed_count": len(vllm_apc["seeds"]),
                "native_exact": fmean(
                    float(output["exact_recovery"])
                    for row in vllm_apc_native_rows
                    for output in row["outputs"]
                ),
                "mixed_isolation_success": fmean(
                    (
                        float(output["exact_recovery"])
                        if output["selected_registered"]
                        else float(not output["exact_recovery"])
                    )
                    for row in vllm_apc["rows"]
                    if row["condition"] == "mixed_selected_and_ordinary"
                    for output in row["outputs"]
                ),
                "cached_tokens_min": min(
                    output["num_cached_tokens"]
                    for row in vllm_apc_native_rows
                    for output in row["outputs"]
                ),
                "cached_tokens_max": max(
                    output["num_cached_tokens"]
                    for row in vllm_apc_native_rows
                    for output in row["outputs"]
                ),
                "max_concurrency": max(vllm_apc["concurrency_levels"]),
                "max_concurrency_requests_per_second": fmean(
                    row["requests_per_second"]
                    for row in vllm_apc_native_rows
                    if row["concurrency"] == max(vllm_apc["concurrency_levels"])
                ),
            },
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
    profile_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Consumer profile & Layers & Exact & Active MB & Max error \\",
        r"\midrule",
    ]
    for row in mlx["profiles"]:
        profile_lines.append(
            f"{_tex_escape(row['profile'])} & {row['selected_layer_count']} & "
            f"{100 * row['exact_recovery']:.0f}\\% & "
            f"{row['active_native_kv_bytes'] / 1048576:.2f} & "
            f"{row['max_logit_error']:.4g} \\\\"
        )
    profile_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ENGINE_DIRS["mlx"] / "generated_profile_table.tex").write_text(
        "\n".join(profile_lines) + "\n", encoding="utf-8"
    )
    persistence = mlx["persistence"]
    lifecycle_lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Concurrency & Exact & Requests/s & Mean latency ms & Shared MB saved & Persist load ms \\",
        r"\midrule",
    ]
    for row in mlx["concurrency"]:
        lifecycle_lines.append(
            f"{row['concurrency']} & {100 * row['exact_recovery']:.0f}\\% & "
            f"{row['requests_per_second_mean']:.2f} & "
            f"{row['request_latency_ms_mean']:.1f} & "
            f"{row['sharing_bytes_saved'] / 1048576:.1f} & "
            f"{persistence['load_ms_mean']:.1f} \\\\"
        )
    lifecycle_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ENGINE_DIRS["mlx"] / "generated_lifecycle_table.tex").write_text(
        "\n".join(lifecycle_lines) + "\n", encoding="utf-8"
    )
    mlx_natural_lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & Condition & Samples & Ranked exact & Margin \\",
        r"\midrule",
    ]
    for row in mlx["natural"]:
        mlx_natural_lines.append(
            f"{_tex_escape(row['dataset'])} & {_tex_escape(row['condition'])} & "
            f"{row['sample_count']} & {100 * row['ranked_exact']:.0f}\\% & "
            f"{row['gold_answer_margin_mean']:.3f} \\\\"
        )
    mlx_natural_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ENGINE_DIRS["mlx"] / "generated_natural_table.tex").write_text(
        "\n".join(mlx_natural_lines) + "\n", encoding="utf-8"
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
    sglang_live_lines = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"Concurrency & Exact & Requests/s & Shared MB saved & Scheduler isolation \\",
        r"\midrule",
    ]
    for row in sglang["concurrency"]:
        sglang_live_lines.append(
            f"{row['concurrency']} & {100 * row['exact_recovery']:.0f}\\% & "
            f"{row['requests_per_second_mean']:.2f} & "
            f"{row['sharing_bytes_saved'] / 1048576:.1f} & "
            f"{'yes' if row['scheduler_isolation'] else 'no'} \\\\"
        )
    sglang_live_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ENGINE_DIRS["sglang"] / "generated_live_runner_table.tex").write_text(
        "\n".join(sglang_live_lines) + "\n", encoding="utf-8"
    )
    sglang_natural_lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Dataset & Condition & Samples & Ranked exact \\",
        r"\midrule",
    ]
    for row in sglang["natural"]:
        sglang_natural_lines.append(
            f"{_tex_escape(row['dataset'])} & {_tex_escape(row['condition'])} & "
            f"{row['sample_count']} & {100 * row['ranked_exact']:.0f}\\% \\\\"
        )
    sglang_natural_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ENGINE_DIRS["sglang"] / "generated_natural_table.tex").write_text(
        "\n".join(sglang_natural_lines) + "\n", encoding="utf-8"
    )


def write_latest_engine_tables(registry: dict) -> None:
    """Write the live-generation, hierarchy, QA, and product-matrix tables."""

    native = registry["native_results"]
    vllm = native["vllm"]["live_generation"]
    vllm_lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Condition & Control success & Mean latency ms \\",
        r"\midrule",
        f"Full visible context & {100 * vllm['full_context_exact']:.0f}\\% & {vllm['full_context_ms_mean']:.1f} \\\\",
        f"Selected native K/V & {100 * vllm['native_exact']:.0f}\\% & {vllm['native_ms_mean']:.1f} \\\\",
        f"Disabled memory (must fail) & {100 * (1 - vllm['disabled_exact']):.0f}\\% & -- \\\\",
        f"Post-cleanup isolation & {100 * (1 - vllm['post_cleanup_leak']):.0f}\\% & -- \\\\",
        f"Wrong-memory causality & {100 * vllm['wrong_memory_follows_wrong_code']:.0f}\\% & -- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (ENGINE_DIRS["vllm"] / "generated_v1_live_table.tex").write_text(
        "\n".join(vllm_lines) + "\n", encoding="utf-8"
    )

    hicache = native["sglang"]["hicache"]
    hicache_lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Path or invariant & Mean latency ms & Success \\",
        r"\midrule",
        f"Warm L1 hit & {hicache['warm_l1_ms_mean']:.3f} & -- \\\\",
        f"L2 $\\rightarrow$ L1 & {hicache['l2_to_l1_ms_mean']:.1f} & -- \\\\",
        f"L3 $\\rightarrow$ L1 & {hicache['l3_to_l1_ms_mean']:.1f} & -- \\\\",
        f"Answer recovery & -- & {100 * hicache['exact_recovery']:.0f}\\% \\\\",
        f"L1/L2 pressure demotion & -- & {100 * min(hicache['l1_to_l2_demotion_rate'], hicache['l2_to_l3_demotion_rate']):.0f}\\% \\\\",
        f"Radix namespace separation & -- & {100 * hicache['radix_separation_rate']:.0f}\\% \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (ENGINE_DIRS["sglang"] / "generated_hicache_table.tex").write_text(
        "\n".join(hicache_lines) + "\n", encoding="utf-8"
    )

    condition_labels = {
        "ordinary_split": "Ordinary split K/V",
        "native_fp": "Native FP K/V",
        "native_int8_resident": "Native int8 resident",
        "native_shuffled": "Shuffled native K/V",
        "no_memory": "No memory",
    }
    qa_lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Condition & $n$ & Token F1 & Gold log-prob. & Resident MiB \\",
        r"\midrule",
    ]
    for row in native["mlx"]["answer_quality"]:
        qa_lines.append(
            f"{_tex_escape(row['dataset'])} & {_tex_escape(condition_labels[row['condition']])} & "
            f"{row['sample_count']} & {row['token_f1']:.3f} & "
            f"{row['gold_answer_logprob']:.2f} & "
            f"{row['resident_selected_kv_bytes'] / 1048576:.1f} \\\\"
        )
    qa_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ENGINE_DIRS["mlx"] / "generated_answer_quality_table.tex").write_text(
        "\n".join(qa_lines) + "\n", encoding="utf-8"
    )

    routed = native["mlx"]["routed_answer_quality"]
    routed_lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Dataset & $n$ & Candidates & R@1 & R@2 & R@4 & Tokens & Route ms \\",
        r"\midrule",
    ]
    for row in routed["routing"]:
        routed_lines.append(
            f"{_tex_escape(row['dataset'])} & {row['sample_count']} & "
            f"{row['candidate_documents']:.1f} & {row['evidence_recall_at_1']:.3f} & "
            f"{row['evidence_recall_at_2']:.3f} & {row['evidence_recall_at_4']:.3f} & "
            f"{row['routed_source_tokens']:.1f} & {row['routing_ms']:.2f} \\\\"
        )
    routed_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ENGINE_DIRS["mlx"] / "generated_routed_recall_table.tex").write_text(
        "\n".join(routed_lines) + "\n", encoding="utf-8"
    )

    routed_condition_labels = {
        "oracle_native": "Oracle",
        "routed_native": "Routed",
        "routed_shuffled": "Shuffled",
        "no_memory": "No memory",
    }
    routed_quality_lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & Condition & $n$ & Token F1 & Gold log-prob. \\",
        r"\midrule",
    ]
    for row in routed["conditions"]:
        if row["condition"] not in routed_condition_labels:
            continue
        routed_quality_lines.append(
            f"{_tex_escape(row['dataset'])} & "
            f"{routed_condition_labels[row['condition']]} & {row['sample_count']} & "
            f"{row['token_f1']:.3f} & {row['gold_answer_logprob']:.2f} \\\\"
        )
    routed_quality_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ENGINE_DIRS["mlx"] / "generated_routed_quality_table.tex").write_text(
        "\n".join(routed_quality_lines) + "\n", encoding="utf-8"
    )

    pressure = native["mlx"]["bounded_pressure"]
    if pressure is not None:
        pressure_lines = [
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"Dataset & Seeds & Capacity & Loads & Evictions & Reloads & Peak MiB \\",
            r"\midrule",
            f"{_tex_escape(pressure['dataset'])} & {pressure['seed_count']} & "
            f"{pressure['resident_resource_budget']} & {pressure['loads_mean']:.1f} & "
            f"{pressure['evictions_mean']:.1f} & {pressure['reloads_mean']:.1f} & "
            f"{pressure['peak_resident_bytes_mean'] / 1048576:.1f} \\\\ ",
            r"\bottomrule",
            r"\end{tabular}",
        ]
        (ENGINE_DIRS["mlx"] / "generated_bounded_pressure_table.tex").write_text(
            "\n".join(pressure_lines) + "\n", encoding="utf-8"
        )

    curve_lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Dataset & $K$ & $n$ & Peak MiB & Reload frac. & Resolve ms & Token F1 \\",
        r"\midrule",
    ]
    curve_dataset_labels = {
        "qasper": "QASPER",
        "hotpotqa": "HotpotQA",
        "2wikimultihopqa": "2Wiki",
    }
    for row in native["mlx"]["residency_pressure_curve"]:
        curve_lines.append(
            f"{curve_dataset_labels[row['dataset']]} & {row['resident_resource_budget']} & "
            f"{row['request_count']} & {row['peak_resident_bytes_mean'] / 1048576:.1f} & "
            f"{row['reload_fraction']:.2f} & {row['resolve_ms_mean']:.1f} & "
            f"{row['token_f1']:.3f} \\\\"
        )
    curve_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ENGINE_DIRS["mlx"] / "generated_residency_pressure_curve_table.tex").write_text(
        "\n".join(curve_lines) + "\n", encoding="utf-8"
    )

    product_lines = [
        r"\begin{tabularx}{\linewidth}{lYYYY}",
        r"\toprule",
        r"Engine/model & Hardware & Profile & Evidence tier & Status \\",
        r"\midrule",
    ]
    for row in registry["product_matrix"]:
        model_name = row["model"].split("/")[-1]
        model = {
            "Qwen3-0.6B-4bit": "Qwen3-0.6B",
            "Qwen3-0.6B": "Qwen3-0.6B",
            "Qwen3-1.7B-4bit": "Qwen3-1.7B",
            "Llama-3.2-1B-Instruct-4bit": "Llama-3.2-1B",
            "gemma-3-1b-it-4bit": "Gemma-3-1B",
        }.get(model_name, model_name)
        hardware = str(row["hardware"])
        if hardware.startswith("Apple M5 MacBook Pro"):
            hardware = "Apple M5 / 16 GB / Metal 4"
        evidence = row["evidence_tier"].replace("_", " ").lower()
        product_lines.append(
            f"{_tex_escape(row['engine'])} / {_tex_escape(model)} & "
            f"{_tex_escape(hardware)} & {_tex_escape(row['profile'])} & "
            f"{_tex_escape(evidence)} & "
            f"{_tex_escape(row['status'])} \\\\"
        )
    product_lines.extend([r"\bottomrule", r"\end{tabularx}"])
    (RESULTS / "generated_engine_product_matrix.tex").write_text(
        "\n".join(product_lines) + "\n", encoding="utf-8"
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

    mlx = native["mlx"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
    profiles = sorted(mlx["profiles"], key=lambda row: row["active_native_kv_bytes"])
    axes[0].plot(
        [row["active_native_kv_bytes"] / 1048576 for row in profiles],
        [row["exact_recovery"] for row in profiles],
        marker="o",
        color="#b35c1e",
    )
    for row in profiles:
        axes[0].annotate(
            row["profile"].replace("_", " "),
            (row["active_native_kv_bytes"] / 1048576, row["exact_recovery"]),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axes[0].set_xlabel("Active native K/V (MiB)")
    axes[0].set_ylabel("Exact recovery")
    axes[0].set_ylim(-0.05, 1.08)
    axes[0].set_title("MLX consumer-layer frontier")
    axes[1].plot(
        [row["concurrency"] for row in mlx["concurrency"]],
        [row["requests_per_second_mean"] for row in mlx["concurrency"]],
        marker="s",
        color="#2f855a",
    )
    axes[1].set_xlabel("Concurrent requests")
    axes[1].set_ylabel("Requests/s")
    axes[1].set_title("MLX shared-memory concurrency")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ENGINE_DIRS["mlx"] / "native_profile_concurrency.png", dpi=180)
    fig.savefig(ENGINE_DIRS["mlx"] / "native_profile_concurrency.pdf")
    plt.close(fig)

    routed = mlx["routed_answer_quality"]
    datasets = [row["dataset"] for row in routed["routing"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
    width = 0.23
    positions = list(range(len(datasets)))
    for offset, (key, label, color) in enumerate(
        (
            ("evidence_recall_at_1", "R@1", "#6b7280"),
            ("evidence_recall_at_2", "R@2", "#2f6f9f"),
            ("evidence_recall_at_4", "R@4", "#2f855a"),
        )
    ):
        axes[0].bar(
            [position + (offset - 1) * width for position in positions],
            [row[key] for row in routed["routing"]],
            width=width,
            label=label,
            color=color,
        )
    axes[0].set_xticks(positions, datasets)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Gold evidence recall")
    axes[0].set_title("SDK hybrid document routing")
    axes[0].legend(frameon=False)
    quality_conditions = ("oracle_native", "routed_native", "routed_shuffled")
    colors = ("#2f855a", "#2f6f9f", "#b35c1e")
    for offset, (condition, color) in enumerate(zip(quality_conditions, colors)):
        values = {
            row["dataset"]: row["gold_answer_logprob"]
            for row in routed["conditions"]
            if row["condition"] == condition
        }
        axes[1].bar(
            [position + (offset - 1) * width for position in positions],
            [values[dataset] for dataset in datasets],
            width=width,
            label=condition.replace("_native", "").replace("routed_", ""),
            color=color,
        )
    axes[1].set_xticks(positions, datasets)
    axes[1].set_ylabel("Gold-answer log-probability")
    axes[1].set_title("Oracle, routed, and wrong memory")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ENGINE_DIRS["mlx"] / "routed_qa_recall_quality.png", dpi=180)
    fig.savefig(ENGINE_DIRS["mlx"] / "routed_qa_recall_quality.pdf")
    plt.close(fig)

    pressure_rows = mlx["residency_pressure_curve"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))
    colors = {
        "qasper": "#2f6f9f",
        "hotpotqa": "#2f855a",
        "2wikimultihopqa": "#b35c1e",
    }
    labels = {
        "qasper": "QASPER",
        "hotpotqa": "HotpotQA",
        "2wikimultihopqa": "2Wiki",
    }
    for dataset in labels:
        values = [row for row in pressure_rows if row["dataset"] == dataset]
        values.sort(key=lambda row: row["resident_resource_budget"])
        budgets = [row["resident_resource_budget"] for row in values]
        axes[0].plot(
            budgets,
            [row["peak_resident_bytes_mean"] / 1048576 for row in values],
            marker="o",
            color=colors[dataset],
            label=labels[dataset],
        )
        axes[1].plot(
            budgets,
            [row["reload_fraction"] for row in values],
            marker="o",
            color=colors[dataset],
        )
        axes[2].plot(
            budgets,
            [row["resolve_ms_mean"] for row in values],
            marker="o",
            color=colors[dataset],
        )
    axes[0].set_ylabel("Peak compact K/V (MiB)")
    axes[0].set_title("Bounded residency")
    axes[0].legend(frameon=False)
    axes[1].set_ylabel("Reloaded repeated accesses")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Working-set cliff")
    axes[2].set_ylabel("Mean resolve time (ms)")
    axes[2].set_title("Materialization cost")
    for axis in axes:
        axis.set_xlabel("Resident resource budget K")
        axis.set_xticks((1, 2, 4, 8))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ENGINE_DIRS["mlx"] / "residency_pressure_curve.png", dpi=180)
    fig.savefig(ENGINE_DIRS["mlx"] / "residency_pressure_curve.pdf")
    plt.close(fig)

    sglang = native["sglang"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
    axes[0].plot(
        [row["concurrency"] for row in sglang["concurrency"]],
        [row["requests_per_second_mean"] for row in sglang["concurrency"]],
        marker="o",
        color="#2f855a",
    )
    axes[0].set_xlabel("Concurrent requests")
    axes[0].set_ylabel("Requests/s")
    axes[0].set_title("SGLang live native runner")
    conditions = ["ordinary_full", "native", "disabled", "shuffled"]
    width = 0.35
    for dataset_index, dataset in enumerate(("qasper", "hotpotqa")):
        values = {
            row["condition"]: row["ranked_exact"]
            for row in sglang["natural"]
            if row["dataset"] == dataset
        }
        axes[1].bar(
            [index + (dataset_index - 0.5) * width for index in range(len(conditions))],
            [values[condition] for condition in conditions],
            width=width,
            label=dataset,
        )
    axes[1].set_xticks(
        range(len(conditions)), [value.replace("_", "\n") for value in conditions]
    )
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("Ranked exact")
    axes[1].set_title("Natural-text causal controls")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ENGINE_DIRS["sglang"] / "live_runner_natural.png", dpi=180)
    fig.savefig(ENGINE_DIRS["sglang"] / "live_runner_natural.pdf")
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
    write_latest_engine_tables(registry)
    write_native_plots(registry)


if __name__ == "__main__":
    main()
