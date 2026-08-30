"""Build next-iteration decision artifacts from measured engine runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Iterable, Mapping


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(rows: Iterable[Mapping[str, object]], field: str) -> float:
    return fmean(float(row[field]) for row in rows)


def _percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot summarize an empty cohort.")
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _tex(value: object) -> str:
    return str(value).replace("_", r"\_")


def _mlx(root: Path) -> dict[str, object]:
    profiles = _load(root / "paper6_2_mlx/profiles_persistence_concurrency.json")
    quant = _load(root / "paper6_2_mlx/selective_kv_quantization_qasper.json")
    pressure = _load(root / "paper6_2_mlx/tier_window_pressure_qasper.json")
    concurrency = _load(root / "paper6_2_mlx/live_storage_concurrency_qasper.json")
    lifecycle = _load(
        root / "paper6_2_mlx/live_storage_lifecycle_scale_qwen3_0_6b_qasper.json"
    )
    segmented = _load(root / "paper6_2_mlx/segmented_attention_profile.json")

    profile_rows = []
    for name in sorted({str(row["profile"]) for row in profiles["profile_rows"]}):
        rows = [row for row in profiles["profile_rows"] if row["profile"] == name]
        profile_rows.append(
            {
                "profile": name,
                "selected_layers": int(rows[0]["selected_layer_count"]),
                "active_native_kv_bytes": _mean(rows, "active_native_kv_bytes"),
                "completion_latency_ms": _mean(rows, "completion_latency_ms"),
                "argmax_parity_rate": fmean(
                    bool(row["argmax_matches_ordinary_split"]) for row in rows
                ),
                "exact_recovery_rate": fmean(bool(row["exact_recovery"]) for row in rows),
                "mean_logit_error": _mean(rows, "mean_logit_error_vs_ordinary_split"),
                "evidence_tier": "CONTROLLED_MODEL_BACKED",
            }
        )

    pressure_rows = list(pressure["rows"])
    source = [row for row in pressure_rows if row["tier_before"] == "source"]
    warm = [row for row in pressure_rows if row["tier_before"] == "warm"]
    hot = [row for row in pressure_rows if row["tier_before"] == "hot"]
    persist_costs = [
        float(row["background_transition_latency_ms"]["warm"])
        for row in lifecycle["rows"]
    ]
    source_ms = _mean(source, "resolve_ms")
    warm_ms = _mean(warm, "resolve_ms")
    saved_ms = max(0.0, source_ms - warm_ms)
    persist_ms = fmean(persist_costs)
    break_even = None if saved_ms <= 0 else math.ceil(persist_ms / saved_ms)

    quant_rows = []
    for name, row in quant["summary"].items():
        quant_rows.append({"profile": name, **dict(row)})

    concurrent_rows = list(concurrency["rows"])
    return {
        "schema_version": "1.0",
        "experiment": "paper6_2_mlx_next_iter_decision_v1",
        "evidence_tier": "SYNTHESIS_OF_MODEL_BACKED_RUNS",
        "consumer_profiles": profile_rows,
        "source_warm_admission": {
            "source_resolve_mean_ms": source_ms,
            "warm_resolve_mean_ms": warm_ms,
            "hot_resolve_mean_ms": _mean(hot, "resolve_ms"),
            "persist_mean_ms": persist_ms,
            "saved_per_reuse_ms": saved_ms,
            "break_even_reuses": break_even,
            "rule": "persist if expected_reuse * (source_ms - warm_ms) > persist_ms",
            "scope": "derived across the measured QASPER pressure and lifecycle cohorts",
        },
        "quantization": quant_rows,
        "concurrency": concurrent_rows,
        "segmented_attention": {
            "evidence_tier": segmented["evidence_tier"],
            "one_attention_normalization": segmented["one_attention_normalization"],
            "model_runner_fused": segmented["model_runner_fused"],
            "rows": list(segmented["rows"]),
            "status": "PRIMITIVE_MEASURED_MODEL_ATTENTION_PATCH_PENDING",
        },
        "staged_pareto_status": {
            "measured_local_windows": list(pressure["local_kv_sizes"]),
            "required_followup_windows": [2048, 8192, 32768, "full"],
            "measured_hot_resources": list(pressure["hot_resource_budgets"]),
            "measured_warm_resources": list(pressure["warm_resource_budgets"]),
            "measured_concurrency": list(concurrency["concurrency"]),
            "status": "PARTIAL_STAGED_FRONTIER",
            "reason": "large-window occupancy requires long local histories; it is not inferred from max_kv_size alone",
        },
        "profile_decision": {
            "reference_correctness": "MEASURED",
            "quality_max_candidate": "CANDIDATE_SMOKE",
            "balanced": "CANDIDATE_CALIBRATION_PENDING",
            "economy": "CANDIDATE_CALIBRATION_PENDING",
        },
    }


def _sglang(root: Path, platform: Mapping[str, object]) -> dict[str, object]:
    prefetch = _load(root / "paper6_1_sglang/builtin_hicache_prefetch.json")
    combined = _load(root / "paper6_1_sglang/radix_hicache_combined.json")
    online = _load(root / "paper6_1_sglang/online_native_gateway_qasper.json")
    prefetch_rows = []
    for lead in sorted({int(row["lead_ms"]) for row in prefetch["rows"]}):
        rows = [row for row in prefetch["rows"] if int(row["lead_ms"]) == lead]
        prefetch_rows.append(
            {
                "lead_ms": lead,
                "ready_rate": fmean(bool(row["ready_at_demand"]) for row in rows),
                "demand_stall_p50_ms": _percentile(
                    (float(row["demand_stall_ms"]) for row in rows), 0.50
                ),
                "demand_stall_p95_ms": _percentile(
                    (float(row["demand_stall_ms"]) for row in rows), 0.95
                ),
                "exact_tensor_rate": fmean(
                    bool(row["exact_tensor_recovery"]) for row in rows
                ),
            }
        )
    conditions = [condition for row in combined["rows"] for condition in row["conditions"]]
    return {
        "schema_version": "1.0",
        "experiment": "paper6_1_sglang_next_iter_decision_v1",
        "evidence_tier": "SYNTHESIS_OF_MODEL_BACKED_RUNS",
        "off_node_gate": platform["interpretation"]["off_node"],
        "off_node_backends": platform["gates"]["sglang_off_node_backends"],
        "off_node_claim_allowed": False,
        "prefetch": prefetch_rows,
        "radix_pra": {
            "examples": len(combined["rows"]),
            "selected_exact_rate": fmean(
                bool(row["exact_recovery"])
                for row in conditions
                if row["condition"] in {"selected_A", "reselected_C"}
            ),
            "ordinary_after_cleanup_exact_rate": fmean(
                bool(row["exact_recovery"])
                for row in conditions
                if row["condition"] == "ordinary_B_after_cleanup"
            ),
            "exactly_one_copy_rate": fmean(
                bool(row["exactly_one_selected_copy"])
                for row in conditions
                if row["exactly_one_selected_copy"] is not None
            ),
        },
        "online_concurrency": list(online["concurrency_rows"]),
        "concurrency_execution": online["concurrency_execution"],
        "selected_cache_fusion": {
            "status": "PROFILED_NOT_FUSED",
            "reason": "current MLX backend consumes one dense cache view; a segmented one-softmax kernel requires an engine attention patch",
        },
    }


def _vllm(root: Path, platform: Mapping[str, object]) -> dict[str, object]:
    apc = _load(root / "paper6_vllm/v1_apc_concurrency.json")
    rows = [row for row in apc["rows"] if row["condition"] == "native_pra_plus_apc"]
    concurrency_rows = []
    for concurrency in apc["concurrency_levels"]:
        selected = [row for row in rows if int(row["concurrency"]) == int(concurrency)]
        outputs = [output for row in selected for output in row["outputs"]]
        concurrency_rows.append(
            {
                "concurrency": int(concurrency),
                "requests_per_second": _mean(selected, "requests_per_second"),
                "wave_p95_ms": _percentile(
                    (float(row["elapsed_ms"]) for row in selected), 0.95
                ),
                "exact_recovery_rate": fmean(
                    bool(output["exact_recovery"]) for output in outputs
                ),
                "apc_hit_rate": fmean(
                    int(output["num_cached_tokens"]) > 0 for output in outputs
                ),
            }
        )
    return {
        "schema_version": "1.0",
        "experiment": "paper6_vllm_next_iter_decision_v1",
        "evidence_tier": "VLLM_METAL_MODEL_BACKED",
        "apc_pra_concurrency": concurrency_rows,
        "cuda_gate": platform["interpretation"]["cuda_vllm"],
        "lmcache_gate": platform["interpretation"]["lmcache"],
        "production_cuda_claim_allowed": False,
        "implemented_on_metal": True,
        "unresolved_production_metrics": [
            "CUDA TTFT/ITL tails",
            "HBM telemetry",
            "PCIe or NVLink transfer",
            "LMCache connector",
            "tensor parallel",
        ],
    }


def _write_tex(path: Path, payload: Mapping[str, object]) -> None:
    mlx = payload["mlx"]
    sglang = payload["sglang"]
    vllm = payload["vllm"]
    admission = mlx["source_warm_admission"]
    lines = [
        f"\\newcommand{{\\MLXWarmBreakEven}}{{{admission['break_even_reuses']}}}",
        f"\\newcommand{{\\MLXSourceResolveMean}}{{{admission['source_resolve_mean_ms']:.1f}}}",
        f"\\newcommand{{\\MLXWarmResolveMean}}{{{admission['warm_resolve_mean_ms']:.1f}}}",
        f"\\newcommand{{\\SGLangOffNodeGate}}{{{_tex(sglang['off_node_gate'])}}}",
        f"\\newcommand{{\\VLLMCudaGate}}{{{_tex(vllm['cuda_gate'])}}}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root", type=Path, default=Path("docs/papers/shared/results")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/papers/shared/results/engine_next_iter_summary.json"),
    )
    args = parser.parse_args()
    root = args.results_root
    mac = _load(root / "engine_platform_gate_mac.json")
    mac_vllm = _load(root / "engine_platform_gate_mac_vllm.json")
    payload = {
        "schema_version": "1.0",
        "experiment": "pra_engine_next_iteration_synthesis_v1",
        "mlx": _mlx(root),
        "sglang": _sglang(root, mac),
        "vllm": _vllm(root, mac_vllm),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_tex(args.output.with_name("generated_engine_next_iter.tex"), payload)
    print(json.dumps({
        "mlx_break_even_reuses": payload["mlx"]["source_warm_admission"]["break_even_reuses"],
        "sglang_off_node_gate": payload["sglang"]["off_node_gate"],
        "vllm_cuda_gate": payload["vllm"]["cuda_gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
