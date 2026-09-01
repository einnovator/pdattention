"""Normalize existing evidence into the cross-engine product matrix v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any, Mapping

from pra_hf.product_matrix import ProductMatrix, ProductMatrixRow, optional_number


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/papers/shared/results"
DEFAULT_OUTPUT = RESULTS / "pra_product_matrix_v2.json"
DEFAULT_TABLE = RESULTS / "generated_pra_product_matrix_v2.tex"


def _slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-") or "unknown"


def _integer(value: object) -> int | None:
    number = optional_number(value)
    return None if number is None else int(number)


def _status(value: object, *, profile: bool = False) -> str:
    normalized = str(value or "NOT_MEASURED").upper()
    if profile and normalized in {"CANDIDATE", "SMOKE"}:
        return "CALIBRATION_PENDING"
    allowed = {
        "MEASURED", "CONTROLLED", "MODEL_BACKED", "NATURAL_WORKLOAD",
        "CANDIDATE", "CALIBRATION_PENDING", "RESEARCH_ONLY", "BLOCKED",
        "NOT_MEASURED", "NOT_APPLICABLE",
    }
    return normalized if normalized in allowed else "NOT_MEASURED"


def _evidence_status(row: Mapping[str, Any]) -> str:
    tier = str(row.get("evidence_tier") or "").upper()
    if any(word in tier for word in ("NATURAL", "HELD_OUT")):
        return "NATURAL_WORKLOAD"
    if any(word in tier for word in ("MODEL", "LIVE")):
        return "MODEL_BACKED"
    if tier in {"SMOKE", "CONTROLLED", "DIAGNOSTIC"}:
        return "CONTROLLED"
    return _status(row.get("measurement_status"))


def _profile_row(row: Mapping[str, Any], index: int) -> ProductMatrixRow:
    active_kv = optional_number(row.get("active_kv_tokens"))
    provenance = str(row.get("artifact_path") or "paper4_5 profile registry")
    return ProductMatrixRow(
        row_id=f"profile-{index:03d}-{_slug(row['model_id'])}-{_slug(row['profile'])}",
        model_family=str(row["model_family"]),
        model_id=str(row["model_id"]),
        model_revision=row.get("model_revision"),
        model_size=_integer(row.get("parameter_count")),
        model_variant=str(row.get("dtype")) if row.get("dtype") else None,
        engine=str(row["engine"]),
        engine_version=row.get("engine_version"),
        hardware=str(row.get("hardware") or "unknown"),
        profile=str(row["profile"]),
        profile_status=_status(row.get("profile_status"), profile=True),
        workload=str(row.get("workload") or "unknown"),
        dataset=str(row.get("dataset") or "unknown"),
        quality_metric=str(row.get("quality_metric") or "unknown"),
        integration_level="E2" if active_kv is not None else "E0",
        representation="E2_HOT" if active_kv is not None else "E0_SELECTED",
        quantization=str(row.get("dtype")) if row.get("dtype") else None,
        accelerator=str(row.get("hardware")) if row.get("hardware") else None,
        quality_score=optional_number(row.get("quality_absolute")),
        quality_reference=optional_number(row.get("quality_reference")),
        quality_delta=optional_number(row.get("quality_delta")),
        source_tokens=optional_number(row.get("visible_initial_tokens")),
        full_visible_tokens=optional_number(row.get("visible_initial_tokens")),
        visible_tokens=optional_number(row.get("visible_recovered_tokens")),
        active_kv_tokens=active_kv,
        active_kv_bytes=optional_number(row.get("active_kv_bytes")),
        reference_kv_tokens=optional_number(row.get("active_kv_reference_tokens")),
        active_kv_reduction=optional_number(row.get("active_kv_saving")),
        consumer_layers=tuple(int(value) for value in row.get("detail_kv_layers") or ()),
        hot_bytes=optional_number(row.get("active_kv_bytes")),
        warm_bytes=optional_number(row.get("backing_bytes")),
        persistence_mode=(
            str(row.get("compression_policy")) if row.get("compression_policy") else None
        ),
        ttft_ms=optional_number(row.get("ttft_ms")),
        itl_ms=optional_number(row.get("inter_token_ms")),
        completion_p50_ms=optional_number(row.get("p50_ms")),
        completion_p95_ms=optional_number(row.get("p95_ms")),
        completion_p99_ms=optional_number(row.get("p99_ms")),
        requests_per_second=optional_number(row.get("throughput")),
        output_tokens_per_second=optional_number(row.get("tokens_per_second")),
        peak_device_memory_bytes=optional_number(row.get("peak_hbm_bytes")),
        peak_host_memory_bytes=optional_number(row.get("peak_ram_bytes")),
        transfer_h2d_bytes=optional_number(row.get("h2d_bytes")),
        transfer_h2d_ms=optional_number(row.get("h2d_ms")),
        prefix_cache_hit_rate=optional_number(row.get("cache_hit_rate")),
        routing_method=str(row.get("search_policy")) if row.get("search_policy") else None,
        sample_count=int(row.get("sample_count") or 0),
        seed_count=int(row.get("seed_count") or 0),
        evidence_tier=str(row.get("evidence_tier") or "UNKNOWN"),
        evidence_provenance=provenance,
        experiment_status=_evidence_status(row),
        notes=str(row.get("notes") or ""),
    )


def _engine_row(row: Mapping[str, Any], index: int) -> ProductMatrixRow:
    condition = str(row.get("condition") or "default")
    quality = optional_number(row.get("quality_absolute"))
    quality_name = str(row.get("quality_metric_name") or "unknown")
    provenance = "docs/papers/shared/results/pra_engine_benchmarks.json"
    return ProductMatrixRow(
        row_id=f"engine-{index:03d}-{_slug(row['engine'])}-{_slug(row['model_id'])}-{_slug(condition)}",
        model_family=str(row["model_family"]),
        model_id=str(row["model_id"]),
        model_revision=row.get("model_revision"),
        model_size=_integer(row.get("parameter_count")),
        model_variant=condition,
        engine=str(row["engine"]),
        engine_version=row.get("engine_version"),
        hardware=str(row.get("hardware") or "unknown"),
        profile=str(row.get("profile") or "CANDIDATE"),
        profile_status=_status(row.get("profile"), profile=True),
        workload=str(row.get("workload") or "unknown"),
        dataset=str(row.get("dataset") or "unknown"),
        quality_metric=quality_name,
        integration_level="E0",
        representation="FULL" if condition == "full_context" else "E0_SELECTED",
        quantization=str(row.get("precision")) if row.get("precision") else None,
        accelerator=str(row.get("hardware")) if row.get("hardware") else None,
        quality_score=quality,
        quality_reference=optional_number(row.get("quality_reference")),
        quality_delta=optional_number(row.get("quality_delta")),
        task_success=quality if "recovery" in quality_name or "success" in quality_name else None,
        source_tokens=optional_number(row.get("visible_initial_tokens")),
        full_visible_tokens=optional_number(row.get("visible_initial_tokens")),
        visible_tokens=optional_number(row.get("visible_recovered_tokens")),
        active_kv_tokens=optional_number(row.get("active_native_kv_tokens")),
        active_kv_bytes=optional_number(row.get("active_native_kv_bytes")),
        hot_bytes=optional_number(row.get("active_native_kv_bytes")),
        warm_bytes=optional_number(row.get("backing_bytes")),
        persistence_mode=str(row.get("precision")) if row.get("precision") else None,
        ttft_ms=optional_number(row.get("ttft_ms_p50")),
        ttft_p50_ms=optional_number(row.get("ttft_ms_p50")),
        completion_ms=optional_number(row.get("completion_latency_ms_p50")),
        completion_p50_ms=optional_number(row.get("completion_latency_ms_p50")),
        prefix_cache_hit_rate=optional_number(row.get("warm_cached_tokens_mean")),
        sample_count=int(row.get("sample_count") or 0),
        seed_count=int(row.get("seed_count") or 0),
        evidence_tier=str(row.get("evidence_tier") or "UNKNOWN"),
        evidence_provenance=provenance,
        experiment_status=_evidence_status(row),
        verified_invariants=("selector_frozen",),
        notes=f"condition={condition}; native_pra_status={row.get('native_pra_status', 'unknown')}",
    )


def _mean(rows: list[Mapping[str, Any]], path: tuple[str, ...]) -> float | None:
    values: list[float] = []
    for row in rows:
        value: object = row
        for key in path:
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(key)
        number = optional_number(value)
        if number is not None:
            values.append(number)
    return statistics.fmean(values) if values else None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _values(rows: list[Mapping[str, Any]], path: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for row in rows:
        value: object = row
        for key in path:
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(key)
        number = optional_number(value)
        if number is not None:
            values.append(number)
    return values


def _model_family(model_id: str) -> str:
    normalized = model_id.lower()
    if "qwen" in normalized:
        return "qwen"
    if "llama" in normalized:
        return "llama"
    if "gemma" in normalized:
        return "gemma"
    return model_id.split("/")[-1].split("-")[0].lower()


def _matched_exact_parity(rows: list[Mapping[str, Any]]) -> float | None:
    pairs: dict[tuple[str, str, int, str], dict[str, str]] = {}
    for row in rows:
        selection = row.get("selection") or {}
        key = (
            str(selection.get("selection_id")),
            str(row.get("regime")),
            int(row.get("request_ordinal") or 0),
            str(row.get("query_sha256")),
        )
        pairs.setdefault(key, {})[str(row.get("condition"))] = str(row.get("output"))
    complete = [
        outputs
        for outputs in pairs.values()
        if {"e0_selected_text", "e2_native_kv"} <= set(outputs)
    ]
    if not complete:
        return None
    return statistics.fmean(
        float(outputs["e0_selected_text"] == outputs["e2_native_kv"])
        for outputs in complete
    )


def _matched_rows() -> list[ProductMatrixRow]:
    """Aggregate the selector-frozen natural E0/E2 artifacts into product rows."""

    engine_dirs = {
        "vllm": "paper6_vllm",
        "sglang": "paper6_1_sglang",
        "mlx": "paper6_2_mlx",
    }
    rows: list[ProductMatrixRow] = []
    for engine, directory in engine_dirs.items():
        for dataset in ("qasper", "hotpotqa", "2wikimultihopqa"):
            path = RESULTS / directory / f"expanded_matched_e0_e2_{dataset}.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            all_rows = list(payload["rows"])
            parity = _matched_exact_parity(all_rows)
            model_id = str(payload["model_id"])
            for condition in ("e0_selected_text", "e2_native_kv"):
                for regime in (
                    "cold_one_shot",
                    "warm_repeated",
                    "multi_query_same_resource",
                    "concurrent_shared_resource",
                ):
                    selected = [
                        row
                        for row in all_rows
                        if row["condition"] == condition and row["regime"] == regime
                    ]
                    if not selected:
                        continue
                    native = condition == "e2_native_kv"
                    ttft = _values(selected, ("metrics", "serving", "ttft_ms"))
                    itl = _values(selected, ("metrics", "serving", "itl_ms"))
                    latency = _values(
                        selected, ("metrics", "serving", "total_latency_ms")
                    )
                    selector_ids = sorted(
                        {str(row["selection"]["selection_id"]) for row in selected}
                    )
                    selector_digest = hashlib.sha256(
                        "\n".join(selector_ids).encode("utf-8")
                    ).hexdigest()
                    provenance = str(path.relative_to(ROOT)).replace("\\", "/")
                    rows.append(
                        ProductMatrixRow(
                            row_id=(
                                f"matched-{engine}-{dataset}-{condition}-{regime}"
                            ),
                            model_family=_model_family(model_id),
                            model_id=model_id,
                            model_revision=payload.get("model_revision"),
                            model_size=None,
                            model_variant=regime,
                            engine=engine,
                            engine_version=payload.get("engine_version"),
                            hardware="Apple M5 16GB unified memory",
                            profile="REFERENCE_CORRECTNESS",
                            profile_status="RESEARCH_ONLY" if native else "MEASURED",
                            workload=f"matched_e0_e2/{regime}",
                            dataset=dataset,
                            quality_metric="token_f1",
                            integration_level="E2" if native else "E0",
                            representation="E2_HOT" if native else "E0_SELECTED",
                            selector_digest=selector_digest,
                            accelerator="Apple M5 GPU",
                            ram_bytes=16 * 1024**3,
                            os="macOS",
                            quality_score=_mean(
                                selected, ("metrics", "quality", "task_score")
                            ),
                            em=_mean(selected, ("metrics", "quality", "exact_match")),
                            f1=_mean(selected, ("metrics", "quality", "token_f1")),
                            exact_pair_parity=parity,
                            gold_answer_log_probability=_mean(
                                selected,
                                ("metrics", "quality", "gold_answer_logprob"),
                            ),
                            evidence_recall=_mean(
                                selected, ("metrics", "quality", "evidence_recall")
                            ),
                            source_tokens=_mean(
                                selected, ("metrics", "input", "candidate_tokens")
                            ),
                            visible_tokens=_mean(
                                selected, ("metrics", "input", "visible_prompt_tokens")
                            ),
                            active_kv_tokens=_mean(
                                selected, ("metrics", "pra", "selected_native_kv_tokens")
                            ),
                            active_kv_bytes=_mean(
                                selected, ("metrics", "pra", "active_detail_bytes")
                            ),
                            hot_bytes=_mean(
                                selected, ("metrics", "pra", "retained_detail_bytes")
                            ),
                            ttft_p50_ms=_percentile(ttft, 0.50),
                            ttft_p95_ms=_percentile(ttft, 0.95),
                            ttft_p99_ms=_percentile(ttft, 0.99),
                            itl_p50_ms=_percentile(itl, 0.50),
                            itl_p95_ms=_percentile(itl, 0.95),
                            itl_p99_ms=_percentile(itl, 0.99),
                            completion_p50_ms=_percentile(latency, 0.50),
                            completion_p95_ms=_percentile(latency, 0.95),
                            completion_p99_ms=_percentile(latency, 0.99),
                            output_tokens_per_second=_mean(
                                selected,
                                ("metrics", "serving", "tokens_per_second"),
                            ),
                            prefix_cache_hit_rate=(
                                statistics.fmean(
                                    float(
                                        (row["metrics"]["reuse"].get(
                                            "ordinary_prefix_cache_hit_tokens"
                                        ) or 0)
                                        > 0
                                    )
                                    for row in selected
                                )
                            ),
                            pra_cache_hit_rate=(
                                statistics.fmean(
                                    float(
                                        bool(
                                            row["metrics"]["reuse"].get("pra_hot_hit")
                                        )
                                    )
                                    for row in selected
                                )
                                if native
                                else None
                            ),
                            queries_per_resource=(
                                float(len(selected) / max(len(selector_ids), 1))
                            ),
                            sample_count=len(selected),
                            seed_count=len(
                                {
                                    row.get("extra", {}).get("seed")
                                    for row in selected
                                    if row.get("extra", {}).get("seed") is not None
                                }
                            ),
                            evidence_tier=str(payload["evidence_tier"]),
                            evidence_provenance=provenance,
                            experiment_status="NATURAL_WORKLOAD",
                            verified_invariants=(
                                "selector_frozen",
                                "paired_representation",
                                "native_prefix_namespace_separate",
                            ),
                            notes=(
                                f"condition={condition}; regime={regime}; "
                                f"selection_count={len(selector_ids)}"
                            ),
                        )
                    )
    return rows


def _warm_lifecycle_rows() -> list[ProductMatrixRow]:
    """Import lossless WARM promotion rows from the shared lifecycle campaign."""

    engine_dirs = {
        "vllm": "paper6_vllm",
        "sglang": "paper6_1_sglang",
        "mlx": "paper6_2_mlx",
    }
    rows: list[ProductMatrixRow] = []
    for engine, directory in engine_dirs.items():
        for dataset in ("qasper", "hotpotqa", "2wikimultihopqa"):
            path = (
                RESULTS
                / directory
                / f"live_storage_lifecycle_scale_qwen3_0_6b_{dataset}.json"
            )
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            source_rows = list(payload.get("rows") or ())
            summary = payload["summary"]
            metrics = summary["metrics"]
            usage = summary["usage"]
            warm_latency = [
                float(row["lifecycle_request_latency_ms"]["warm"])
                for row in source_rows
            ]
            hot_latency = [
                float(row["lifecycle_request_latency_ms"]["hot"])
                for row in source_rows
            ]
            warm_hits = float(metrics["hits"].get("warm", 0))
            warm_misses = float(metrics["misses"].get("warm", 0))
            provenance = str(path.relative_to(ROOT)).replace("\\", "/")
            examples = int(summary["examples"])
            rows.append(
                ProductMatrixRow(
                    row_id=f"lifecycle-{engine}-{dataset}-e2-warm",
                    model_family="qwen",
                    model_id=str(payload["model_id"]),
                    model_revision=None,
                    model_size=None,
                    model_variant="warm_promotion",
                    engine=engine,
                    engine_version=payload.get("engine_version"),
                    hardware="Apple M5 16GB unified memory",
                    profile="REFERENCE_CORRECTNESS",
                    profile_status="RESEARCH_ONLY",
                    workload="matched_e2_warm_lifecycle",
                    dataset=dataset,
                    quality_metric="hot_warm_exact_parity",
                    integration_level="E2",
                    representation="E2_WARM",
                    accelerator="Apple M5 GPU",
                    ram_bytes=16 * 1024**3,
                    os="macOS",
                    quality_score=float(summary["hot_warm_exact"]) / examples,
                    exact_pair_parity=float(summary["hot_warm_exact"]) / examples,
                    source_tokens=statistics.fmean(
                        float(row["source_tokens"]) for row in source_rows
                    ),
                    active_kv_tokens=statistics.fmean(
                        float(row["source_tokens"]) for row in source_rows
                    ),
                    active_kv_bytes=statistics.fmean(
                        float(row["native_bytes"]) for row in source_rows
                    ),
                    hot_bytes=float(usage["hot_bytes"]),
                    warm_bytes=float(usage["warm_bytes"]),
                    cold_bytes=float(usage["cold_bytes"]),
                    persistence_mode="lossless_warm",
                    completion_p50_ms=_percentile(warm_latency, 0.50),
                    completion_p95_ms=_percentile(warm_latency, 0.95),
                    completion_p99_ms=_percentile(warm_latency, 0.99),
                    warm_hit_rate=(
                        warm_hits / (warm_hits + warm_misses)
                        if warm_hits + warm_misses > 0
                        else None
                    ),
                    reloads=float(metrics["reloads"]),
                    reload_amplification=(
                        statistics.fmean(warm_latency)
                        / statistics.fmean(hot_latency)
                        if hot_latency and statistics.fmean(hot_latency) > 0
                        else None
                    ),
                    sample_count=examples,
                    seed_count=0,
                    evidence_tier="NATURAL_QA_STORAGE_LIFECYCLE",
                    evidence_provenance=provenance,
                    experiment_status="NATURAL_WORKLOAD",
                    verified_invariants=(
                        "hot_warm_exact",
                        "restart_recovered",
                        "request_lifetime_pinning",
                    ),
                    notes=(
                        f"bytes_read_total={metrics['bytes_read']}; "
                        f"promotions={metrics['promotions']}"
                    ),
                )
            )
    return rows


def _airllm_natural_rows() -> list[ProductMatrixRow]:
    """Import the bounded AirLLM/HF natural E0/E2 CUDA follow-up."""

    result_dir = RESULTS / "paper6_6_airllm"
    expanded = result_dir / "tinyllama_rtx5060_natural_60_summary.json"
    path = (
        expanded
        if expanded.exists()
        else result_dir / "tinyllama_cuda_natural_summary.json"
    )
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = str(path.relative_to(ROOT)).replace("\\", "/")
    rows: list[ProductMatrixRow] = []
    for comparison in payload["comparisons"]:
        for integration_level in ("E0", "E2"):
            native = integration_level == "E2"
            prefix = "e2" if native else "e0"
            rows.append(
                ProductMatrixRow(
                    row_id=(
                        "airllm-natural-"
                        f"{comparison['dataset']}-{comparison['regime']}-{integration_level.lower()}"
                    ),
                    model_family="llama",
                    model_id=str(payload["model_id"]),
                    model_revision=None,
                    model_size=1_100_000_000,
                    model_variant="airllm_hf_layer_streamed",
                    engine="airllm",
                    engine_version="3.3.0",
                    hardware=(
                        f"{payload['device']} 8 GiB"
                        if "5060" in str(payload["device"])
                        else f"{payload['device']} 4 GiB"
                    ),
                    profile="REFERENCE_CORRECTNESS",
                    profile_status="RESEARCH_ONLY" if native else "CALIBRATION_PENDING",
                    workload=f"selector_frozen_natural/{comparison['regime']}",
                    dataset=str(comparison["dataset"]),
                    quality_metric="token_f1",
                    integration_level="E1" if native else integration_level,
                    representation=(
                        "E2_CANDIDATE_NATIVE" if native else "E0_SELECTED"
                    ),
                    quantization="float16",
                    accelerator=str(payload["device"]),
                    quality_score=float(comparison[f"{prefix}_token_f1"]),
                    f1=float(comparison[f"{prefix}_token_f1"]),
                    task_success=float(comparison[f"{prefix}_answer_containment"]),
                    exact_pair_parity=float(comparison["exact_output_pair_parity"]),
                    visible_tokens=float(
                        comparison["e2_visible_tokens"]
                        if native
                        else comparison["e0_visible_tokens"]
                    ),
                    active_kv_tokens=(
                        float(comparison["e2_native_tokens"]) if native else None
                    ),
                    peak_device_memory_bytes=float(
                        comparison[f"{prefix}_peak_cuda_bytes"]
                    ),
                    completion_ms=1000.0
                    * float(comparison[f"{prefix}_completion_seconds"]),
                    completion_p50_ms=1000.0
                    * float(comparison[f"{prefix}_completion_seconds"]),
                    sample_count=int(comparison["samples_per_condition"]),
                    seed_count=1,
                    evidence_tier=str(payload["evidence_tier"]),
                    evidence_provenance=provenance,
                    experiment_status="NATURAL_WORKLOAD",
                    verified_invariants=(
                        "selector_frozen",
                        "matched_selection",
                        "native_kv_consumed" if native else "ordinary_selected_text",
                    ),
                    notes=(
                        f"visible_reduction={comparison['visible_token_reduction']:.6f}; "
                        f"e2_over_e0_completion={comparison['e2_over_e0_completion']:.6f}; "
                        f"{payload['example_count']}-example transport cohort; "
                        f"mean_reference_encode_seconds="
                        f"{payload.get('mean_reference_encode_seconds', 'NOT_MEASURED')}; "
                        f"native_e2_candidate={str(native).lower()}; semantic parity open"
                    ),
                )
            )
    return rows


def _mlx_m4_cross_model_rows() -> list[ProductMatrixRow]:
    """Import the M4 Pro Qwen3-1.7B/4B oracle-evidence replication."""

    path = RESULTS / "paper6_2_mlx/m4_cross_model_summary.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = str(path.relative_to(ROOT)).replace("\\", "/")
    conditions = {
        (str(row["model_id"]), str(row["dataset"]), str(row["condition"])): row
        for row in payload["conditions"]
    }
    rows: list[ProductMatrixRow] = []
    variants = (
        ("ordinary_split", "E0", "E0_SELECTED", "float16", "CALIBRATION_PENDING"),
        ("native_fp", "E2", "E2_HOT", "float16", "CALIBRATION_PENDING"),
        (
            "native_int8_resident",
            "E2",
            "E2_COLD_INT8_RESIDENT",
            "int8_resident_float16_active",
            "RESEARCH_ONLY",
        ),
    )
    for comparison in payload["comparisons"]:
        model_id = str(comparison["model_id"])
        dataset = str(comparison["dataset"])
        model_size = 4_000_000_000 if "4B" in model_id else 1_700_000_000
        for condition, level, representation, quantization, status in variants:
            source = conditions[(model_id, dataset, condition)]
            resident = float(source["resident_selected_kv_bytes"])
            rows.append(
                ProductMatrixRow(
                    row_id=f"mlx-m4-{_slug(model_id)}-{dataset}-{condition}",
                    model_family="qwen",
                    model_id=model_id,
                    model_revision="main",
                    model_size=model_size,
                    model_variant=condition,
                    engine="mlx-lm",
                    engine_version="0.31.3",
                    hardware="Apple M4 Pro, 20-core GPU, 48 GiB",
                    profile="REFERENCE_CORRECTNESS",
                    profile_status=status,
                    workload="oracle_evidence_original_answer_qa",
                    dataset=dataset,
                    quality_metric="token_f1",
                    integration_level=level,
                    representation=representation,
                    quantization=quantization,
                    accelerator="Apple M4 Pro 20-core GPU",
                    ram_bytes=48 * 1024**3,
                    quality_score=float(source["token_f1"]),
                    f1=float(source["token_f1"]),
                    gold_answer_log_probability=float(source["gold_answer_logprob"]),
                    hot_bytes=(
                        resident if condition == "native_fp" else None
                    ),
                    cold_bytes=(
                        resident if condition == "native_int8_resident" else None
                    ),
                    completion_ms=float(source["completion_latency_ms"]),
                    sample_count=int(source["sample_count"]),
                    seed_count=int(source["seed_count"]),
                    evidence_tier=str(payload["evidence_tier"]),
                    evidence_provenance=provenance,
                    experiment_status="NATURAL_WORKLOAD",
                    verified_invariants=(
                        "selector_frozen",
                        "oracle_evidence",
                        "ordinary_native_aggregate_f1_parity",
                    ),
                    notes=(
                        "request-local dequantization before attention"
                        if condition == "native_int8_resident"
                        else "in-process MLX runner; online queueing not measured"
                    ),
                )
            )
    return rows


def _vllm_cuda_concurrency_rows() -> list[ProductMatrixRow]:
    """Import request-scope controls for the prefix-shaped CUDA connector."""

    path = RESULTS / "paper6_vllm/cuda_connector_concurrency_rtx5060_summary.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = str(path.relative_to(ROOT)).replace("\\", "/")
    rows: list[ProductMatrixRow] = []
    for source in payload["rows"]:
        concurrency = int(source["concurrency"])
        condition = str(source["condition"])
        rows.append(
            ProductMatrixRow(
                row_id=f"vllm-cuda-connector-{condition}-c{concurrency}",
                model_family="llama",
                model_id=str(payload["model_id"]),
                model_revision="main",
                model_size=1_100_000_000,
                model_variant=condition,
                engine="vllm",
                engine_version=str(payload["engine_version"]),
                hardware=str(payload["device"]),
                profile="REFERENCE_CORRECTNESS",
                profile_status="RESEARCH_ONLY",
                workload=f"offline_connector_concurrency_c{concurrency}",
                dataset="synthetic_code_recovery",
                quality_metric="expected_resource_recovery",
                integration_level="E1",
                representation="E2_CANDIDATE_PREFIX_SHAPED",
                quantization="bfloat16",
                accelerator=str(payload["device"]),
                quality_score=float(source["recovery_rate"]),
                task_success=float(source["recovery_rate"]),
                completion_ms=float(source["completion_ms"]),
                requests_per_second=float(source["requests_per_second"]),
                output_tokens_per_second=float(source["output_tokens_per_second"]),
                peak_device_memory_bytes=float(source["peak_allocated_mib"])
                * 1048576,
                batch_occupancy=float(concurrency),
                sample_count=int(source["requests"]),
                seed_count=1,
                evidence_tier=str(payload["evidence_tier"]),
                evidence_provenance=provenance,
                experiment_status="RESEARCH_ONLY",
                verified_invariants=(
                    "request_scope_isolation",
                    "zero_forbidden_leakage",
                    "wrong_memory_causal_control",
                ),
                notes=(
                    f"leakage_rate={float(source['leakage_rate']):.6f}; "
                    "source slots remain scheduler-visible; offline V1 batch"
                ),
            )
        )
    return rows


def _mlx_m4_pressure_rows() -> list[ProductMatrixRow]:
    """Import the Qwen3-4B eight-resource compact-residency frontier."""

    path = RESULTS / "paper6_2_mlx/m4_pressure_summary.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = str(path.relative_to(ROOT)).replace("\\", "/")
    rows: list[ProductMatrixRow] = []
    for source in payload["rows"]:
        budget = int(source["resident_resource_budget"])
        rows.append(
            ProductMatrixRow(
                row_id=f"mlx-m4-pressure-{source['dataset']}-k{budget}",
                model_family="qwen",
                model_id=str(source["model_id"]),
                model_revision="main",
                model_size=4_000_000_000,
                model_variant=f"int8_resident_k{budget}",
                engine="mlx-lm",
                engine_version="0.31.3",
                hardware="Apple M4 Pro, 20-core GPU, 48 GiB",
                profile="ECONOMY_CANDIDATE",
                profile_status="RESEARCH_ONLY",
                workload="bounded_residency_three_round_session",
                dataset=str(source["dataset"]),
                quality_metric="token_f1",
                integration_level="E2",
                representation="E2_COLD_INT8_RESIDENT",
                quantization="int8_resident_float16_active",
                accelerator="Apple M4 Pro 20-core GPU",
                ram_bytes=48 * 1024**3,
                quality_score=float(source["token_f1"]),
                f1=float(source["token_f1"]),
                gold_answer_log_probability=float(source["gold_answer_logprob"]),
                cold_bytes=float(source["resident_mib_mean"]) * 1048576,
                completion_ms=float(source["completion_ms_mean"]),
                completion_p95_ms=float(source["completion_ms_p95"]),
                reloads=float(source["reloads_mean"]),
                evictions=float(source["evictions_mean"]),
                reload_amplification=float(source["reload_fraction"]),
                queries_per_resource=25.0 / 8.0,
                sample_count=int(source["sample_count"]),
                seed_count=int(source["seed_count"]),
                evidence_tier=str(payload["evidence_tier"]),
                evidence_provenance=provenance,
                experiment_status="RESEARCH_ONLY",
                verified_invariants=(
                    "quality_invariant_across_budget",
                    "request_lifetime_pinning",
                    "working_set_threshold",
                ),
                notes=(
                    f"resident_resources={budget}/8; "
                    f"resolve_ms_mean={float(source['resolve_ms_mean']):.3f}; "
                    f"resolve_ms_p95={float(source['resolve_ms_p95']):.3f}"
                ),
            )
        )
    return rows


def _mlx_consumer_scaling_rows() -> list[ProductMatrixRow]:
    """Import corrected MLX model/layer scaling into the product registry.

    The corrected campaign synchronizes both ordinary source prefill and native
    K/V encoding before starting request timers. ``E2_CONCAT_WARM`` consumes
    memory at every eligible layer and is the evidence behind ``BALANCED``.
    Segmented and reduced-layer realizations remain calibration candidates.
    """

    result_root = RESULTS / "paper6_2_mlx"
    paths = sorted((result_root / "model_consumer_scaling").glob("qwen3_*.json"))
    paths.extend(sorted((result_root / "model_consumer_scaling_m5").glob("*.json")))
    paths.extend(sorted((result_root / "model_consumer_cross_family").glob("*.json")))
    warmed_stems = {
        path.stem.removesuffix("_warmed")
        for path in paths
        if path.parent.name == "model_consumer_scaling_m5"
        and path.stem.endswith("_warmed")
    }
    rows: list[ProductMatrixRow] = []
    for path in paths:
        if (
            path.parent.name == "model_consumer_scaling_m5"
            and path.stem in warmed_stems
        ):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("experiment") != "matched_economics_and_consumer_depth":
            continue
        source_rows = list(payload["rows"])
        runtime = payload["runtime"]
        model_id = str(payload["model_id"])
        layer_count = int(payload["layer_count"])
        model_size_match = re.search(r"(\d+(?:\.\d+)?)B", model_id, re.IGNORECASE)
        model_size = (
            int(float(model_size_match.group(1)) * 1_000_000_000)
            if model_size_match
            else None
        )
        hardware_model = str(runtime.get("hardware_model") or "Apple Silicon")
        physical_memory = optional_number(runtime.get("physical_memory_bytes"))
        provenance = str(path.relative_to(ROOT)).replace("\\", "/")
        by_condition = {
            str(aggregate["condition"]): aggregate
            for aggregate in payload["aggregate"]
        }
        for condition, aggregate in by_condition.items():
            selected = [row for row in source_rows if row["condition"] == condition]
            native = condition != "E0_WARM"
            all_layers = condition in {"E2_CONCAT_WARM", "E2_SEGMENTED_ALL_LAYERS"}
            concat = condition == "E2_CONCAT_WARM"
            profile = "BALANCED" if (not native or concat) else "REDUCED_CANDIDATE"
            profile_status = "MEASURED" if (not native or concat) else "CALIBRATION_PENDING"
            representation = (
                "E0_SELECTED"
                if not native
                else "E2_HOT" if concat else "E2_SEGMENTED_CANDIDATE"
            )
            consumer_layers = tuple(selected[0].get("consumer_layers") or ())
            source_tokens = statistics.fmean(float(row["source_tokens"]) for row in selected)
            active_layer_tokens = statistics.fmean(
                float(row["source_tokens"]) * float(row["consumer_layer_count"])
                for row in selected
            )
            ttft = [float(row["ttft_ms"]) for row in selected]
            itl = [float(row["itl_ms"]) for row in selected]
            completion = [float(row["completion_latency_ms"]) for row in selected]
            rows.append(
                ProductMatrixRow(
                    row_id=(
                        f"mlx-consumer-{_slug(hardware_model)}-{_slug(model_id)}-"
                        f"{_slug(path.parent.name)}-{_slug(path.stem)}-{_slug(condition)}"
                    ),
                    model_family=_model_family(model_id),
                    model_id=model_id,
                    model_revision=payload.get("model_revision"),
                    model_size=model_size,
                    model_variant=condition,
                    engine="mlx-lm",
                    engine_version=str(runtime.get("mlx_lm") or "unknown"),
                    hardware=(
                        f"{hardware_model}, {int(physical_memory / 1024**3)} GiB unified memory"
                        if physical_memory is not None
                        else hardware_model
                    ),
                    profile=profile,
                    profile_status=profile_status,
                    workload="oracle_evidence_original_answer_qa/layer_scaling",
                    dataset="qasper+hotpotqa+2wikimultihopqa",
                    quality_metric="token_f1",
                    integration_level="E2" if native else "E0",
                    representation=representation,
                    quantization="4bit_model/float16_kv",
                    accelerator=f"{hardware_model} GPU",
                    ram_bytes=physical_memory,
                    quality_score=float(aggregate["token_f1"]),
                    f1=float(aggregate["token_f1"]),
                    exact_pair_parity=float(aggregate["sequence_agreement_vs_e0"]),
                    gold_answer_log_probability=float(aggregate["gold_answer_logprob"]),
                    source_tokens=source_tokens,
                    visible_tokens=statistics.fmean(
                        float(row["visible_prompt_tokens"]) for row in selected
                    ),
                    active_kv_tokens=active_layer_tokens if native else None,
                    active_kv_bytes=(
                        float(aggregate["active_detail_bytes"]) if native else None
                    ),
                    reference_kv_tokens=(source_tokens * layer_count if native else None),
                    consumer_layers=consumer_layers,
                    hot_bytes=float(aggregate["active_detail_bytes"]) if native else None,
                    peak_memory_bytes=float(aggregate["peak_unified_memory_bytes"]),
                    ttft_p50_ms=_percentile(ttft, 0.50),
                    ttft_p95_ms=_percentile(ttft, 0.95),
                    ttft_p99_ms=_percentile(ttft, 0.99),
                    itl_p50_ms=_percentile(itl, 0.50),
                    itl_p95_ms=_percentile(itl, 0.95),
                    itl_p99_ms=_percentile(itl, 0.99),
                    completion_p50_ms=_percentile(completion, 0.50),
                    completion_p95_ms=_percentile(completion, 0.95),
                    completion_p99_ms=_percentile(completion, 0.99),
                    sample_count=int(aggregate["samples"]),
                    seed_count=int(aggregate["seeds"]),
                    evidence_tier=str(selected[0]["evidence_tier"]),
                    evidence_provenance=provenance,
                    experiment_status="NATURAL_WORKLOAD",
                    verified_invariants=(
                        "selector_frozen",
                        "ordinary_prefill_synchronized",
                        "native_encoding_synchronized",
                        "all_eligible_consumer_layers" if all_layers else "reduced_layer_candidate",
                    ),
                    notes=(
                        f"consumer_layer_fraction={aggregate['consumer_layer_fraction']:.6f}; "
                        f"cold_usable_context_ms={aggregate['cold_usable_context_ms']:.6f}; "
                        "BALANCED is all eligible layers; reduced and segmented paths "
                        "remain CALIBRATION_PENDING"
                    ),
                )
            )
    return rows


def _openvino_distractor_rows() -> list[ProductMatrixRow]:
    """Import the frozen-evidence OpenVINO natural distractor ablation."""

    path = RESULTS / "paper6_3_openvino/distractor_ablation_summary.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = str(path.relative_to(ROOT)).replace("\\", "/")
    rows: list[ProductMatrixRow] = []
    for source in payload["rows"]:
        mode = str(source["mode"])
        distractors = int(source["distractor_count"])
        condition = "evidence" if mode == "evidence_only" else f"{mode}_{distractors}"
        rows.append(
            ProductMatrixRow(
                row_id=f"openvino-distractor-{source['dataset']}-{condition}",
                model_family="qwen",
                model_id="OpenVINO/Qwen2-0.5B-Instruct-int4-ov",
                model_revision=None,
                model_size=500_000_000,
                model_variant=condition,
                engine="openvino_genai",
                engine_version=str(payload["engine_version"]),
                hardware="Intel Iris Xe integrated GPU",
                profile="REFERENCE_CORRECTNESS",
                profile_status="CALIBRATION_PENDING",
                workload="frozen_evidence_distractor_ablation",
                dataset=str(source["dataset"]),
                quality_metric="token_f1",
                integration_level="E0",
                representation="E0_SELECTED",
                quantization="int4",
                cpu="Intel Core i7-1355U",
                accelerator="Intel Iris Xe",
                os="Windows",
                quality_score=float(source["token_f1"]),
                task_success=float(source["answer_containment"]),
                em=float(source["exact_match"]),
                f1=float(source["token_f1"]),
                evidence_recall=float(source["evidence_recall_at_4"]),
                source_tokens=float(source["mean_source_tokens"]),
                visible_tokens=float(source["mean_source_tokens"]),
                ttft_ms=float(source["ttft_p50_ms"]),
                ttft_p50_ms=float(source["ttft_p50_ms"]),
                ttft_p95_ms=float(source["ttft_p95_ms"]),
                completion_ms=float(source["completion_p50_ms"]),
                completion_p50_ms=float(source["completion_p50_ms"]),
                requests_per_second=float(source["successful_requests_per_second"]),
                successful_requests_per_second=float(
                    source["successful_requests_per_second"]
                ),
                sample_count=int(source["sample_count"]),
                seed_count=1,
                evidence_tier=str(payload["evidence_tier"]),
                evidence_provenance=provenance,
                experiment_status="NATURAL_WORKLOAD",
                verified_invariants=(
                    "selector_frozen",
                    "answer_evidence_preserved",
                    "ordinary_selected_text",
                ),
                notes=(
                    f"mode={mode}; distractor_count={distractors}; "
                    f"mean_distractor_tokens={source['mean_distractor_tokens']:.3f}; "
                    "gold answer log-probability not measured"
                ),
            )
        )
    return rows


def _openvino_cross_model_rows() -> list[ProductMatrixRow]:
    """Import matched selected/full OpenVINO rows across model families."""

    path = RESULTS / "paper6_3_openvino/cross_model_summary.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = str(path.relative_to(ROOT)).replace("\\", "/")
    rows: list[ProductMatrixRow] = []
    for source in payload["rows"]:
        raw_model = str(source["model_id"]).replace("\\", "/")
        model_id = raw_model.rsplit("/", 1)[-1]
        lowered = model_id.lower()
        if "tinyllama" in lowered:
            model_family, model_size = "tinyllama", 1_100_000_000
        elif "1.5b" in lowered:
            model_family, model_size = "qwen", 1_500_000_000
        else:
            model_family, model_size = "qwen", 500_000_000
        for condition in ("selected", "full"):
            selected = condition == "selected"
            visible = float(
                source["selected_prompt_tokens"]
                if selected
                else source["full_prompt_tokens"]
            )
            score = float(source[f"{condition}_f1"])
            containment = float(source[f"{condition}_answer_containment"])
            rows.append(
                ProductMatrixRow(
                    row_id=(
                        f"openvino-cross-model-{model_size}-{source['dataset']}-{condition}"
                    ),
                    model_family=model_family,
                    model_id=model_id,
                    model_revision=None,
                    model_size=model_size,
                    model_variant=f"int4_{condition}",
                    engine="openvino_genai",
                    engine_version=str(payload["engine_version"]),
                    hardware="Intel Iris Xe integrated GPU",
                    profile=("REFERENCE_CORRECTNESS" if selected else "FULL_CONTEXT"),
                    profile_status="CALIBRATION_PENDING",
                    workload="matched_natural_e0_cross_model",
                    dataset=str(source["dataset"]),
                    quality_metric="token_f1",
                    integration_level="E0",
                    representation="E0_SELECTED" if selected else "FULL_CONTEXT",
                    quantization="int4",
                    cpu="Intel Core i7-1355U",
                    accelerator="Intel Iris Xe",
                    os="Windows",
                    quality_score=score,
                    quality_reference=float(source["full_f1"]),
                    quality_delta=score - float(source["full_f1"]),
                    task_success=containment,
                    f1=score,
                    full_visible_tokens=float(source["full_prompt_tokens"]),
                    visible_tokens=visible,
                    ttft_ms=float(source[f"{condition}_ttft_p50_ms"]),
                    ttft_p50_ms=float(source[f"{condition}_ttft_p50_ms"]),
                    ttft_p95_ms=float(source[f"{condition}_ttft_p95_ms"]),
                    sample_count=int(source["samples_per_condition"]),
                    seed_count=1,
                    evidence_tier=str(payload["evidence_tier"]),
                    evidence_provenance=provenance,
                    experiment_status="NATURAL_WORKLOAD",
                    verified_invariants=(
                        "selector_frozen",
                        "matched_selection",
                        "ordinary_selected_text",
                    ),
                    notes=(
                        f"condition={condition}; full_over_selected_ttft="
                        f"{source['full_over_selected_ttft']:.6f}; "
                        "native K/V not available"
                    ),
                )
            )
    return rows


def build_matrix() -> ProductMatrix:
    profile_path = RESULTS / "paper4_5_runtime/pra_profile_benchmarks.json"
    engine_path = RESULTS / "pra_engine_benchmarks.json"
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))["benchmarks"]
    engines = json.loads(engine_path.read_text(encoding="utf-8"))["rows"]
    rows = [*(_profile_row(row, i) for i, row in enumerate(profiles, 1))]
    rows.extend(_engine_row(row, i) for i, row in enumerate(engines, 1))
    rows.extend(_matched_rows())
    rows.extend(_warm_lifecycle_rows())
    rows.extend(_airllm_natural_rows())
    rows.extend(_mlx_m4_cross_model_rows())
    rows.extend(_mlx_m4_pressure_rows())
    rows.extend(_mlx_consumer_scaling_rows())
    rows.extend(_vllm_cuda_concurrency_rows())
    rows.extend(_openvino_distractor_rows())
    rows.extend(_openvino_cross_model_rows())
    return ProductMatrix("2026-08-engine-qualification-v3", tuple(rows))


def _percent(value: float | None) -> str:
    return "--" if value is None else f"{100 * value:.1f}\\%"


def _tex(value: object) -> str:
    return str(value).replace("_", "\\_")


def write_table(matrix: ProductMatrix, path: Path) -> None:
    profile_rows = [row for row in matrix.rows if row.row_id.startswith("profile-")]
    lines = [
        "\\begin{tabular}{lllrrrrl}",
        "\\toprule",
        "Model & Profile & Level & Quality $\\Delta$ & Visible $\\downarrow$ & Active KV $\\downarrow$ & TTFT & Status \\\\",
        "\\midrule",
    ]
    for row in profile_rows:
        lines.append(
            f"{row.model_family.title()} & {_tex(row.profile)} & {row.integration_level} & "
            f"{_percent(row.quality_delta)} & {_percent(row.visible_token_reduction)} & "
            f"{_percent(row.active_kv_reduction)} & "
            f"{'--' if row.ttft_p50_ms is None else f'{row.ttft_p50_ms:.1f}'} & "
            f"{row.profile_status.replace('_', ' ')} \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    args = parser.parse_args()
    matrix = build_matrix()
    matrix.write(args.output)
    write_table(matrix, args.table)
    print(json.dumps({
        "schema_version": matrix.schema_version,
        "registry_version": matrix.registry_version,
        "rows": len(matrix.rows),
        "quality_adjusted_rows": sum(
            row.successful_requests_per_second is not None for row in matrix.rows
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
