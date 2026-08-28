"""Build the canonical Paper 4.5 model/profile benchmark registry and tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from pra_hf.profile_benchmarks import (
    RUNTIME_METRIC_FIELDS,
    ProfileBenchmarkRegistry,
    normalized_quality,
    normalized_saving,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/papers/shared/results/paper4_5_runtime"
LAYER_RESULTS = RESULTS / "layer_profiles"
PACKAGED = ROOT / "src/pra_hf/model_profiles/pra_profile_benchmarks.json"
REGISTRY = RESULTS / "pra_profile_benchmarks.json"
SOURCE_COMMIT = "2f2d4b6794adb8c5263198ba73f9333977e92670"
SOURCE_TIMESTAMP = "2026-08-28T15:06:00+01:00"
PAPER7_COMMIT = "c65e85cf30fa69e46cdb4428ae90afa49a077e11"
PROFILE_VERSION = "2026-08-product-profile-v1"
PROFILE_ORDER = (
    "REFERENCE_CORRECTNESS",
    "QUALITY_MAX",
    "BALANCED",
    "ECONOMY",
)
PROFILE_CANDIDATES = {
    "qwen": {
        "REFERENCE_CORRECTNESS": "all_layers",
        "QUALITY_MAX": "last_24",
        "BALANCED": "all_layers",
        "ECONOMY": "last_1",
    },
    "llama": {
        "REFERENCE_CORRECTNESS": "all_layers",
        "QUALITY_MAX": "all_layers",
        "BALANCED": "all_layers",
        "ECONOMY": "last_1",
    },
    "gemma": {
        "REFERENCE_CORRECTNESS": "all_layers",
        "QUALITY_MAX": "all_layers",
        "BALANCED": "all_layers",
        "ECONOMY": "last_1",
    },
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _average(rows: Iterable[Mapping[str, Any]], field: str) -> float:
    return mean(float(row[field]) for row in rows)


def _rounded(value: float, digits: int = 12) -> float:
    return round(float(value), digits)


def _layers(value: str) -> list[int]:
    return [int(layer) for layer in json.loads(value)]


def _model_metadata() -> dict[str, dict[str, Any]]:
    return {
        key: json.loads((RESULTS / f"hf_{key}_cross_model.json").read_text(encoding="utf-8"))
        for key in ("qwen", "llama", "gemma")
    }


def _recommended_use(model: str, profile: str) -> str:
    if profile == "REFERENCE_CORRECTNESS":
        return "Parity and regression reference"
    if profile == "QUALITY_MAX":
        return "Maximum observed smoke quality; validate on target workload"
    if profile == "BALANCED":
        return "Conservative default; no sparse candidate met the current quality gate"
    return "Research-only minimum-residency diagnostic"


def _profile_rows() -> list[dict[str, Any]]:
    rows = _read_csv(LAYER_RESULTS / "layer_calibration_candidates.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_key"], row["candidate"])].append(row)
    metadata = _model_metadata()
    result: list[dict[str, Any]] = []
    for model in ("qwen", "llama", "gemma"):
        reference_rows = grouped[(model, "all_layers")]
        quality_reference = mean(math.exp(float(row["target_log_probability"])) for row in reference_rows)
        active_reference = _average(reference_rows, "active_layer_tokens")
        detail_reference = _average(reference_rows, "detail_kv_bytes")
        model_meta = metadata[model]
        for profile in PROFILE_ORDER:
            candidate = PROFILE_CANDIDATES[model][profile]
            selected = grouped[(model, candidate)]
            quality = mean(math.exp(float(row["target_log_probability"])) for row in selected)
            retention, delta = normalized_quality(quality, quality_reference)
            active = _average(selected, "active_layer_tokens")
            detail = _average(selected, "detail_kv_bytes")
            physical_layers = _layers(selected[0]["physical_layers"])
            materialized = mean(
                float(row["active_layer_tokens"]) / float(row["consumer_layer_count"])
                for row in selected
            )
            status = "PARTIAL_TOPOLOGY" if model == "gemma" else "MEASURED"
            row: dict[str, Any] = {
                "model_family": model_meta["model"],
                "model_id": model_meta["model_id"],
                "model_revision": model_meta["revision"],
                "parameter_count": model_meta["parameter_count"],
                "num_layers": model_meta["layers"],
                "workload": "semantic_smoke",
                "dataset": "paper4_5_cross_model_diagnostic",
                "split": "three_fixed_cases",
                "profile": profile,
                "profile_registry_version": PROFILE_VERSION,
                "quality_metric": "mean_target_token_probability",
                "quality_absolute": _rounded(quality),
                "quality_reference": _rounded(quality_reference),
                "quality_retention": _rounded(retention),
                "quality_delta": _rounded(delta),
                "visible_initial_tokens": None,
                "visible_recovered_tokens": None,
                "materialized_tokens": _rounded(materialized, 6),
                "active_kv_tokens": _rounded(active, 6),
                "active_kv_reference_tokens": _rounded(active_reference, 6),
                "active_kv_bytes": _rounded(detail, 6),
                "active_kv_saving": _rounded(normalized_saving(active, active_reference)),
                "detail_kv_bytes": _rounded(detail, 6),
                "detail_kv_reference_bytes": _rounded(detail_reference, 6),
                "detail_kv_saving": _rounded(normalized_saving(detail, detail_reference)),
                "address_index_bytes": _rounded(_average(selected, "address_index_bytes"), 0),
                "backing_bytes": None,
                "compression_policy": "none_semantic_smoke",
                "search_policy": "fixed_full_reference",
                "materialization_profile": "full_selected_record",
                "address_layers": [physical_layers[-1]],
                "detail_kv_layers": physical_layers,
                "routing_layers": [physical_layers[-1]],
                "consumer_layers": physical_layers,
                "eligible_consumer_layers": int(model_meta["full_consumption_layers"]),
                "consumer_layer_fraction": _rounded(_average(selected, "consumer_layer_fraction")),
                "engine": "huggingface_eager",
                "engine_version": None,
                "hardware": "NVIDIA GeForce GTX 950M",
                "dtype": "torch.float16",
                "sample_count": len(selected),
                "seed_count": 1,
                "evidence_tier": "SMOKE",
                "measurement_status": status,
                "runtime_measurement_status": "NOT_MEASURED",
                "recommended_use": _recommended_use(model, profile),
                "artifact_path": "docs/papers/shared/results/paper4_5_runtime/layer_profiles/layer_calibration_candidates.csv",
                "commit": SOURCE_COMMIT,
                "timestamp": SOURCE_TIMESTAMP,
                "notes": (
                    "Three-case target-token diagnostic; not workload-scale quality. "
                    + ("Only four native global-attention layers are PRA-eligible; 22 local sliding layers remain unchanged." if model == "gemma" else "All decoder layers are PRA-eligible.")
                ),
            }
            row.update({field: None for field in RUNTIME_METRIC_FIELDS})
            result.append(row)
    return result


def _paper7_evidence() -> dict[str, Any]:
    inherited = RESULTS / "paper7_inherited"
    frontier = {row["condition"]: row for row in _read_csv(inherited / "headroom_pra_cost_frontier.csv")}
    reverse = _read_csv(inherited / "headroom_reverse_eval_hybrid.csv")
    by_channel: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in reverse:
        by_channel[row["channel"]].append(row)

    def weighted_recall(channel: str) -> float:
        rows = by_channel[channel]
        total = sum(int(row["eligible_n"]) for row in rows)
        return sum(float(row["recall_at_4"]) * int(row["eligible_n"]) for row in rows) / total

    current = frontier["PRA_CURRENT_COMPACTOR"]
    compact = frontier["PRA_TYPE_AWARE"]
    recovered = frontier["PRA_TYPE_AWARE_BM25_EMBED"]
    return {
        "source_paper": "Paper 7",
        "source_branch": "research/paper7-typed-adaptive-context",
        "source_commit": PAPER7_COMMIT,
        "source_artifacts": [
            "paper7_inherited/headroom_pra_cost_frontier.csv",
            "paper7_inherited/headroom_reverse_eval_hybrid.csv",
            "paper7_inherited/product_lifecycle_cost_table.csv",
        ],
        "scope": "32-row external exact-evidence diagnostic; 24 extractive-eligible retrieval rows",
        "evidence_tier": "CONTROLLED",
        "metrics": {
            "previous_visible_tokens": float(current["initial_visible_tokens"]),
            "type_aware_visible_tokens": float(compact["initial_visible_tokens"]),
            "compact_only_exact_evidence": float(compact["task_success"]),
            "automatic_hybrid_exact_evidence": float(recovered["task_success"]),
            "typed_recall_at_4": weighted_recall("typed"),
            "bm25_recall_at_4": weighted_recall("bm25"),
            "embedding_recall_at_4": weighted_recall("embedding"),
            "hybrid_recall_at_4": weighted_recall("hybrid"),
        },
        "claim_boundary": "Retrieval/exact-evidence evidence only; not generated-answer quality or general semantic retrieval.",
    }


def build_registry() -> dict[str, Any]:
    payload = {
        "schema_version": "1.0",
        "registry_version": PROFILE_VERSION,
        "description": "Append-only semantic profile and physical realization registry for PRA runtimes.",
        "benchmarks": _profile_rows(),
        "inherited_evidence": [_paper7_evidence()],
    }
    ProfileBenchmarkRegistry(payload)
    return payload


def _tex(value: Any) -> str:
    if value is None:
        return r"\textsc{Not Measured}"
    return str(value).replace("_", r"\_").replace("%", r"\%")


def _percent(value: float | None) -> str:
    return r"\textsc{Not Measured}" if value is None else f"{100.0 * value:.1f}\\%"


def _model_label(row: Mapping[str, Any]) -> str:
    labels = {"qwen": "Qwen3-0.6B", "llama": "Llama-3.2-1B", "gemma": "Gemma-3-1B"}
    return labels[str(row["model_family"])]


def _profile_matrix(rows: list[Mapping[str, Any]]) -> str:
    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llrrrrrrrl@{}}",
        r"\toprule",
        r"Model & Profile & Quality & Retention & $\Delta$ & Mat. & Active K/V & K/V save & Detail save & Evidence/status\\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_model_label(row)} & {_tex(row['profile'])} & {row['quality_absolute']:.4f} & "
            f"{_percent(row['quality_retention'])} & {row['quality_delta']:+.4f} & "
            f"{row['materialized_tokens']:.1f} & {row['active_kv_tokens']:.1f} & "
            f"{_percent(row['active_kv_saving'])} & {_percent(row['detail_kv_saving'])} & "
            f"{row['evidence_tier']}/{_tex(row['measurement_status'])}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}"])
    return "\n".join(lines) + "\n"


def _technical_matrix(rows: list[Mapping[str, Any]]) -> str:
    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lllllrrrr@{}}",
        r"\toprule",
        r"Model & Profile & Compression & Search & Materialization & Address & Detail & Route & Consumer\\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_model_label(row)} & {_tex(row['profile'])} & {_tex(row['compression_policy'])} & "
            f"{_tex(row['search_policy'])} & {_tex(row['materialization_profile'])} & "
            f"{len(row['address_layers'])} & {len(row['detail_kv_layers'])} & "
            f"{len(row['routing_layers'])} & {len(row['consumer_layers'])}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}"])
    return "\n".join(lines) + "\n"


def _runtime_matrix(rows: list[Mapping[str, Any]]) -> str:
    balanced = [row for row in rows if row["profile"] == "BALANCED"]
    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llllrrrr@{}}",
        r"\toprule",
        r"Model & Engine & Hardware & dtype & TTFT & tok/s & p95 & Runtime status\\",
        r"\midrule",
    ]
    for row in balanced:
        lines.append(
            f"{_model_label(row)} & {_tex(row['engine'])} & {_tex(row['hardware'])} & "
            f"{_tex(row['dtype'])} & {_tex(row['ttft_ms'])} & {_tex(row['tokens_per_second'])} & "
            f"{_tex(row['p95_ms'])} & {_tex(row['runtime_measurement_status'])}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}"])
    return "\n".join(lines) + "\n"


def write_outputs(payload: Mapping[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    PACKAGED.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    REGISTRY.write_text(encoded, encoding="utf-8")
    PACKAGED.write_text(encoded, encoding="utf-8")
    rows = list(payload["benchmarks"])
    (RESULTS / "generated_profile_matrix.tex").write_text(_profile_matrix(rows), encoding="utf-8")
    (RESULTS / "generated_profile_technical_matrix.tex").write_text(_technical_matrix(rows), encoding="utf-8")
    (RESULTS / "generated_runtime_matrix.tex").write_text(_runtime_matrix(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    payload = build_registry()
    if not args.validate_only:
        write_outputs(payload)
    ProfileBenchmarkRegistry(payload)
    print(f"validated {len(payload['benchmarks'])} semantic profile realizations")


if __name__ == "__main__":
    main()
