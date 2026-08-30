"""Normalize profile and engine evidence into the stable product-matrix schema."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

from pra_hf.product_matrix import ProductMatrix, ProductMatrixRow, optional_number


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/papers/shared/results"
DEFAULT_OUTPUT = RESULTS / "pra_product_matrix_v1.json"
DEFAULT_TABLE = RESULTS / "generated_pra_product_matrix_v1.tex"


def _slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-") or "unknown"


def _integer(value: object) -> int | None:
    number = optional_number(value)
    return None if number is None else int(number)


def _status(value: object, *, profile: bool = False) -> str:
    normalized = str(value or "NOT_MEASURED").upper()
    if profile and normalized in {"CANDIDATE", "SMOKE"}:
        return "CALIBRATION_PENDING"
    if normalized in {
        "MEASURED", "CALIBRATION_PENDING", "RESEARCH_ONLY", "BLOCKED", "NOT_MEASURED"
    }:
        return normalized
    return "NOT_MEASURED"


def _profile_row(row: Mapping[str, Any], index: int) -> ProductMatrixRow:
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
        quality_score=optional_number(row.get("quality_absolute")),
        quality_reference=optional_number(row.get("quality_reference")),
        quality_delta=optional_number(row.get("quality_delta")),
        source_tokens=optional_number(row.get("visible_initial_tokens")),
        visible_tokens=optional_number(row.get("visible_recovered_tokens")),
        active_kv_tokens=optional_number(row.get("active_kv_tokens")),
        reference_kv_tokens=optional_number(row.get("active_kv_reference_tokens")),
        active_kv_reduction=optional_number(row.get("active_kv_saving")),
        warm_bytes=optional_number(row.get("backing_bytes")),
        persistence_mode=str(row.get("compression_policy")) if row.get("compression_policy") else None,
        ttft_ms=optional_number(row.get("ttft_ms")),
        itl_ms=optional_number(row.get("inter_token_ms")),
        completion_ms=optional_number(row.get("p50_ms")),
        requests_per_second=optional_number(row.get("throughput")),
        peak_memory_bytes=(
            optional_number(row.get("peak_hbm_bytes"))
            or optional_number(row.get("peak_ram_bytes"))
        ),
        transfer_bytes=optional_number(row.get("h2d_bytes")),
        routing_method=str(row.get("search_policy")) if row.get("search_policy") else None,
        sample_count=int(row.get("sample_count") or 0),
        seed_count=int(row.get("seed_count") or 0),
        evidence_tier=str(row.get("evidence_tier") or "UNKNOWN"),
        evidence_provenance=str(row.get("artifact_path") or "paper4_5 profile registry"),
        experiment_status=_status(row.get("measurement_status")),
        notes=str(row.get("notes") or ""),
    )


def _engine_row(row: Mapping[str, Any], index: int) -> ProductMatrixRow:
    condition = str(row.get("condition") or "default")
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
        quality_metric=str(row.get("quality_metric_name") or "unknown"),
        quality_score=optional_number(row.get("quality_absolute")),
        quality_reference=optional_number(row.get("quality_reference")),
        quality_delta=optional_number(row.get("quality_delta")),
        source_tokens=optional_number(row.get("visible_initial_tokens")),
        visible_tokens=optional_number(row.get("visible_recovered_tokens")),
        active_kv_tokens=optional_number(row.get("active_native_kv_tokens")),
        hot_bytes=optional_number(row.get("active_native_kv_bytes")),
        warm_bytes=optional_number(row.get("backing_bytes")),
        persistence_mode=str(row.get("precision")) if row.get("precision") else None,
        ttft_ms=optional_number(row.get("ttft_ms_p50")),
        completion_ms=optional_number(row.get("completion_latency_ms_p50")),
        sample_count=int(row.get("sample_count") or 0),
        seed_count=int(row.get("seed_count") or 0),
        evidence_tier=str(row.get("evidence_tier") or "UNKNOWN"),
        evidence_provenance="docs/papers/shared/results/pra_engine_benchmarks.json",
        experiment_status=_status(row.get("measurement_status")),
        notes=f"condition={condition}; native_pra_status={row.get('native_pra_status', 'unknown')}",
    )


def build_matrix() -> ProductMatrix:
    profile_path = RESULTS / "paper4_5_runtime/pra_profile_benchmarks.json"
    engine_path = RESULTS / "pra_engine_benchmarks.json"
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))["benchmarks"]
    engines = json.loads(engine_path.read_text(encoding="utf-8"))["rows"]
    rows = [*(_profile_row(row, i) for i, row in enumerate(profiles, 1))]
    rows.extend(_engine_row(row, i) for i, row in enumerate(engines, 1))
    return ProductMatrix("2026-08-product-matrix-v1", tuple(rows))


def _percent(value: float | None) -> str:
    return "--" if value is None else f"{100 * value:.1f}\\%"


def _tex(value: object) -> str:
    return str(value).replace("_", "\\_")


def write_table(matrix: ProductMatrix, path: Path) -> None:
    profile_rows = [row for row in matrix.rows if row.row_id.startswith("profile-")]
    lines = [
        "\\begin{tabular}{llrrrrrl}",
        "\\toprule",
        "Model & Profile & Quality $\\Delta$ & Visible $\\downarrow$ & Active KV $\\downarrow$ & TTFT $\\Delta$ & Memory $\\downarrow$ & Status \\\\",
        "\\midrule",
    ]
    for row in profile_rows:
        lines.append(
            f"{row.model_family.title()} & {_tex(row.profile)} & "
            f"{_percent(row.quality_delta)} & {_percent(row.visible_token_reduction)} & "
            f"{_percent(row.active_kv_reduction)} & -- & -- & "
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
        "measured_rows": sum(row.experiment_status == "MEASURED" for row in matrix.rows),
    }, indent=2))


if __name__ == "__main__":
    main()
