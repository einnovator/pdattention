"""Build the precision qualification ledger and publication artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pvariance
from typing import Any, Mapping

import matplotlib.pyplot as plt
import yaml

from pra_hf.bundle import PRAModelBundle
from pra_hf.bundle_catalog import load_bundle_catalog
from pra_hf.canonical_evidence import (
    CanonicalEvidenceRecord,
    EvidenceCondition,
    MeasurementState,
)


ROOT = Path(__file__).resolve().parents[2]
LADDER = ROOT / "src/pra_hf/model_profiles/precision_ladder.yaml"
BUNDLES = ROOT / "artifacts/pra_hf/bundles"
DEFAULT_OUTPUT = (
    ROOT
    / "docs/papers/shared/results/paper4_5_runtime_productization/precision_ladder"
)


def _bundle_path(repo: str) -> Path:
    return BUNDLES / repo.split("/", 1)[-1]


def _canonical_records(bundle: PRAModelBundle) -> list[CanonicalEvidenceRecord]:
    raw = bundle.qualification.get("canonical_evidence", [])
    values = raw if isinstance(raw, list) else [raw]
    fields = CanonicalEvidenceRecord.model_fields
    return [
        CanonicalEvidenceRecord.model_validate(
            {name: value[name] for name in fields if name in value}
        )
        for value in values
        if isinstance(value, Mapping)
    ]


def _value(
    record: CanonicalEvidenceRecord | None,
    condition: EvidenceCondition,
    metric: str,
) -> float | None:
    if record is None:
        return None
    observation = record.conditions[condition].metrics.get(metric)
    if observation is None or observation.state != MeasurementState.MEASURED:
        return None
    return observation.value


def _condition_state(
    record: CanonicalEvidenceRecord | None, condition: EvidenceCondition
) -> str:
    if record is None:
        return (
            "NO_QUALIFIED_ADAPTER"
            if condition == EvidenceCondition.PRA_ADAPTOR_BUNDLE
            else "NEEDS_RUN"
        )
    observations = tuple(record.conditions[condition].metrics.values())
    if any(value.state == MeasurementState.MEASURED for value in observations):
        return "MEASURED"
    states = {value.state.value for value in observations}
    for state in (
        "BLOCKED",
        "NO_QUALIFIED_ADAPTER",
        "CALIBRATION_PENDING",
        "NEEDS_RUN",
        "NOT_MEASURED",
        "NOT_APPLICABLE",
    ):
        if state in states:
            return state
    return "NOT_MEASURED"


def _measured_row(target: Mapping[str, Any], catalog: Mapping[str, Any]) -> dict[str, Any]:
    bundle_path = _bundle_path(str(target["catalog_repo"]))
    bundle = PRAModelBundle.from_pretrained(bundle_path)
    records = _canonical_records(bundle)
    record = next((value for value in records if value.key.task == "combined"), None)
    if record is None and records:
        record = records[0]
    headline = bundle.qualification.get("headline", [])
    headline_rows = [row for row in headline if isinstance(row, Mapping)]
    combined = next((row for row in headline_rows if row.get("dataset") == "combined"), None)
    if combined is None and headline_rows:
        combined = headline_rows[0]
    precision = str(target["precisions"][0])
    base_model = str(target.get("base_model", target["model"]))
    no_pra_quality = _value(record, EvidenceCondition.NO_PRA, "token_f1")
    pra_quality = _value(record, EvidenceCondition.PRA_NO_ADAPTOR, "token_f1")
    adaptor_quality = _value(
        record, EvidenceCondition.PRA_ADAPTOR_BUNDLE, "token_f1"
    )
    quality_delta = (
        pra_quality - no_pra_quality
        if pra_quality is not None and no_pra_quality is not None
        else None
    )
    return {
        "model": target["model"],
        "base_model": base_model,
        "model_revision": bundle.base_model.get("revision"),
        "size": target["size"],
        "family": target["family"],
        "tier": target["tier"],
        "priority": target["priority"],
        "precision_family": precision,
        "precision_encoding": catalog["precision_encoding"],
        "engine": catalog["engine"],
        "mode": catalog["recommendation"].split(" with ")[0],
        "profile": catalog["profile"],
        "qualification": catalog["evidence_tier"],
        "datasets": catalog.get("datasets", []),
        "bundle": target["catalog_repo"],
        "artifact": catalog.get("artifact"),
        "condition_no_pra": _condition_state(record, EvidenceCondition.NO_PRA),
        "condition_pra_no_adaptor": _condition_state(
            record, EvidenceCondition.PRA_NO_ADAPTOR
        ),
        "condition_pra_adaptor_bundle": _condition_state(
            record, EvidenceCondition.PRA_ADAPTOR_BUNDLE
        ),
        "token_f1_no_pra": no_pra_quality,
        "token_f1_pra_no_adaptor": pra_quality,
        "token_f1_pra_adaptor_bundle": adaptor_quality,
        "token_f1_delta_pra_no_adaptor": quality_delta,
        "token_f1_incremental_adaptor_gain": (
            adaptor_quality - pra_quality
            if adaptor_quality is not None and pra_quality is not None
            else None
        ),
        "visible_tokens_no_pra": _value(
            record, EvidenceCondition.NO_PRA, "visible_tokens"
        ),
        "visible_tokens_pra_no_adaptor": _value(
            record, EvidenceCondition.PRA_NO_ADAPTOR, "visible_tokens"
        ),
        "ttft_p95_ms_no_pra": _value(
            record, EvidenceCondition.NO_PRA, "ttft_p95_ms"
        ),
        "ttft_p95_ms_pra_no_adaptor": _value(
            record, EvidenceCondition.PRA_NO_ADAPTOR, "ttft_p95_ms"
        ),
        "output_tokens_per_second_no_pra": _value(
            record, EvidenceCondition.NO_PRA, "output_tokens_per_second"
        ),
        "output_tokens_per_second_pra_no_adaptor": _value(
            record, EvidenceCondition.PRA_NO_ADAPTOR, "output_tokens_per_second"
        ),
        "peak_memory_bytes_no_pra": _value(
            record, EvidenceCondition.NO_PRA, "peak_memory_bytes"
        ),
        "peak_memory_bytes_pra_no_adaptor": _value(
            record, EvidenceCondition.PRA_NO_ADAPTOR, "peak_memory_bytes"
        ),
        "exact_pairs": (combined or {}).get("semantic_equivalence", {}).get(
            "exact_output_pairs"
        ),
        "paired_examples": (combined or {}).get("semantic_equivalence", {}).get(
            "paired_examples"
        ),
        "memory_gate": "NOT_MEASURED",
        "feature_extraction_precision": "NOT_MEASURED",
        "adaptor_parameter_precision": "NOT_MEASURED",
    }


def _planned_row(target: Mapping[str, Any], precision: str) -> dict[str, Any]:
    return {
        "model": target["model"],
        "base_model": target.get("base_model", target["model"]),
        "model_revision": None,
        "size": target["size"],
        "family": target["family"],
        "tier": target["tier"],
        "priority": target["priority"],
        "precision_family": precision,
        "precision_encoding": f"native-{str(precision).lower()}",
        "engine": "NOT_MEASURED",
        "mode": "Native Memory",
        "profile": "BALANCED",
        "qualification": "NOT_MEASURED",
        "datasets": [],
        "bundle": None,
        "artifact": None,
        "condition_no_pra": "NEEDS_RUN",
        "condition_pra_no_adaptor": "NEEDS_RUN",
        "condition_pra_adaptor_bundle": "NO_QUALIFIED_ADAPTER",
        "token_f1_no_pra": None,
        "token_f1_pra_no_adaptor": None,
        "token_f1_pra_adaptor_bundle": None,
        "token_f1_delta_pra_no_adaptor": None,
        "token_f1_incremental_adaptor_gain": None,
        "visible_tokens_no_pra": None,
        "visible_tokens_pra_no_adaptor": None,
        "ttft_p95_ms_no_pra": None,
        "ttft_p95_ms_pra_no_adaptor": None,
        "output_tokens_per_second_no_pra": None,
        "output_tokens_per_second_pra_no_adaptor": None,
        "peak_memory_bytes_no_pra": None,
        "peak_memory_bytes_pra_no_adaptor": None,
        "exact_pairs": None,
        "paired_examples": None,
        "memory_gate": "NOT_MEASURED",
        "feature_extraction_precision": "NOT_MEASURED",
        "adaptor_parameter_precision": "NOT_MEASURED",
    }


def build_ladder() -> dict[str, Any]:
    plan = yaml.safe_load(LADDER.read_text(encoding="utf-8"))
    catalog = load_bundle_catalog()
    catalog_by_repo = {row["repo"]: row for row in catalog["bundles"]}
    measured = [row for row in plan["targets"] if row.get("catalog_repo")]
    measured_keys = {
        (str(row.get("base_model", row["model"])), str(row["precisions"][0]))
        for row in measured
    }
    rows = [
        _measured_row(row, catalog_by_repo[str(row["catalog_repo"])])
        for row in measured
    ]
    for target in plan["targets"]:
        if target.get("catalog_repo"):
            continue
        for precision in target["precisions"]:
            if (str(target["model"]), str(precision)) in measured_keys:
                continue
            rows.append(_planned_row(target, str(precision)))
    rows.sort(
        key=lambda row: (
            str(row["priority"]), str(row["base_model"]), str(row["precision_family"])
        )
    )
    deltas: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["token_f1_delta_pra_no_adaptor"] is not None:
            deltas[str(row["base_model"])].append(
                float(row["token_f1_delta_pra_no_adaptor"])
            )
    stability = {
        model: {
            "measured_precisions": len(values),
            "mean_pra_token_f1_delta": mean(values),
            "pra_gain_variance": pvariance(values) if len(values) > 1 else None,
        }
        for model, values in sorted(deltas.items())
    }
    return {
        "schema_version": 1,
        "updated": plan["updated"].isoformat()
        if hasattr(plan["updated"], "isoformat")
        else str(plan["updated"]),
        "canonical_key": [
            "task", "hardware", "engine", "model", "precision_family",
            "precision_encoding", "mode", "profile",
        ],
        "rows": rows,
        "cross_precision_stability": stability,
        "summary": {
            "rows": len(rows),
            "measured_variants": sum(
                row["condition_no_pra"] == "MEASURED"
                and row["condition_pra_no_adaptor"] == "MEASURED"
                for row in rows
            ),
            "adaptor_bundle_variants": sum(
                row["condition_pra_adaptor_bundle"] == "MEASURED" for row in rows
            ),
            "needs_run": sum(row["qualification"] == "NOT_MEASURED" for row in rows),
        },
    }


def write_ladder(payload: Mapping[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "precision_ladder.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    rows = list(payload["rows"])
    with (output / "precision_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_tex(rows, output / "generated_precision_ladder.tex")
    _write_card(rows, output / "precision_card_fragment.md")
    _plot(rows, output / "precision_observed_metrics")
    _plot_metric(
        rows, "token_f1_delta_pra_no_adaptor", "PRA quality delta",
        output / "pra_quality_delta_vs_precision", include_zero=True,
    )
    _plot_metric(
        rows, "ttft_p95_ms_pra_no_adaptor", "PRA TTFT p95 (ms)",
        output / "ttft_vs_precision",
    )
    _plot_metric(
        rows, "output_tokens_per_second_pra_no_adaptor", "PRA output tokens/s",
        output / "tokens_per_second_vs_precision",
    )
    _plot_metric(
        rows, "peak_memory_bytes_pra_no_adaptor", "PRA peak memory (bytes)",
        output / "peak_memory_vs_precision",
    )
    _plot_metric(
        rows, "visible_tokens_pra_no_adaptor", "PRA visible tokens",
        output / "visible_tokens_vs_precision",
    )
    _plot_metric(
        rows, "token_f1_incremental_adaptor_gain", "Incremental adaptor F1 gain",
        output / "adaptor_gain_vs_precision", include_zero=True,
    )


def _write_tex(rows: list[Mapping[str, Any]], path: Path) -> None:
    measured = [row for row in rows if row["condition_no_pra"] == "MEASURED"]
    lines = [
        r"\begin{tabular}{llllrrr}",
        r"\toprule",
        r"Model & Encoding & Evidence & Pairs & $\Delta$F1 & Vis. base & Vis. PRA \\",
        r"\midrule",
    ]
    for row in measured:
        delta = row["token_f1_delta_pra_no_adaptor"]
        lines.append(
            " & ".join(
                (
                    str(row["base_model"]).replace("_", r"\_"),
                    str(row["precision_encoding"]).replace("_", r"\_"),
                    str(row["qualification"]).replace("_", r"\_"),
                    f"{row['exact_pairs']}/{row['paired_examples']}",
                    "--" if delta is None else f"{float(delta):+.4f}",
                    "--" if row["visible_tokens_no_pra"] is None else f"{float(row['visible_tokens_no_pra']):.1f}",
                    "--" if row["visible_tokens_pra_no_adaptor"] is None else f"{float(row['visible_tokens_pra_no_adaptor']):.1f}",
                )
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_card(rows: list[Mapping[str, Any]], path: Path) -> None:
    lines = [
        "## Precision qualification",
        "",
        "A precision row qualifies only the exact conversion, engine, mode, profile, and linked evidence.",
        "",
        "| Model | Size | Family | Precision/encoding | Engine | Mode | Profile | Qualification | Datasets |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if row["bundle"] is None:
            continue
        lines.append(
            f"| `{row['model']}` | {row['size']} | {row['family']} "
            f"| {row['precision_family']} / {row['precision_encoding']} | {row['engine']} "
            f"| {row['mode']} | {row['profile']} | {row['qualification']} "
            f"| {', '.join(row['datasets']) or 'NOT_MEASURED'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(rows: list[Mapping[str, Any]], prefix: Path) -> None:
    measured = [
        row
        for row in rows
        if row["visible_tokens_no_pra"] is not None
        and row["visible_tokens_pra_no_adaptor"] is not None
    ]
    if not measured:
        return
    labels = [
        f"{str(row['base_model']).split('/')[-1]}\n{row['precision_encoding']}"
        for row in measured
    ]
    x = list(range(len(measured)))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(
        [value - 0.2 for value in x],
        [float(row["visible_tokens_no_pra"]) for row in measured],
        width=0.4,
        label="No PRA",
    )
    axes[0].bar(
        [value + 0.2 for value in x],
        [float(row["visible_tokens_pra_no_adaptor"]) for row in measured],
        width=0.4,
        label="PRA, no adaptor",
    )
    axes[0].set_ylabel("Mean visible tokens")
    axes[0].legend()
    axes[1].bar(
        x,
        [
            float(row["token_f1_delta_pra_no_adaptor"] or 0.0)
            for row in measured
        ],
        color="#c54f31",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("PRA token-F1 delta")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(prefix.with_suffix(".png"), dpi=180)
    figure.savefig(prefix.with_suffix(".pdf"))
    plt.close(figure)


def _plot_metric(
    rows: list[Mapping[str, Any]],
    metric: str,
    ylabel: str,
    prefix: Path,
    *,
    include_zero: bool = False,
) -> None:
    """Render one precision metric without converting missing cells to zero."""

    measured = [
        row for row in rows
        if row.get(metric) is not None
    ]
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    if measured:
        labels = [
            f"{str(row['base_model']).split('/')[-1]}\n{row['precision_encoding']}"
            for row in measured
        ]
        values = [float(row[metric]) for row in measured]
        axis.bar(range(len(values)), values, color="#26766f")
        axis.set_xticks(range(len(values)), labels, rotation=25, ha="right")
        if include_zero:
            axis.axhline(0.0, color="black", linewidth=0.8)
        axis.grid(axis="y", alpha=0.25)
    else:
        axis.text(
            0.5, 0.5, "No matched measurements",
            ha="center", va="center", transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    axis.set_ylabel(ylabel)
    figure.tight_layout()
    figure.savefig(prefix.with_suffix(".png"), dpi=180)
    figure.savefig(prefix.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_ladder()
    write_ladder(payload, args.output)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
