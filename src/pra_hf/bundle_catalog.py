"""Declarative public PRA bundle catalog and documentation renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .bundle_evidence import EVIDENCE_TIERS


DEFAULT_CATALOG = Path(__file__).parent / "model_profiles" / "bundle_catalog.yaml"
DEFAULT_BUNDLES = Path(__file__).parents[2] / "artifacts" / "pra_hf" / "bundles"


def load_bundle_catalog(path: str | Path = DEFAULT_CATALOG) -> dict[str, Any]:
    """Load and validate the ordered public catalog manifest."""

    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    rows = value.get("bundles", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("Bundle catalog requires a non-empty bundles list.")
    orders = [row.get("order") for row in rows if isinstance(row, Mapping)]
    repos = [row.get("repo") for row in rows if isinstance(row, Mapping)]
    if len(orders) != len(set(orders)) or orders != sorted(orders):
        raise ValueError("Bundle catalog order must be unique and ascending.")
    if len(repos) != len(set(repos)):
        raise ValueError("Bundle catalog repository IDs must be unique.")
    for row in rows:
        missing = [key for key in ("repo", "model", "role", "evidence_tier", "recommendation", "qualification_date") if not row.get(key)]
        if missing:
            raise ValueError(f"Catalog row {row.get('repo')!r} is missing: {', '.join(missing)}")
        if row["evidence_tier"] not in EVIDENCE_TIERS:
            raise ValueError(f"Catalog row {row['repo']!r} has an invalid evidence tier.")
    return value


def validate_collection_membership(catalog: Mapping[str, Any], repo_ids: set[str]) -> None:
    """Reject a Collection that omits any manifest row marked as published."""

    expected = {row["repo"] for row in catalog["bundles"] if row.get("publication_status") == "PUBLISHED"}
    missing = sorted(expected - repo_ids)
    if missing:
        raise ValueError("Hugging Face Collection is missing: " + ", ".join(missing))


def render_catalog(catalog: Mapping[str, Any]) -> str:
    lines = [
        "# PRA Runtime Bundle Catalog", "",
        "PRA Runtime Bundles package structural mappings, profiles, optional learned components, exact compatibility metadata, and qualification evidence. They do not replace or duplicate model weights.", "",
        "| Order | Runtime bundle | Exact model identity | Role | Evidence | Recommendation | Release |",
        "| ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for row in catalog["bundles"]:
        lines.append(
            f"| {row['order']} | [`{row['repo']}`](https://huggingface.co/{row['repo']}) | `{row['model']}` | {row['role']} | {row['evidence_tier']} | {row['recommendation']} | {row.get('publication_status', 'CANDIDATE')} |"
        )
    lines += ["", "The order reflects useful measured evidence, not publication date. `AVAILABLE`, `QUALIFIED`, and `RECOMMENDED` are independent states.", ""]
    return "\n".join(lines)


def render_qualification_matrix(catalog: Mapping[str, Any]) -> str:
    lines = [
        "# Bundle Qualification Matrix", "",
        "Every row is scoped to the exact model revision, quantization, engine, profile, mode, hardware, and linked artifact. Family resemblance does not transfer qualification.", "",
        "| Bundle | Engine | Recommended mode | Profile | Quality gate | Context saving | Evidence | Artifact |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for row in catalog["bundles"]:
        artifact = row.get("artifact", "")
        artifact_link = f"[source](https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime/{artifact})" if artifact else "NOT_MEASURED"
        lines.append(
            f"| [`{row['repo'].split('/', 1)[-1]}`](https://huggingface.co/{row['repo']}) | {row['engine']} | {row['recommendation'].split(' with ')[0]} | {row['profile']} | {row['quality_gate']} | {row['context_saving']} | {row['evidence_tier']} | {artifact_link} |"
        )
    lines += [
        "", "## Canonical condition audit", "",
        "This audit asks whether the same task, exact model, engine/hardware, mode, and profile have been measured under all three conditions. `AVAILABLE_EXISTING` here means that at least the quality, context, serving, and memory fields present in the linked selector-frozen artifact can be imported; it does not imply that every requested metric exists.", "",
        "| Task/dataset | HW/engine | Model | Mode | Profile | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in catalog["bundles"]:
        paired_transport = "mac_scaling/" in str(row.get("artifact", ""))
        state = "AVAILABLE_EXISTING" if paired_transport else "NEEDS_RUN"
        bundle_state = "NEEDS_RUN"
        lines.append(
            f"| {'Natural QA (QASPER / HotpotQA / 2Wiki)' if paired_transport else 'Exact-identity qualification workload'} "
            f"| {row['engine']} / artifact-recorded hardware | `{row['model']}` "
            f"| {row['recommendation'].split(' with ')[0]} | {row['profile']} "
            f"| `{state}` | `{state}` | `{bundle_state}` "
            f"| `{'AVAILABLE_EXISTING' if paired_transport else 'NEEDS_RUN'}` | `{bundle_state}` |"
        )
    lines += [
        "", "The three MLX natural-QA rows predate immutable bundle resolution: their original-model and generic native-PRA conditions can be normalized, while the Runtime Bundle condition remains `NEEDS_RUN`. Routing-only artifacts remain research diagnostics and do not fill end-task cells.",
        "", "## Evidence tiers", "",
        "| Tier | Meaning |", "| --- | --- |",
        "| `PRODUCTION_QUALIFIED` | Production-scale workload, isolation, reliability, and economic gates passed. |",
        "| `ENGINE_QUALIFIED` | Paired end-task and engine behavior measured for an exact identity. |",
        "| `CONTROLLED` | Bounded controlled evidence; production generalization is not established. |",
        "| `RESEARCH` | Mechanism research or dataset-specific component; not a deployment default. |",
        "| `SMOKE` | Small feasibility check only. |",
        "| `NOT_MEASURED` | The metric or condition has not been measured. |",
        "| `NOT_APPLICABLE` | The metric does not apply to this realization. |",
        "| `BLOCKED` | A known external or implementation dependency prevents measurement. |",
        "", "Reduced consumer-layer profiles remain `CALIBRATION_PENDING`: held-out quality did not support promotion. BALANCED therefore retains all eligible consumer layers for the qualified MLX identities.", "",
    ]
    return "\n".join(lines)


def render_canonical_evidence_catalog(
    catalog: Mapping[str, Any], bundles: str | Path = DEFAULT_BUNDLES
) -> str:
    """Render exact-identity condition metrics for every local public bundle."""

    from .bundle import PRAModelBundle
    from .canonical_evidence import EvidenceCondition, MetricGroup, render_markdown_table

    root = Path(bundles)
    loaded = []
    lines = [
        "# Canonical Evidence Matrix", "",
        "This page compares the same task, hardware, engine, model, mode, and profile under **No PRA**, **PRA - No Adaptor**, and **PRA - Adaptor Bundle**. Values are absolute measurements; each delta is candidate minus No PRA. Missing data is never rendered as zero.", "",
        "## Coverage by model, engine, mode, and profile", "",
        "| Model | Engine | Mode | Profile | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Evidence tier |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for catalog_row in catalog["bundles"]:
        bundle_dir = root / catalog_row["repo"].split("/", 1)[-1]
        if not bundle_dir.is_dir():
            continue
        bundle = PRAModelBundle.from_pretrained(bundle_dir)
        records = _catalog_canonical_records(bundle)
        loaded.append((catalog_row, bundle, records))
        for profile_name, raw_profile in bundle.profiles.items():
            profile = raw_profile if isinstance(raw_profile, Mapping) else {}
            engine = str(profile.get("engine", catalog_row["engine"]))
            mode = str(profile.get("mode", catalog_row["recommendation"].split(" with ")[0]))
            matches = [
                record for record in records
                if record.key.profile.lower() == str(profile_name).lower()
                and record.key.mode.replace("_", "-").lower() == mode.replace(" ", "-").replace("_", "-").lower()
                and (record.key.engine.lower() == engine.lower() or record.key.engine.lower().startswith(engine.lower() + "-"))
            ]
            lines.append(
                f"| `{catalog_row['model']}` | {engine} | {mode} | {str(profile_name).upper()} "
                f"| {_catalog_condition_coverage(matches, EvidenceCondition.NO_PRA)} "
                f"| {_catalog_condition_coverage(matches, EvidenceCondition.PRA_NO_ADAPTOR)} "
                f"| {_catalog_condition_coverage(matches, EvidenceCondition.PRA_ADAPTOR_BUNDLE)} "
                f"| {catalog_row['evidence_tier']} |"
            )

    lines += [
        "", "A `MEASURED (n)` cell reports the number of scalar metrics available for that exact condition. Detailed values follow only for measured records; profile rows without matched evidence remain explicit.", "",
        "## Measured absolute values and deltas", "",
    ]
    detailed = 0
    for catalog_row, _, records in loaded:
        for record in records:
            detailed += 1
            lines += [
                f"### {catalog_row['model']} / {record.key.engine} / {record.key.mode} / {record.key.profile}", "",
                f"Task: `{record.key.task}`. Hardware: `{record.key.hardware}`. Evidence: `{record.evidence_tier}`.", "",
            ]
            for group in MetricGroup:
                if not any(definition.group == group for definition in record.metric_definitions.values()):
                    continue
                lines += [f"#### {group.value.title()}", "", render_markdown_table(record, group).rstrip(), ""]
    if not detailed:
        lines.append("No exact-identity canonical records are currently packaged.")
    lines += [
        "", "## Interpretation", "",
        "The adaptor-bundle column is intentionally distinct from generic PRA. A published bundle may contain only structural mapping and profile metadata, or may include an opt-in learned router. A bundle cell becomes measured only when the immutable bundle revision was resolved during the run.", "",
        "Routing-only recall is reported in each model card's research diagnostics and does not substitute for answer quality, TTFT, ITL, throughput, or memory measurements.", "",
    ]
    return "\n".join(lines)


def _catalog_canonical_records(bundle: Any) -> list[Any]:
    from .canonical_evidence import CanonicalEvidenceRecord

    raw = bundle.qualification.get("canonical_evidence", []) if isinstance(bundle.qualification, Mapping) else []
    values = raw if isinstance(raw, list) else [raw]
    fields = CanonicalEvidenceRecord.model_fields
    return [
        CanonicalEvidenceRecord.model_validate({name: value[name] for name in fields if name in value})
        for value in values if isinstance(value, Mapping)
    ]


def _catalog_condition_coverage(records: list[Any], condition: Any) -> str:
    observations = [
        observation for record in records
        for observation in record.conditions[condition].metrics.values()
    ]
    measured = sum(observation.state.value == "MEASURED" for observation in observations)
    if measured:
        return f"MEASURED ({measured})"
    states = {observation.state.value for observation in observations}
    return next((state for state in ("BLOCKED", "NOT_MEASURED", "NOT_APPLICABLE") if state in states), "NOT_MEASURED")
