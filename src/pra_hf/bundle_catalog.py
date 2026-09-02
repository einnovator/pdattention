"""Declarative public PRA bundle catalog and documentation renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .bundle_evidence import EVIDENCE_TIERS


DEFAULT_CATALOG = Path(__file__).parent / "model_profiles" / "bundle_catalog.yaml"


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
