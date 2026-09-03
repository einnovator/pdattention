"""Audit PRA evidence artifacts for condition-attribution errors.

The audit is deliberately conservative.  It reports a legacy ``baseline`` as
ambiguous instead of guessing that it means ordinary No-PRA inference.  Known
selector-frozen E0/E2 layouts are identified as Selected Context versus Native
Memory, which is the attribution error this tool was introduced to prevent.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from pra_hf.canonical_evidence import (
    BUNDLE_CONDITIONS,
    EvidenceCondition,
    MeasurementState,
    PRA_ONLY_METRICS,
    classify_legacy_condition,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = (
    ROOT / "artifacts/pra_hf/bundles",
    ROOT / "docs/papers/shared/results",
    ROOT / "docs/site",
)
STRUCTURED_SUFFIXES = {".json", ".yaml", ".yml"}
TEXT_SUFFIXES = {".md", ".tex"}
LEGACY_KEYS = {"baseline", "no_pra", "pra", "pra_no_adaptor", "pra_adaptor_bundle", "e0", "e2"}


@dataclass(frozen=True)
class AuditFinding:
    severity: str
    code: str
    path: str
    location: str
    message: str


def _walk(value: Any, location: str = "$") -> Iterable[tuple[str, Any]]:
    yield location, value
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{location}[{index}]")


def _condition_metrics(value: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = value.get("metrics", value)
    return metrics if isinstance(metrics, Mapping) else {}


def _is_measured(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, Mapping):
        return str(value.get("state", MeasurementState.MEASURED.value)) == MeasurementState.MEASURED.value and value.get("value") is not None
    return False


def audit_document(document: Any, path: str = "<memory>") -> list[AuditFinding]:
    """Return attribution findings for one parsed JSON/YAML document."""

    findings: list[AuditFinding] = []
    for location, value in _walk(document):
        if not isinstance(value, Mapping):
            continue
        conditions = value.get("conditions")
        if isinstance(conditions, Mapping):
            names = {str(key) for key in conditions}
            legacy = names.intersection(LEGACY_KEYS)
            if legacy:
                mode = str(value.get("key", {}).get("mode", "")) if isinstance(value.get("key"), Mapping) else ""
                code = "LEGACY_E0_E2_ATTRIBUTION" if "native" in mode.lower() and {"no_pra", "pra_no_adaptor"}.issubset(names) else "AMBIGUOUS_LEGACY_CONDITION"
                findings.append(AuditFinding(
                    "ERROR", code, path, f"{location}.conditions",
                    "Legacy conditions require explicit migration; baseline/no_pra must not be inferred as ordinary inference.",
                ))
            no_pra = conditions.get(EvidenceCondition.NO_PRA.value)
            if isinstance(no_pra, Mapping):
                invalid = sorted(PRA_ONLY_METRICS.intersection(_condition_metrics(no_pra)))
                if invalid:
                    findings.append(AuditFinding(
                        "ERROR", "NO_PRA_HAS_PRA_FIELDS", path, f"{location}.conditions.NO_PRA",
                        f"NO_PRA contains PRA-only metrics: {', '.join(invalid)}",
                    ))
            for condition in BUNDLE_CONDITIONS:
                item = conditions.get(condition.value)
                if not isinstance(item, Mapping):
                    continue
                measured = any(_is_measured(metric) for metric in _condition_metrics(item).values())
                if measured and not (item.get("bundle_id") and item.get("bundle_revision")):
                    findings.append(AuditFinding(
                        "ERROR", "BUNDLE_WITHOUT_EXACT_IDENTITY", path,
                        f"{location}.conditions.{condition.value}",
                        "Measured bundle evidence lacks bundle_id and immutable bundle_revision.",
                    ))
        if "condition" in value:
            label = str(value["condition"])
            if label in LEGACY_KEYS:
                classification = classify_legacy_condition(
                    label,
                    provenance=" ".join(str(value.get(key, "")) for key in ("description", "cohort", "provenance")),
                )
                findings.append(AuditFinding(
                    "ERROR" if classification.value == "AMBIGUOUS" else "WARNING",
                    "LEGACY_CONDITION_ID", path, f"{location}.condition",
                    f"Legacy condition {label!r} classifies as {classification.value}.",
                ))
        anonymous = sorted(
            str(key) for key in value
            if str(key) in {"quality_delta", "latency_delta", "context_delta", "delta_no_adaptor", "delta_bundle"}
        )
        if anonymous:
            findings.append(AuditFinding(
                "ERROR", "ANONYMOUS_DELTA", path, location,
                f"Delta keys do not name source and target: {', '.join(anonymous)}",
            ))
    return findings


def audit_text(text: str, path: str) -> list[AuditFinding]:
    lowered = " ".join(text.lower().split())
    findings = []
    if "no pra" in lowered and "same selected evidence" in lowered:
        findings.append(AuditFinding(
            "WARNING", "PROSE_BASELINE_REVIEW", path, "$text",
            "Document mentions No PRA and the same selected evidence; verify this is not E0 versus E2.",
        ))
    if "pra vs baseline" in lowered:
        findings.append(AuditFinding(
            "WARNING", "ANONYMOUS_PRA_BASELINE", path, "$text",
            "Use an explicit No PRA -> Selected Context or Selected Context -> Native Memory comparison.",
        ))
    return findings


def _iter_files(root: Path) -> Iterable[Path]:
    """Yield files without following links or failing on vanished run output."""

    if root.is_file():
        yield root
        return
    if not root.exists():
        return
    for directory, _, names in os.walk(root, followlinks=False, onerror=lambda _: None):
        parent = Path(directory)
        for name in names:
            yield parent / name


def audit_paths(roots: Iterable[Path]) -> tuple[list[AuditFinding], int]:
    findings: list[AuditFinding] = []
    scanned = 0
    for root in roots:
        for path in _iter_files(root):
            if not path.is_file() or path.suffix.lower() not in STRUCTURED_SUFFIXES | TEXT_SUFFIXES:
                continue
            scanned += 1
            relative = str(path.resolve())
            try:
                text = path.read_text(encoding="utf-8-sig")
                if path.suffix.lower() == ".json":
                    findings.extend(audit_document(json.loads(text), relative))
                elif path.suffix.lower() in {".yaml", ".yml"}:
                    findings.extend(audit_document(yaml.safe_load(text), relative))
                else:
                    findings.extend(audit_text(text, relative))
            except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as error:
                findings.append(AuditFinding("WARNING", "UNREADABLE", relative, "$", str(error)))
    return findings, scanned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=list(DEFAULT_ROOTS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when attribution errors exist.")
    args = parser.parse_args()
    findings, scanned = audit_paths(args.paths)
    payload = {
        "schema_version": 1,
        "scanned_files": scanned,
        "errors": sum(item.severity == "ERROR" for item in findings),
        "warnings": sum(item.severity == "WARNING" for item in findings),
        "findings": [asdict(item) for item in findings],
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.strict and payload["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
