"""Apply exact precision identities from the public catalog to local bundles."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from pra_hf.bundle import BundleBuilder, PRAModelBundle
from pra_hf.bundle_catalog import load_bundle_catalog


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLES = ROOT / "artifacts/pra_hf/bundles"


def precision_entry(row: dict[str, Any]) -> dict[str, Any]:
    """Return the qualified precision scope represented by one catalog row."""

    return {
        "precision_family": str(row["precision_family"]).upper(),
        "encoding": str(row["precision_encoding"]),
        "serving_precision": str(row["precision_family"]).upper(),
        "feature_extraction_precision": "NOT_MEASURED",
        "adaptor_parameter_precision": "NOT_MEASURED",
        "engine": str(row["engine"]),
        "mode": str(row["recommendation"]).split(" with ")[0],
        "profile": str(row["profile"]),
        "evidence_tier": str(row["evidence_tier"]),
        "datasets": list(row.get("datasets", ())),
        "artifact": row.get("artifact", "NOT_MEASURED"),
    }


def apply_catalog(*, bundles: Path = DEFAULT_BUNDLES) -> list[Path]:
    """Update manifests and regenerate cards without changing component payloads."""

    changed: list[Path] = []
    for row in load_bundle_catalog()["bundles"]:
        directory = bundles / str(row["repo"]).split("/", 1)[-1]
        manifest = directory / "bundle.yaml"
        if not manifest.is_file():
            continue
        value = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        expected = [precision_entry(dict(row))]
        if value.get("supported_precisions") != expected:
            value["supported_precisions"] = expected
            manifest.write_text(
                yaml.safe_dump(value, sort_keys=False, allow_unicode=False),
                encoding="utf-8",
            )
            changed.append(manifest)

        bundle = PRAModelBundle.from_pretrained(directory, validate=False)
        card = BundleBuilder.model_card(bundle)
        card_path = directory / "README.md"
        if not card_path.is_file() or card_path.read_text(encoding="utf-8") != card:
            card_path.write_text(card, encoding="utf-8")
            changed.append(card_path)
        PRAModelBundle.from_pretrained(directory).validate()
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", type=Path, default=DEFAULT_BUNDLES)
    args = parser.parse_args()
    changed = apply_catalog(bundles=args.bundles)
    print(f"Updated {len(changed)} bundle artifacts.")


if __name__ == "__main__":
    main()
