"""Generate public bundle catalog pages from the release manifest."""

from pathlib import Path

from pra_hf.bundle_catalog import load_bundle_catalog, render_catalog, render_qualification_matrix


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    catalog = load_bundle_catalog()
    target = ROOT / "docs/site/bundles"
    (target / "catalog.md").write_text(render_catalog(catalog), encoding="utf-8")
    (target / "qualification-matrix.md").write_text(render_qualification_matrix(catalog), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
