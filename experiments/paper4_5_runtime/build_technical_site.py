"""Generate public engine documentation from the runtime evidence registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "src/pra_hf/model_profiles/engine_documentation_registry.json"
ENGINE_DIR = ROOT / "docs/site/engines"


def _escape(value: object) -> str:
    """Escape text for a Markdown table cell."""

    return str(value).replace("|", "\\|").replace("\n", " ")


def load_registry(path: Path = REGISTRY) -> dict:
    """Load and validate documentation claims and their checked-in provenance."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = set(payload["evidence_labels"])
    slugs: set[str] = set()
    for engine in payload["engines"]:
        slug = str(engine["slug"])
        if slug in slugs:
            raise ValueError(f"Duplicate engine documentation slug: {slug}")
        slugs.add(slug)
        if engine["evidence"] not in labels:
            raise ValueError(f"Unknown evidence label for {slug}: {engine['evidence']}")
        for metric in engine["metrics"]:
            if metric["status"] not in labels:
                raise ValueError(
                    f"Unknown metric evidence label for {slug}: {metric['status']}"
                )
            source = ROOT / metric["source"]
            if not source.exists():
                raise FileNotFoundError(f"Missing evidence source for {slug}: {source}")
    product_matrix = ROOT / payload["product_matrix"]
    if not product_matrix.is_file():
        raise FileNotFoundError(product_matrix)
    payload["product_matrix_rows"] = len(
        json.loads(product_matrix.read_text(encoding="utf-8"))["rows"]
    )
    return payload


def render_overview(registry: dict) -> str:
    """Render the public capability and recommendation matrix."""

    lines = [
        "# Engine Support",
        "",
        "PRA separates context selection from model-native reuse and scheduler",
        "integration. A deeper integration is not automatically a better deployment.",
        "Start with the recommendation in this table, then qualify the next capability",
        "against the same frozen evidence selection.",
        "",
        "| Engine | Selected Context | Typed PRA Transport | Native Memory | Native Serving | Recommended today | Evidence |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for engine in registry["engines"]:
        capabilities = engine["capabilities"]
        lines.append(
            "| "
            f"[{_escape(engine['name'])}]({_escape(engine['slug'])}.md) | "
            f"{_escape(capabilities['selected_context'])} | "
            f"{_escape(capabilities['typed_transport'])} | "
            f"{_escape(capabilities['native_memory'])} | "
            f"{_escape(capabilities['native_serving'])} | "
            f"{_escape(engine['recommended_today'])} | "
            f"{_escape(engine['evidence'])} |"
        )
    lines.extend(
        [
            "",
            "## Reading the matrix",
            "",
            "- **Validated** means the named capability passed the scope described on",
            "  that engine's page. It is not a claim about every model or workload.",
            "- **Candidate** or **Research only** means the mechanism exists but has not",
            "  passed the product qualification boundary.",
            "- **Not measured** is unknown. It is never interpreted as zero.",
            "- **Not applicable** means the required engine seam does not currently exist.",
            "",
            "See [Metrics & Qualification](../metrics.md) for comparison contracts and",
            "[Research / Evidence](../research/index.md) for paper terminology and raw",
            "artifact provenance.",
            "",
            f"_Generated from the engine documentation registry and {registry['product_matrix_rows']} "
            f"product-matrix rows; evidence current through {registry['evidence_as_of']}._",
            "",
        ]
    )
    return "\n".join(lines)


def render_engine(engine: dict, registry: dict) -> str:
    """Render one engine page using the standard public documentation template."""

    capabilities = engine["capabilities"]
    lines = [
        f"# {engine['name']}",
        "",
        f"_Evidence current through {registry['evidence_as_of']}; generated from checked-in registries._",
        "",
        "## What this engine is for",
        "",
        engine["purpose"],
        "",
        "## Best PRA deployment today",
        "",
        engine["best_deployment"],
        "",
        "## Supported PRA capabilities",
        "",
        "| Capability | Status |",
        "| --- | --- |",
        f"| Selected Context | {capabilities['selected_context']} |",
        f"| Typed PRA Transport | {capabilities['typed_transport']} |",
        f"| Native Memory | {capabilities['native_memory']} |",
        f"| Native Serving | {capabilities['native_serving']} |",
        "",
        "## Architecture",
        "",
        engine["summary"],
        "",
        "```text",
        "application -> typed context -> PRA route/select/materialize",
        f"            -> {engine['name']} -> generated response",
        "```",
        "",
        "## Requirements and tested boundary",
        "",
    ]
    lines.extend(f"- {item}" for item in engine["requirements"])
    lines.extend(["", "## Quickstart", "", "```bash"])
    lines.extend(engine["quickstart"])
    lines.extend(
        [
            "```",
            "",
            "Inspect the capability report before relying on anything beyond Selected",
            "Context. An unavailable capability must fail explicitly or fall back only",
            "when the request permits that fallback.",
            "",
            "## Measured results",
            "",
            "| Metric | Value | Evidence |",
            "| --- | --- | --- |",
        ]
    )
    for metric in engine["metrics"]:
        lines.append(
            f"| {_escape(metric['name'])} | {_escape(metric['value'])} | "
            f"{_escape(metric['status'])} |"
        )
    lines.extend(
        [
            "",
            "## Metrics and explicit gaps",
            "",
        ]
    )
    for metric in engine["metrics"]:
        lines.append(
            f"- **{metric['name']}:** {metric['value']}  "
            f"Provenance: `{metric['source']}`; evidence: {metric['status']}."
        )
    lines.extend(
        [
            "",
            "Unknown metrics remain `NOT_MEASURED`; this page does not convert them to",
            "zero or infer economic benefit from token reduction alone.",
            "",
            "## When to choose Selected Context",
            "",
            engine["selected_context_when"],
            "",
            "## When Native Memory may help",
            "",
            engine["native_memory_when"],
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in engine["limitations"])
    lines.extend(["", "## Research evidence", ""])
    lines.append(
        f"Current public evidence label: **{engine['evidence']}**. See the "
        "[research appendix](../research/index.md) for paper-level names and the "
        "[qualification contract](../metrics.md) before comparing engines."
    )
    lines.extend(["", "## Troubleshooting", ""])
    lines.extend(f"- {item}" for item in engine["troubleshooting"])
    lines.extend(
        [
            "",
            "## Production recommendation",
            "",
            engine["production_recommendation"],
            "",
        ]
    )
    return "\n".join(lines)


def generated_files(registry: dict) -> dict[Path, str]:
    """Return every generated site path and its expected contents."""

    files = {ENGINE_DIR / "overview.md": render_overview(registry)}
    files.update(
        {
            ENGINE_DIR / f"{engine['slug']}.md": render_engine(engine, registry)
            for engine in registry["engines"]
        }
    )
    return files


def build(*, check: bool = False) -> None:
    """Write generated pages, or fail when checked-in pages are stale."""

    registry = load_registry()
    stale: list[Path] = []
    for path, expected in generated_files(registry).items():
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                stale.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected, encoding="utf-8", newline="\n")
    if stale:
        names = ", ".join(str(path.relative_to(ROOT)) for path in stale)
        raise SystemExit(f"Generated technical-site pages are stale: {names}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(check=parse_args().check)
