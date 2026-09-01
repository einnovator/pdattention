"""Generate public engine documentation from the runtime evidence registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "src/pra_hf/model_profiles/engine_documentation_registry.json"
MODEL_REGISTRY = ROOT / "src/pra_hf/model_profiles/model_documentation_registry.json"
ENGINE_DIR = ROOT / "docs/site/engines"
MODEL_PAGE = ROOT / "docs/site/models.md"
REPOSITORY_BLOB = "https://github.com/einnovator/pdattention/blob/research/paper4-5-runtime"


def _escape(value: object) -> str:
    """Escape text for a Markdown table cell."""

    return str(value).replace("|", "\\|").replace("\n", " ")


def _status_icon(status: object) -> str:
    """Add a compact visual cue without replacing the textual status."""

    normalized = str(status).lower().replace("-", " ")
    if normalized in {"validated", "measured", "recommended", "available"}:
        return "✅"
    if normalized in {"candidate", "research only", "partial topology", "engine dependent"}:
        return "🧪"
    if normalized in {"not measured", "not qualified", "qualification pending"}:
        return "⏳"
    if normalized in {"not applicable", "blocked", "unavailable"}:
        return "⛔"
    return "ℹ️"


def _status(value: object) -> str:
    return f"{_status_icon(value)} {_escape(value)}"


def _artifact_link(path: str) -> str:
    normalized = path.replace("\\", "/")
    return f"[artifact]({REPOSITORY_BLOB}/{normalized})"


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


def load_model_registry(path: Path = MODEL_REGISTRY) -> dict:
    """Load model-family claims and validate their checked-in evidence registry."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    profile_registry = ROOT / payload["profile_registry"]
    if not profile_registry.is_file():
        raise FileNotFoundError(profile_registry)
    slugs = [str(row["slug"]) for row in payload.get("families", ())]
    if not slugs or len(slugs) != len(set(slugs)):
        raise ValueError("Model documentation slugs must be non-empty and unique.")
    for family in payload["families"]:
        for key in ("name", "adapter_requirement", "adapter_detail", "evidence"):
            if not family.get(key):
                raise ValueError(f"Missing model documentation field {family['slug']}.{key}")
        examples = set(family.get("examples", ()))
        for model, bundle in family.get("bundle_links", {}).items():
            if model not in examples:
                raise ValueError(f"Bundle link targets an unlisted model: {model}")
            if not bundle.get("url", "").startswith("https://huggingface.co/"):
                raise ValueError(f"Invalid Hugging Face bundle URL for {model}")
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
            f"{_status(capabilities['selected_context'])} | "
            f"{_status(capabilities['typed_transport'])} | "
            f"{_status(capabilities['native_memory'])} | "
            f"{_status(capabilities['native_serving'])} | "
            f"{_escape(engine['recommended_today'])} | "
            f"{_escape(engine['evidence'])} |"
        )
    lines.extend(
        [
            "",
            "## Key",
            "",
            "- ✅ **Validated / measured / recommended:** passed the stated evidence boundary.",
            "- 🧪 **Candidate / research-only:** implemented or under study, but not a default.",
            "- ⏳ **Qualification pending / not measured:** evidence is incomplete; this is not zero.",
            "- ⛔ **Blocked / not applicable:** the required engine seam is unavailable in the stated scope.",
            "",
            "## Reading the matrix",
            "",
            "**Validated** means the named capability passed the scope described on",
            "  that engine's page. It is not a claim about every model or workload.",
            "**Candidate** or **Research only** means the mechanism exists but has not",
            "  passed the product qualification boundary.",
            "**Not measured** is unknown. It is never interpreted as zero.",
            "**Not applicable** means the required engine seam does not currently exist.",
            "",
            "See [Metrics & Qualification](../metrics.md) for comparison contracts and",
            "[Research / Evidence](../research/index.md) for paper terminology and raw",
            "artifact provenance.",
            "",
            f"_Generated from the engine documentation registry and {registry['product_matrix_rows']} "
            f"product-matrix rows; evidence current through {registry['evidence_as_of']}._",
            "",
            "## Observability integration",
            "",
            "Every engine dashboard combines PRA's normalized selection, realization,",
            "prefix/native reuse, and storage metrics with the engine's own telemetry where",
            "available. vLLM can propagate native OTel and exposes serving metrics; SGLang,",
            "OpenVINO/OVMS, TensorRT-LLM/Triton, and llama.cpp expose useful native metrics;",
            "MLX, HF, AirLLM, Ollama, and FreeToken use wrapper observations where a native",
            "surface is absent. Capability does not imply enablement. See the",
            "[observability overview](../observability.md) and [Grafana dashboards](../observability/grafana.md).",
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
        "## What PRA adds to this engine",
        "",
        f"PRA gives {engine['name']} a query-addressed context layer above ordinary",
        "prompt construction. Long-lived documents, tool results, task state, and other",
        "typed resources remain separately addressable; the request receives only the",
        "authorized regions selected for that operation. This reduces visible context",
        "without requiring Native Memory. Deeper native reuse is enabled only where the",
        "table below says it has been measured for this engine.",
        "",
        f"For {engine['name']}, the practical boundary is: {engine['best_deployment']}",
        "",
        "## Three kinds of reuse",
        "",
        "Selected Context session deduplication is owned by the shared PRA runtime.",
        "Engine-native prefix caching is measured independently. Native semantic",
        "memory is used only when this engine/model/hardware path is qualified.",
        "PRA avoids sending selected context again when it is already active, lets",
        "the inference engine reuse ordinary prefix cache where available, and can",
        "reuse native semantic memory on qualified integrations.",
        "",
        "## Supported PRA capabilities",
        "",
        "| Capability | Status |",
        "| --- | --- |",
        f"| Selected Context | {_status(capabilities['selected_context'])} |",
        f"| Typed PRA Transport | {_status(capabilities['typed_transport'])} |",
        f"| Native Memory | {_status(capabilities['native_memory'])} |",
        f"| Native Serving | {_status(capabilities['native_serving'])} |",
        "",
        "**Key:** ✅ qualified evidence · 🧪 candidate/research · ⏳ pending/unmeasured · ⛔ unavailable.",
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
    lines.extend(["", "## Install and launch", "", "Run these commands in order:", "", "```bash"])
    lines.extend(engine["quickstart"])
    lines.extend(
        [
            "```",
            "",
            "",
            "### Command options",
            "",
            "- `--engine` / `-e` selects the runtime provider used for inspection or launch.",
            "- `--mode` / `-m` selects `auto`, `selected-context`, `native-memory`, or",
            "  `native-serving`. Native modes require qualification; `auto` remains",
            "  conservative when incremental economics are not qualified.",
            "- `--profile recommended` selects the current qualified model profile; it",
            "  does not promote smoke-only consumer-layer candidates.",
            "- `--storage memory|balanced|persistent|minimal` controls native-resource",
            "  lifecycle when the selected engine exposes it.",
            "- `--backend` names a gateway adapter; `--backend-url` is the existing",
            "  OpenAI-compatible endpoint. The gateway does not own that engine process.",
            "- `--measurements RESULTS.json` imports selector-frozen quality, latency,",
            "  memory, and lifecycle results into `pra evaluate`.",
            "",
            "Inspect the capability report before relying on anything beyond Selected",
            "Context. An unavailable capability must fail explicitly or fall back only",
            "when the request permits that fallback.",
            "",
            "### Qualify this exact deployment",
            "",
            "```bash",
            f"pra engines --details {engine['slug']}",
            f"pra evaluate MODEL --engine {engine['slug']} --dataset DATASET \\",
            "  --measurements RESULTS.json -o .pra/runs/engine-evaluation",
            "pra recommend .pra/runs/engine-evaluation",
            "pra report .pra/runs/engine-evaluation --format html",
            "```",
            "",
            "## Metrics from the engine paper",
            "",
            "These values are imported from the checked-in paper artifacts. They apply to",
            "the named model, workload, hardware, and engine version rather than every deployment.",
            "",
            "| Metric | Value | Evidence | Source |",
            "| --- | --- | --- | --- |",
        ]
    )
    for metric in engine["metrics"]:
        lines.append(
            f"| {_escape(metric['name'])} | {_escape(metric['value'])} | "
            f"{_escape(metric['status'])} | {_artifact_link(metric['source'])} |"
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


def render_models(registry: dict) -> str:
    """Render model support by family and make adapter requirements explicit."""

    lines = [
        "# Model Support",
        "",
        "PRA support has two distinct boundaries. **Selected Context** works with any",
        "model that accepts ordinary text through a supported runtime or endpoint; it",
        "does not require an attention adapter. **Native Memory** changes model execution",
        "and therefore requires a known structural mapping plus model-specific validation.",
        "",
        "| Family | Selected Context | Native Memory | Structural adapter | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for family in registry["families"]:
        lines.append(
            f"| [{_escape(family['name'])}](#{family['slug']}) | {_status(family['selected_context'])} | "
            f"{_status(family['native_memory'])} | **{_escape(family['adapter_requirement'])}** | "
            f"{_escape(family['evidence'])} |"
        )
    lines.extend([
        "",
        "**Key:** ✅ available/validated · 🧪 partial or engine-dependent · ⏳ qualification pending.",
        "",
    ])
    for family in registry["families"]:
        lines.extend([
            f"## {family['name']} {{ #{family['slug']} }}",
            "",
            f"**Model types:** `{', '.join(family['model_types'])}`  ",
            f"**Adapter:** {family['adapter_requirement']}",
            "",
            family["adapter_detail"],
            "",
            "**Known examples and published bundles**",
            "",
        ])
        lines.extend([
            "| Model | PRA bundle/model card | Status | Validated engines | Recommended mode | Last qualification |",
            "| --- | --- | --- | --- | --- | --- |",
        ])
        bundle_links = family.get("bundle_links", {})
        for model in family["examples"]:
            bundle = bundle_links.get(model)
            if bundle:
                card = f"[{_escape(bundle['repo'])}]({_escape(bundle['url'])})"
                lines.append(
                    f"| `{_escape(model)}` | {card} | {_escape(bundle['status'])} | "
                    f"{_escape(bundle['engines'])} | {_escape(bundle['recommended_mode'])} | "
                    f"{_escape(bundle['qualified_at'])} |"
                )
            else:
                lines.append(
                    f"| `{_escape(model)}` | Not published | NOT_MEASURED | NOT_MEASURED | "
                    "Inspect and qualify locally | NOT_MEASURED |"
                )
        lines.extend(["", "**Inspect and launch**", "", "```bash"])
        lines.extend(family["commands"])
        lines.extend(["```", "", "**Evidence boundary**", "", family["evidence"], "", "**Limitations**", ""])
        lines.extend(f"- {item}" for item in family["limitations"])
        lines.append("")
    lines.extend([
        "## Adapter decision",
        "",
        "1. Start with `pra inspect MODEL --engine ENGINE`.",
        "2. If Selected Context is the target, no model structural adapter is required.",
        "3. For a built-in family, run `pra model validate`; exporting an adapter pins the mapping.",
        "4. For a partial or unknown family, run `pra model adapt` and the full validation ladder.",
        "5. Promote Native Memory only after quality, geometry, lifecycle, and economics pass",
        "   for the exact model revision, tokenizer, quantization, engine, and hardware.",
        "",
        f"_Generated from the model registry; evidence current through {registry['evidence_as_of']}._",
        "",
    ])
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
    files[MODEL_PAGE] = render_models(load_model_registry())
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
