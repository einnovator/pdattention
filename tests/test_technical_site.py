from __future__ import annotations

import re
import shlex
from pathlib import Path

from click.testing import CliRunner

from experiments.paper4_5_runtime.build_technical_site import (
    ROOT,
    generated_files,
    load_model_registry,
    load_registry,
)
from experiments.paper4_5_runtime.build_cli_reference import (
    EXAMPLES as CLI_EXAMPLES,
    OUTPUTS as CLI_OUTPUTS,
    public_commands,
    render as render_cli_reference,
)
from pra_hf.cli import cli
from pra_hf.gateway_cli import resolve_gateway_mode


SITE = ROOT / "docs/site"
RESEARCH = SITE / "research"
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")
RESEARCH_LEVEL = re.compile(r"(?<![A-Za-z0-9_])E[0-3](?![A-Za-z0-9_])")


def _public_pages() -> list[Path]:
    return [path for path in SITE.rglob("*.md") if RESEARCH not in path.parents]


def test_generated_engine_pages_match_registry() -> None:
    registry = load_registry()
    for path, expected in generated_files(registry).items():
        assert path.read_text(encoding="utf-8") == expected


def test_generated_cli_reference_matches_public_click_tree() -> None:
    commands = public_commands()
    assert len(commands) >= 40
    assert set(commands) == set(CLI_EXAMPLES) == set(CLI_OUTPUTS)
    text = (SITE / "cli-reference.md").read_text(encoding="utf-8")
    assert text == render_cli_reference()
    for path in commands:
        assert f"### `{path}`" in text
    assert "--pra-level" not in text
    assert "--allow-unqualified-native" not in text


def test_engine_matrix_and_pages_are_product_self_contained() -> None:
    overview = (SITE / "engines/overview.md").read_text(encoding="utf-8")
    for marker in ("✅", "🧪", "⏳", "⛔", "## Key"):
        assert marker in overview
    for page in (SITE / "engines").glob("*.md"):
        if page.name == "overview.md":
            continue
        text = page.read_text(encoding="utf-8")
        for section in (
            "## What PRA adds to this engine", "## Install and launch",
            "### Command options", "## Metrics from the engine paper",
            "## Production recommendation",
        ):
            assert section in text, f"{page.name} is missing {section}"
        assert "pra evaluate MODEL" in text
        assert "--measurements RESULTS.json" in text


def test_model_page_matches_registry_and_explains_adapter_boundary() -> None:
    registry = load_model_registry()
    text = (SITE / "models.md").read_text(encoding="utf-8")
    assert "# Model Support" in text
    assert "## Adapter decision" in text
    assert "Selected Context" in text
    assert "Native Memory" in text
    for family in registry["families"]:
        assert f"## {family['name']}" in text
        assert family["adapter_requirement"] in text
        assert family["evidence"] in text


def test_web_ui_is_a_dedicated_operational_page() -> None:
    text = (SITE / "web-ui.md").read_text(encoding="utf-8")
    for value in (
        "pra agent start", "pra agent stop", "--detach", "--open",
        "/api/sessions", "/ws/sessions/{id}", "authentication layer",
    ):
        assert value in text
    mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert mkdocs.index("- CLI: cli.md") < mkdocs.index("- Concepts: concepts.md")
    assert "- Model Support: models.md" in mkdocs
    assert "- Web UI: web-ui.md" in mkdocs


def test_agent_guides_cover_gateway_and_direct_engine_paths() -> None:
    agents = SITE / "agents"
    expected = {
        "pra-agent.md",
        "deepseek-harness.md",
        "opencode.md",
        "pi-coding-agent.md",
        "openhands.md",
        "aider.md",
        "codex-cli.md",
        "claude-code.md",
    }
    section_pages = {
        "index.md", "benchmarks.md", "configuration.md", "mcp-control-plane.md",
        "tui.md", "slash-commands.md", "opencode-audit.md",
    }
    assert expected == {
        path.name for path in agents.glob("*.md") if path.name not in section_pages
    }
    for name in expected:
        text = (agents / name).read_text(encoding="utf-8")
        assert "## Through the PRA gateway" in text
        assert "## Direct PRA engine" in text
    overview = (agents / "index.md").read_text(encoding="utf-8")
    for name in expected:
        assert f"({name})" in overview
    mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "- Agents:" in mkdocs
    assert "- Benchmarks: agents/benchmarks.md" in mkdocs
    assert mkdocs.index("- Agents:") < mkdocs.index("- Model Support:")


def test_deepseek_and_pi_guides_explain_plugin_contract_and_setup() -> None:
    deepseek = (SITE / "agents/deepseek-harness.md").read_text(encoding="utf-8")
    pi = (SITE / "agents/pi-coding-agent.md").read_text(encoding="utf-8")
    for text, adapter in (
        (deepseek, "DeepSeekHarnessPRAAdapter"),
        (pi, "PiCodingAgentPRAAdapter"),
    ):
        assert "## What the PRA plugin does" in text
        assert "## Set up the PRA plugin bridge" in text
        assert adapter in text
        assert "not" in text.lower() and "published" in text.lower()


def test_all_internal_markdown_links_resolve() -> None:
    missing: list[str] = []
    for page in SITE.rglob("*.md"):
        for raw_target in MARKDOWN_LINK.findall(page.read_text(encoding="utf-8")):
            target = raw_target.split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (page.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{page.relative_to(ROOT)} -> {raw_target}")
    assert not missing, "Missing documentation links:\n" + "\n".join(missing)


def test_public_docs_use_product_terms_and_portable_links() -> None:
    violations: list[str] = []
    for page in _public_pages():
        text = page.read_text(encoding="utf-8")
        if RESEARCH_LEVEL.search(text):
            violations.append(f"paper-level integration label in {page.relative_to(ROOT)}")
        if "file:///" in text:
            violations.append(f"file URI in {page.relative_to(ROOT)}")
    assert not violations, "\n".join(violations)


def test_engine_claims_have_evidence_and_existing_provenance() -> None:
    registry = load_registry()
    labels = set(registry["evidence_labels"])
    missing_values = 0
    for engine in registry["engines"]:
        assert engine["evidence"] in labels
        assert engine["recommended_today"]
        for metric in engine["metrics"]:
            assert metric["status"] in labels
            assert (ROOT / metric["source"]).exists()
            if metric["value"] == "NOT_MEASURED":
                missing_values += 1
    overview = (SITE / "engines/overview.md").read_text(encoding="utf-8")
    assert missing_values > 0
    assert "Not measured" in overview
    assert "never interpreted as zero" in overview


def test_registry_quickstart_commands_exist_in_click_tree() -> None:
    registry = load_registry()
    for engine in registry["engines"]:
        for command in engine["quickstart"]:
            tokens = shlex.split(command, posix=True)
            assert tokens[0] == "pra"
            current = cli
            for token in tokens[1:]:
                if token.startswith("-") or not hasattr(current, "commands"):
                    break
                if token not in current.commands:
                    break
                current = current.commands[token]
            else:
                continue
            consumed = tokens[1 : 1 + (2 if len(tokens) > 2 else 1)]
            assert consumed[0] in cli.commands, f"Unknown quickstart: {command}"
            if len(consumed) > 1 and hasattr(cli.commands[consumed[0]], "commands"):
                assert consumed[1] in cli.commands[consumed[0]].commands, (
                    f"Unknown quickstart: {command}"
                )


def test_gateway_product_aliases_resolve_and_render_in_help() -> None:
    assert resolve_gateway_mode("passthrough") == "G00"
    assert resolve_gateway_mode("selected-context") == "G10"
    assert resolve_gateway_mode("upgrade") == "G01"
    assert resolve_gateway_mode("typed-transport") == "G11"
    result = CliRunner().invoke(cli, ["gateway", "serve", "--help"])
    assert result.exit_code == 0
    assert "selected-context" in result.output
    assert "typed-transport" in result.output


def test_gateway_page_explains_request_lifecycle_and_boundaries() -> None:
    text = (SITE / "deployment/gateway.md").read_text(encoding="utf-8")
    for value in (
        "## What the gateway does",
        "## Operating modes",
        "### Gateway and engine responsibilities",
        "## Sessions and resource deltas",
        "## Fallback placement",
        "Native K/V never crosses the gateway protocol",
        "Resources are consumed in",
        "request order",
        "DELETE /v1/pra/sessions/{session_id}",
    ):
        assert value in text


def test_static_html_build_never_contains_file_uris() -> None:
    built = ROOT / "site"
    if not built.is_dir():
        return
    violations = [
        str(path.relative_to(ROOT))
        for path in built.rglob("*.html")
        if "file:///" in path.read_text(encoding="utf-8")
    ]
    assert not violations
