from __future__ import annotations

import re
import shlex
from pathlib import Path

from click.testing import CliRunner

from experiments.paper4_5_runtime.build_technical_site import (
    ROOT,
    generated_files,
    load_registry,
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
