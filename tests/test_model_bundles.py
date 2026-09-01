from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from pra_hf.bundle import (
    BundleBuilder,
    BundleRegistryEntry,
    BundleResolver,
    BundleValidationError,
    PRAModelBundle,
    TrustedBundleRegistry,
    validate_model_card,
)
from pra_hf.cli import cli


REVISION = "0123456789abcdef0123456789abcdef01234567"


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    structural = run / "structural_adapter"
    learned = run / "learned_adapters" / "router-v1"
    structural.mkdir(parents=True)
    learned.mkdir(parents=True)
    (structural / "pra_adapter.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    (learned / "config.json").write_text(json.dumps({"type": "routing"}), encoding="utf-8")
    (learned / "adapter_model.pt").write_bytes(b"weights")
    qualification = run / "qualification"
    qualification.mkdir()
    (qualification / "profile_evidence.json").write_text(
        json.dumps({"evidence_tier": "SMOKE"}), encoding="utf-8"
    )
    payload = {
        "schema_version": 2,
        "base_model": {
            "id": "org/model",
            "revision": REVISION,
            "architecture": "TestForCausalLM",
        },
        "structural_adapter": {"path": "structural_adapter", "status": "validated"},
        "learned_adapters": {
            "router-v1": {
                "path": "learned_adapters/router-v1",
                "type": "routing",
                "status": "validated",
                "default": True,
            }
        },
        "profiles": {
            "balanced": {
                "purpose": "test",
                "routing_adapter": "router-v1",
                "status": "validated",
            }
        },
        "runtime_compatibility": {
            "hf": {
                "selected_context": "validated",
                "native_memory": "validated",
                "native_serving": "NOT_MEASURED",
                "recommended": "Selected Context",
            }
        },
        "qualification": {
            "status": "CONTROLLED",
            "metrics": [],
            "limitations": ["test fixture"],
        },
        "provenance": {"pra_version": "0.2.0rc1", "pra_commit": "abc", "bundle_build_commit": "abc"},
    }
    (run / "pra.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return run


def test_builder_packages_every_component_and_preserves_fingerprints(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bundle"
    built = BundleBuilder().build(_run(tmp_path), bundle_path)

    assert built.schema_version == 2
    assert (bundle_path / "structural_adapter/pra_adapter.yaml").is_file()
    assert (bundle_path / "learned_adapters/router-v1/adapter_model.pt").is_file()
    assert (bundle_path / "qualification/profile_evidence.json").is_file()
    assert "qualification/profile_evidence.json" in built.checksums
    assert len(built.structural_adapter["fingerprint"]) == 64
    assert len(built.learned_adapters["router-v1"]["fingerprint"]) == 64
    assert built.validate()["status"] == "VALID"
    assert built.selected_learned_adapters("balanced")["router-v1"].is_dir()


def test_builder_rejects_missing_and_absolute_component_paths(tmp_path: Path) -> None:
    run = _run(tmp_path)
    payload = yaml.safe_load((run / "pra.yaml").read_text(encoding="utf-8"))
    payload["learned_adapters"]["router-v1"]["path"] = "missing"
    (run / "pra.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(BundleValidationError, match="Missing learned adapter"):
        BundleBuilder().build(run, tmp_path / "missing")

    payload["learned_adapters"]["router-v1"]["path"] = "C:/private/router"
    (run / "pra.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(BundleValidationError, match="absolute Windows path"):
        BundleBuilder().build(run, tmp_path / "absolute")


def test_validation_detects_payload_tampering(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bundle"
    BundleBuilder().build(_run(tmp_path), bundle_path)
    (bundle_path / "learned_adapters/router-v1/adapter_model.pt").write_bytes(b"changed")

    with pytest.raises(BundleValidationError, match="fingerprint mismatch|checksum mismatch"):
        PRAModelBundle.from_pretrained(bundle_path)


def test_generated_card_is_complete_and_uses_public_terms(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bundle"
    BundleBuilder().build(_run(tmp_path), bundle_path)
    text = (bundle_path / "README.md").read_text(encoding="utf-8")

    metadata = validate_model_card(text, PRAModelBundle.from_pretrained(bundle_path))
    assert metadata["base_model"] == "org/model"
    assert "Native Memory" in text
    assert " E2 " not in text


def test_resolver_supports_none_local_and_trusted_auto(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bundle"
    BundleBuilder().build(_run(tmp_path), bundle_path)
    entry = BundleRegistryEntry(
        name="test", base_model="org/model", base_revision=REVISION,
        architecture="TestForCausalLM", bundle_repo=str(bundle_path),
        bundle_revision="fedcba", pra_version=">=0.2,<0.3", schema_version=2,
        trust="eInnovator-qualified", engine_compatibility={"hf": "validated"},
        profiles=["balanced"], qualification="CONTROLLED",
    )
    resolver = BundleResolver(TrustedBundleRegistry([entry]))

    assert resolver.resolve("none", model="org/model").status == "DISABLED"
    assert resolver.resolve(str(bundle_path), model="org/model").trust == "local/private"
    automatic = resolver.resolve("auto", model="org/model", model_revision=REVISION, engine="hf")
    assert automatic.status == "RESOLVED"
    assert automatic.resolved_revision == "fedcba"


def test_untrusted_community_bundle_is_explicit_only(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bundle"
    BundleBuilder().build(_run(tmp_path), bundle_path)
    registry = TrustedBundleRegistry([
        BundleRegistryEntry(
            name="community", base_model="org/model", base_revision=REVISION,
            architecture="TestForCausalLM", bundle_repo=str(bundle_path),
            bundle_revision="abc", pra_version=">=0.2,<0.3", schema_version=2,
            trust="community", engine_compatibility={"hf": "validated"},
            profiles=["balanced"], qualification="CONTROLLED",
        )
    ])
    assert BundleResolver(registry).resolve("auto", model="org/model", engine="hf").status == "NO_TRUSTED_MATCH"
    assert BundleResolver(registry).resolve(str(bundle_path), model="org/model").status == "RESOLVED"


def test_bundle_cli_card_list_resolve_and_hf_dry_run(tmp_path: Path, monkeypatch) -> None:
    bundle_path = tmp_path / "bundle"
    BundleBuilder().build(_run(tmp_path), bundle_path)
    runner = CliRunner()

    assert runner.invoke(cli, ["bundle", "card", str(bundle_path), "--update"]).exit_code == 0
    assert runner.invoke(cli, ["bundle", "list", "--json"]).exit_code == 0
    disabled = runner.invoke(cli, ["bundle", "resolve", "org/model", "-a", "none", "--json"])
    assert json.loads(disabled.output)["status"] == "DISABLED"
    dry = runner.invoke(cli, ["hf", "push", str(bundle_path), "owner/repo", "--dry-run", "--json"])
    assert dry.exit_code == 0, dry.output
    assert json.loads(dry.output)["schema_version"] == 2
