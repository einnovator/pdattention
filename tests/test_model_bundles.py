from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner

from pra_hf.bundle import (
    BundleBuilder,
    BundleRegistryEntry,
    BundleResolver,
    BundleValidationError,
    HubBundleCatalog,
    HubPublisher,
    PRAModelBundle,
    TrustedBundleRegistry,
    validate_model_card,
    _tree_fingerprint,
)
from pra_hf.cli import cli


REVISION = "0123456789abcdef0123456789abcdef01234567"


def test_tree_fingerprint_has_platform_neutral_mixed_case_order(tmp_path: Path) -> None:
    component = tmp_path / "adapter"
    component.mkdir()
    for name, value in (
        ("README.md", b"card"),
        ("adapter_model.pt", b"weights"),
        ("config.json", b"{}"),
    ):
        (component / name).write_bytes(value)

    digest = hashlib.sha256()
    for name in ("adapter_model.pt", "config.json", "README.md"):
        digest.update(name.encode("utf-8"))
        digest.update(hashlib.sha256((component / name).read_bytes()).hexdigest().encode("ascii"))

    assert _tree_fingerprint(component) == digest.hexdigest()


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
        "provenance": {
            "pra_version": "0.2.0rc1",
            "pra_commit": "abc",
            "bundle_build_commit": "abc",
            "hf_collection": "EInnovator/pra-bundles-test",
        },
        "trust": {"publisher": "EInnovator"},
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
    assert "collections/EInnovator/pra-bundles-test" in text
    assert "huggingface.co/EInnovator" in text


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


def test_discovery_is_download_free_and_reports_exact_compatibility(
    monkeypatch,
) -> None:
    entry = BundleRegistryEntry(
        name="test", base_model="org/model", base_revision=REVISION,
        architecture="TestForCausalLM", bundle_repo="owner/pra-model",
        bundle_revision="fedcba", pra_version=">=0.2,<0.3", schema_version=2,
        trust="eInnovator-qualified", engine_compatibility={"hf": "validated"},
        profiles=["balanced"], qualification="CONTROLLED",
    )
    resolver = BundleResolver(TrustedBundleRegistry([entry]))
    monkeypatch.setattr(
        PRAModelBundle,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: pytest.fail("discovery downloaded a bundle")),
    )

    found = resolver.discover(model="org/model", model_revision=REVISION, engine="hf")
    mismatch = resolver.discover(model="org/model", model_revision="different", engine="hf")
    unresolved = resolver.resolve(
        "auto", model="org/model", model_revision="different", engine="hf"
    )

    assert found.status == "FOUND"
    assert found.compatibility == "exact"
    assert found.bundle_revision == "fedcba"
    assert mismatch.status == "INCOMPATIBLE"
    assert mismatch.compatibility == "base-revision-mismatch"
    assert unresolved.status == "INCOMPATIBLE"


def test_inspect_discovers_without_pull_and_explicit_auto_resolves(
    tmp_path: Path, monkeypatch
) -> None:
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
    metadata = {
        "model": {
            "id": "org/model", "revision": REVISION,
            "architecture": "TestForCausalLM", "family": "test",
            "variant": "base", "parameter_count_approx": 1,
        },
        "attention": {},
        "pra": {"structural_adapter": {"status": "VALIDATED"}},
    }
    monkeypatch.setattr("pra_hf.cli.ModelInspector.inspect", lambda *args, **kwargs: metadata)
    monkeypatch.setattr("pra_hf.cli.BundleResolver", lambda: resolver)
    runner = CliRunner()

    discovered = runner.invoke(cli, ["inspect", "org/model", "-e", "hf"])
    resolved = runner.invoke(cli, ["inspect", "org/model", "-e", "hf", "-a", "auto"])

    assert discovered.exit_code == 0, discovered.output
    assert "Published PRA bundle found" in discovered.output
    assert "Compatibility: exact" in discovered.output
    assert "Local path:" not in discovered.output
    assert resolved.exit_code == 0, resolved.output
    assert "PRA bundle resolution" in resolved.output
    assert "Status: RESOLVED" in resolved.output
    assert "Compatibility: exact" in resolved.output


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


def test_hf_list_filters_the_trusted_registry() -> None:
    result = CliRunner().invoke(
        cli,
        ["hf", "list", "--family", "qwen", "--engine", "MLX", "--json"],
    )

    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    assert value["source"] == "trusted-registry"
    assert value["count"] == 3
    assert all("qwen" in row["base_model"].lower() for row in value["bundles"])
    assert all("mlx" in row["engine_compatibility"] for row in value["bundles"])


def test_hub_catalog_search_marks_only_registry_entries_auto_resolvable() -> None:
    calls = []

    class FakeApi:
        def list_models(self, **kwargs):
            calls.append(kwargs)
            return [
                SimpleNamespace(
                    id="EInnovator/pra-qwen3-0.6b",
                    tags=["pra"],
                    cardData={"library_name": "pra", "base_model": "Qwen/Qwen3-0.6B"},
                    sha="hub-head",
                    downloads=12,
                    likes=3,
                    lastModified=None,
                    private=False,
                    gated=False,
                ),
                SimpleNamespace(
                    id="EInnovator/ordinary-model",
                    tags=["transformers"],
                    cardData={"library_name": "transformers"},
                    sha="ignored",
                ),
                SimpleNamespace(
                    id="EInnovator/pra-experimental",
                    tags=[],
                    cardData={"base_model": "org/experimental"},
                    sha="experimental-head",
                    downloads=1,
                    likes=0,
                    lastModified=None,
                    private=False,
                    gated=False,
                ),
            ]

    rows = HubBundleCatalog(api=FakeApi()).search("qwen", limit=10)

    assert calls[0]["author"] == "EInnovator"
    assert [row["repo_id"] for row in rows] == [
        "EInnovator/pra-qwen3-0.6b",
        "EInnovator/pra-experimental",
    ]
    assert rows[0]["auto_resolvable"] is True
    assert rows[0]["trust"] == "eInnovator-qualified"
    assert rows[1]["auto_resolvable"] is False
    assert rows[1]["trust"] == "hub-discovered"


def test_hf_search_cli_emits_normalized_live_results(monkeypatch) -> None:
    monkeypatch.setattr(
        "pra_hf.cli.HubBundleCatalog.search",
        lambda self, query, author, limit: [
            {
                "repo_id": "EInnovator/pra-test",
                "base_model": "org/test",
                "qualification": "CONTROLLED",
                "trust": "eInnovator-qualified",
                "auto_resolvable": True,
                "profiles": ["balanced"],
            }
        ],
    )

    result = CliRunner().invoke(
        cli, ["hf", "search", "test", "--limit", "5", "--json"]
    )

    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    assert value["source"] == "hugging-face-hub"
    assert value["author"] == "EInnovator"
    assert value["bundles"][0]["auto_resolvable"] is True


def test_hub_update_checks_remote_manifest_without_full_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    bundle_path = tmp_path / "bundle"
    built = BundleBuilder().build(_run(tmp_path), bundle_path)
    remote_manifest = tmp_path / "remote-bundle.yaml"
    remote_manifest.write_text(
        yaml.safe_dump({"base_model": built.base_model}), encoding="utf-8"
    )

    class FakeApi:
        def create_repo(self, *args, **kwargs):
            return None

        def upload_folder(self, **kwargs):
            return SimpleNamespace(oid="remote-commit", repo_url="https://example.test")

        def create_tag(self, *args, **kwargs):
            return None

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *args, **kwargs: str(remote_manifest)
    )
    result = HubPublisher().push(bundle_path, "org/pra-model")

    assert result["commit"] == "remote-commit"
    assert result["base_model"]["revision"] == REVISION


def test_default_registry_contains_published_cross_family_catalog() -> None:
    entries = {entry.name: entry for entry in TrustedBundleRegistry.default().entries}
    expected = {
        "pra-qwen3-4b-mlx-4bit": "49c18674ce15c8e267d5d19230d6dc152bca778b",
        "pra-qwen3-14b-mlx-4bit": "9853a17f84aeebc33e209c87e360715559b2c5c8",
        "pra-llama3-1-8b-mlx-4bit": "0d14b5eb65cefa56be0ff0c677818b8928d607a2",
        "pra-gemma3-1b-mlx-4bit": "afb67d45289bcffc180890089c2bfc71bb9ff636",
    }

    assert {name: entries[name].bundle_revision for name in expected} == expected
    assert all("qasper-learned" in entries[name].profiles for name in expected)
