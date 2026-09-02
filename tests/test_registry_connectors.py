from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pra_registry.connectors import FilesystemConnector, HuggingFaceConnector


def test_filesystem_connector_pins_manifest_digest(tmp_path: Path) -> None:
    (tmp_path / "bundle.yaml").write_text(
        "schema_version: 2\nbase_model:\n  id: org/model\n  revision: base-sha\nprofiles:\n  balanced: {}\n",
        encoding="utf-8",
    )
    model, bundle = FilesystemConnector().inspect(str(tmp_path))
    assert model.repo == "org/model"
    assert len(bundle.immutable_revision) == 64
    assert bundle.artifact_sources[0].source_type.value == "filesystem"


def test_hf_connector_records_resolved_sha_without_weight_download(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "bundle.yaml"
    manifest.write_text("base_model:\n  id: org/base\n  revision: base-sha\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda repo, filename, revision: calls.append((repo, filename, revision)) or str(manifest))
    api = SimpleNamespace(model_info=lambda repo, revision=None: SimpleNamespace(sha="hub-sha", private=False))
    model, bundle = HuggingFaceConnector(api).inspect("org/pra-base")
    assert model.repo == "org/base"
    assert bundle.immutable_revision == "hub-sha"
    assert calls == [("org/pra-base", "bundle.yaml", "hub-sha")]
