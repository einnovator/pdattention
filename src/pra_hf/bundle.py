"""Portable, engine-neutral PRA model bundles."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class PRAModelBundle:
    """Resolved structural, learned, profile, evidence, and engine metadata."""

    base_model: Mapping[str, Any]
    structural_adapter: Mapping[str, Any]
    learned_adapters: Mapping[str, Any] = field(default_factory=dict)
    profiles: Mapping[str, Any] = field(default_factory=dict)
    benchmark_evidence: Mapping[str, Any] = field(default_factory=dict)
    runtime_compatibility: Mapping[str, Any] = field(default_factory=dict)
    engine_realizations: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    @classmethod
    def from_pretrained(cls, source: str | Path, *, revision: str | None = None) -> "PRAModelBundle":
        path = Path(source).expanduser()
        if not path.exists():
            try:
                from huggingface_hub import snapshot_download
            except ImportError as error:
                raise ImportError("Hub bundle resolution requires the 'hf-hub' optional dependency.") from error
            path = Path(snapshot_download(str(source), revision=revision))
        manifest = path / "bundle.yaml"
        if not manifest.is_file():
            manifest = path / "pra.yaml"
        value = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        return cls(
            base_model=value.get("base_model", value.get("model", {})),
            structural_adapter=value.get("structural_adapter", {}),
            learned_adapters=value.get("learned_adapters", {}),
            profiles=value.get("profiles", {}),
            benchmark_evidence=value.get("benchmark_evidence", {}),
            runtime_compatibility=value.get("runtime_compatibility", {}),
            engine_realizations=value.get("engine_realizations", {}),
            provenance=value.get("provenance", {}),
            schema_version=int(value.get("schema_version", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "base_model": dict(self.base_model),
            "structural_adapter": dict(self.structural_adapter),
            "learned_adapters": dict(self.learned_adapters),
            "profiles": dict(self.profiles),
            "benchmark_evidence": dict(self.benchmark_evidence),
            "runtime_compatibility": dict(self.runtime_compatibility),
            "engine_realizations": dict(self.engine_realizations),
            "provenance": dict(self.provenance),
        }

    def save_pretrained(self, directory: str | Path) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / "bundle.yaml"
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return path


class BundleBuilder:
    """Build a portable bundle from a completed onboarding/calibration run."""

    def build(self, run: str | Path, output: str | Path) -> PRAModelBundle:
        source = Path(run)
        runtime = yaml.safe_load((source / "pra.yaml").read_text(encoding="utf-8"))
        manifest_path = source / "manifest.json"
        provenance = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        bundle = PRAModelBundle(
            base_model=runtime.get("model", {}),
            structural_adapter=runtime.get("structural_adapter", {}),
            learned_adapters=runtime.get("learned_adapters", {}),
            profiles=runtime.get("profiles", {}),
            provenance=provenance,
        )
        target = Path(output)
        bundle.save_pretrained(target)
        adapter_source = source / "structural_adapter"
        if adapter_source.is_dir():
            shutil.copytree(adapter_source, target / "structural_adapter", dirs_exist_ok=True)
        (target / "README.md").write_text(self.model_card(bundle), encoding="utf-8")
        return bundle

    @staticmethod
    def model_card(bundle: PRAModelBundle) -> str:
        model = bundle.base_model.get("id", "unknown")
        revision = bundle.base_model.get("revision", "unresolved")
        return (
            "---\nlibrary_name: pra\n"
            f"base_model: {model}\n"
            f"base_model_revision: {revision}\n"
            "pra_schema_version: 1\ntags:\n  - pra\n---\n\n"
            f"# PRA bundle for {model}\n\n"
            "This bundle contains PRA adapters, profiles, evidence metadata, and provenance; it does not contain base-model weights.\n"
        )


class HubPublisher:
    """Optional Hugging Face Hub transport isolated from runtime integration."""

    def push(self, bundle: str | Path, repo_id: str, *, revision: str | None = None, dry_run: bool = False) -> dict[str, Any]:
        source = Path(bundle)
        files = sorted(str(path.relative_to(source)) for path in source.rglob("*") if path.is_file())
        if dry_run:
            return {"repo_id": repo_id, "revision": revision, "files": files, "dry_run": True}
        try:
            from huggingface_hub import HfApi
        except ImportError as error:
            raise ImportError("Publishing requires the 'hf-hub' optional dependency.") from error
        api = HfApi()
        api.create_repo(repo_id, exist_ok=True)
        api.upload_folder(repo_id=repo_id, folder_path=source, revision=revision)
        return {"repo_id": repo_id, "revision": revision, "files": files, "dry_run": False}

    def pull(self, repo_id: str, output: str | Path, *, revision: str | None = None) -> Path:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise ImportError("Hub pull requires the 'hf-hub' optional dependency.") from error
        return Path(snapshot_download(repo_id, revision=revision, local_dir=output))
