"""Metadata-only artifact connectors for PRA Registry ingestion."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Protocol

import yaml

from .contracts import ArtifactSourceCreate, ArtifactSourceType, BundleCreate, ModelCreate


class ArtifactConnector(Protocol):
    """Extension point for S3, OCI, Artifactory, MLflow, and private stores."""

    def inspect(self, locator: str, *, revision: str | None = None) -> tuple[ModelCreate, BundleCreate]: ...


def stable_id(prefix: str, locator: str, revision: str) -> str:
    readable = locator.lower().replace("/", "-").replace("_", "-")
    digest = hashlib.sha256(f"{locator}@{revision}".encode()).hexdigest()[:10]
    return f"{prefix}-{readable}-{digest}"


def _manifest_payload(value: Mapping[str, Any], *, locator: str, revision: str, source_type: ArtifactSourceType) -> tuple[ModelCreate, BundleCreate]:
    base = dict(value.get("base_model") or value.get("model") or {})
    model_repo = str(base.get("id") or base.get("repo") or "unknown/model")
    model_revision = str(base.get("revision") or "unresolved")
    model_id = stable_id("model", model_repo, model_revision)
    bundle_id = stable_id("bundle", locator, revision)
    model = ModelCreate(
        id=model_id,
        provider=model_repo.split("/", 1)[0] if "/" in model_repo else "local",
        repo=model_repo,
        revision=model_revision,
        architecture=base.get("architecture") or value.get("architecture"),
        tokenizer=base.get("tokenizer"),
        fingerprint=base.get("fingerprint"),
        license_metadata={"license": (value.get("provenance") or {}).get("license")},
        approval_state="CANDIDATE",
        provenance={"imported_from": locator},
    )
    profiles = value.get("profiles") or {}
    profile_ids = sorted(profiles.keys()) if isinstance(profiles, Mapping) else sorted(str(item) for item in profiles)
    trust_value = value.get("trust") or "hub-imported"
    trust = str(trust_value.get("level", "hub-imported")) if isinstance(trust_value, Mapping) else str(trust_value)
    bundle = BundleCreate(
        id=bundle_id,
        immutable_revision=revision,
        base_model_id=model_id,
        base_model_revision=model_revision,
        schema_version=int(value.get("schema_version", 1)),
        structural_adapter_status=str((value.get("structural_adapter") or {}).get("status", "AVAILABLE")),
        learned_adapters=dict(value.get("learned_adapters") or {}),
        profile_ids=profile_ids,
        engine_compatibility=dict(value.get("runtime_compatibility") or value.get("engine_compatibility") or {}),
        qualification_summary=dict(value.get("qualification") or value.get("benchmark_evidence") or {}),
        trust=trust,
        publisher=locator.split("/", 1)[0] if "/" in locator else None,
        checksums=dict(value.get("checksums") or {}),
        approval_state="CANDIDATE",
        provenance=dict(value.get("provenance") or {}),
        artifact_sources=[ArtifactSourceCreate(
            id=stable_id("source", locator, revision),
            source_type=source_type,
            locator=locator,
            immutable_revision=revision,
        )],
    )
    return model, bundle


class FilesystemConnector:
    """Read one local bundle manifest without copying its artifact payload."""

    def inspect(self, locator: str, *, revision: str | None = None) -> tuple[ModelCreate, BundleCreate]:
        directory = Path(locator).expanduser().resolve()
        manifest = next((directory / name for name in ("bundle.yaml", "bundle.yml", "manifest.yaml", "manifest.yml") if (directory / name).is_file()), None)
        if manifest is None:
            raise FileNotFoundError(f"No PRA bundle manifest found in {directory}")
        value = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        resolved = revision or hashlib.sha256(manifest.read_bytes()).hexdigest()
        return _manifest_payload(value, locator=str(directory), revision=resolved, source_type=ArtifactSourceType.FILESYSTEM)


class HuggingFaceConnector:
    """Inspect a Hub bundle manifest and pin the resolved commit SHA."""

    def __init__(self, api: Any | None = None) -> None:
        self.api = api

    def inspect(self, locator: str, *, revision: str | None = None) -> tuple[ModelCreate, BundleCreate]:
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError as error:
            raise ImportError("Hugging Face import requires the 'hf-hub' extra") from error
        api = self.api or HfApi()
        info = api.model_info(locator, revision=revision)
        resolved = str(info.sha)
        last_error: Exception | None = None
        for filename in ("bundle.yaml", "bundle.yml", "manifest.yaml", "manifest.yml"):
            try:
                path = hf_hub_download(locator, filename, revision=resolved)
                value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
                source_type = ArtifactSourceType.PRIVATE_HUGGINGFACE if bool(getattr(info, "private", False)) else ArtifactSourceType.HUGGINGFACE
                return _manifest_payload(value, locator=locator, revision=resolved, source_type=source_type)
            except Exception as error:
                last_error = error
        raise FileNotFoundError(f"No PRA bundle manifest found in {locator}@{resolved}") from last_error

    def collection_items(self, slug: str) -> list[str]:
        try:
            from huggingface_hub import HfApi
        except ImportError as error:
            raise ImportError("Hugging Face sync requires the 'hf-hub' extra") from error
        collection = (self.api or HfApi()).get_collection(slug)
        return [str(item.item_id) for item in collection.items if getattr(item, "item_type", None) == "model"]


CONNECTOR_TYPES: dict[str, type[ArtifactConnector]] = {
    "huggingface": HuggingFaceConnector,
    "private_huggingface": HuggingFaceConnector,
    "filesystem": FilesystemConnector,
}
