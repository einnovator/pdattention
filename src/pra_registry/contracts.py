"""Stable Pydantic contracts shared by the registry API and clients."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _contains_secret_field(value: Any) -> bool:
    forbidden = {"token", "secret", "password", "credential", "api_key", "authorization"}
    if isinstance(value, dict):
        return any(
            str(key).lower() in forbidden or _contains_secret_field(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_secret_field(item) for item in value)
    return False


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)


class ApprovalState(str, Enum):
    DRAFT = "DRAFT"
    CANDIDATE = "CANDIDATE"
    APPROVED = "APPROVED"
    DEPRECATED = "DEPRECATED"
    REVOKED = "REVOKED"


class ArtifactSourceType(str, Enum):
    HUGGINGFACE = "huggingface"
    PRIVATE_HUGGINGFACE = "private_huggingface"
    S3 = "s3"
    OCI = "oci"
    FILESYSTEM = "filesystem"
    OLLAMA = "ollama"
    CUSTOM = "custom"


class ManagedInstanceType(str, Enum):
    """Runtime kinds discoverable through the Registry."""

    ENGINE = "ENGINE"
    GATEWAY = "GATEWAY"


class ManagedInstanceStatus(str, Enum):
    """Registry-computed liveness state."""

    ONLINE = "ONLINE"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"


class RegistrationSource(str, Enum):
    SELF = "SELF"
    MANUAL = "MANUAL"
    DISCOVERY = "DISCOVERY"


class ResourceBase(StrictModel):
    id: str = Field(min_length=1, max_length=255)
    approval_state: ApprovalState = ApprovalState.DRAFT
    owner: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class ModelCreate(ResourceBase):
    provider: str
    repo: str
    revision: str
    architecture: str | None = None
    tokenizer: str | None = None
    fingerprint: str | None = None
    license_metadata: dict[str, Any] = Field(default_factory=dict)
    parameter_class: str | None = None


class ModelPatch(StrictModel):
    architecture: str | None = None
    tokenizer: str | None = None
    fingerprint: str | None = None
    license_metadata: dict[str, Any] | None = None
    parameter_class: str | None = None
    owner: str | None = None
    provenance: dict[str, Any] | None = None


class ArtifactSourceCreate(StrictModel):
    id: str
    source_type: ArtifactSourceType
    locator: str
    credential_reference: str | None = None
    immutable_revision: str
    digest: str | None = None
    mirrors: list[str] = Field(default_factory=list)

    @field_validator("credential_reference")
    @classmethod
    def reject_embedded_credentials(cls, value: str | None) -> str | None:
        if value and ("://" in value or value.lower().startswith(("token ", "bearer "))):
            raise ValueError("credential_reference must name a secret, not contain one")
        return value


class BundleCreate(ResourceBase):
    immutable_revision: str
    base_model_id: str
    base_model_revision: str
    schema_version: int = Field(default=2, ge=1)
    structural_adapter_status: str = "AVAILABLE"
    learned_adapters: dict[str, Any] = Field(default_factory=dict)
    profile_ids: list[str] = Field(default_factory=list)
    engine_compatibility: dict[str, Any] = Field(default_factory=dict)
    qualification_summary: dict[str, Any] = Field(default_factory=dict)
    trust: str = "unverified"
    publisher: str | None = None
    checksums: dict[str, str] = Field(default_factory=dict)
    lifecycle_state: str = "active"
    artifact_sources: list[ArtifactSourceCreate] = Field(default_factory=list)


class BundlePatch(StrictModel):
    profile_ids: list[str] | None = None
    engine_compatibility: dict[str, Any] | None = None
    qualification_summary: dict[str, Any] | None = None
    lifecycle_state: str | None = None
    owner: str | None = None


class ProfileCreate(ResourceBase):
    version: str
    immutable_revision: str
    model_ids: list[str] = Field(default_factory=list)
    bundle_ids: list[str] = Field(default_factory=list)
    policy_payload: dict[str, Any] = Field(default_factory=dict)
    qualification_status: str = "NOT_MEASURED"


class ProfilePatch(StrictModel):
    policy_payload: dict[str, Any] | None = None
    qualification_status: str | None = None
    owner: str | None = None


class CompatibilityCreate(StrictModel):
    id: str
    engine: str
    engine_version_range: str = "*"
    model_id: str | None = None
    bundle_id: str | None = None
    execution_mode: str
    mechanism_status: str = "NOT_MEASURED"
    quality_status: str = "NOT_MEASURED"
    economics_status: str = "NOT_MEASURED"
    recommendation_status: str = "CALIBRATION_PENDING"
    limitations: list[str] = Field(default_factory=list)
    evidence_references: list[str] = Field(default_factory=list)


class QualificationCreate(StrictModel):
    id: str
    model_id: str
    model_revision: str
    bundle_id: str | None = None
    bundle_revision: str | None = None
    engine: str
    engine_version: str | None = None
    hardware: dict[str, Any] = Field(default_factory=dict)
    workload: str
    profile_id: str | None = None
    mode: str
    cohort_size: int = Field(ge=0)
    metrics: dict[str, Any] = Field(default_factory=dict)
    quality_gate: str = "NOT_MEASURED"
    recommendation: str = "CALIBRATION_PENDING"
    evidence_level: str = "SMOKE"
    pra_commit: str | None = None
    artifact_links: list[str] = Field(default_factory=list)
    annotations: dict[str, Any] = Field(default_factory=dict)


class DesiredModel(StrictModel):
    """One model-specific desired state inside an engine deployment."""

    runtime_model_id: str = "default"
    model_id: str
    bundle_id: str | None = None
    profile_id: str | None = None
    mode: str = "E0"


class DeploymentCreate(ResourceBase):
    environment: str
    cluster: str
    engine_instance_selector: dict[str, Any] = Field(default_factory=dict)
    desired_model_id: str | None = None
    desired_bundle_id: str | None = None
    desired_profile_id: str | None = None
    desired_mode: str = "E0"
    desired_models: list[DesiredModel] = Field(default_factory=list)
    allow_extra_models: bool = True
    storage_policy: dict[str, Any] = Field(default_factory=dict)
    observability_policy: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize_desired_models(self) -> "DeploymentCreate":
        if not self.desired_models and not self.desired_model_id:
            raise ValueError("desired_model_id or desired_models is required")
        if not self.desired_models:
            self.desired_models = [DesiredModel(
                runtime_model_id="default",
                model_id=str(self.desired_model_id),
                bundle_id=self.desired_bundle_id,
                profile_id=self.desired_profile_id,
                mode=self.desired_mode,
            )]
        runtime_ids = [row.runtime_model_id for row in self.desired_models]
        if len(runtime_ids) != len(set(runtime_ids)):
            raise ValueError("desired_models runtime_model_id values must be unique")
        if self.desired_model_id is None:
            first = self.desired_models[0]
            self.desired_model_id = first.model_id
            self.desired_bundle_id = first.bundle_id
            self.desired_profile_id = first.profile_id
            self.desired_mode = first.mode
        return self


class DeploymentPatch(StrictModel):
    engine_instance_selector: dict[str, Any] | None = None
    desired_model_id: str | None = None
    desired_bundle_id: str | None = None
    desired_profile_id: str | None = None
    desired_mode: str | None = None
    desired_models: list[DesiredModel] | None = None
    allow_extra_models: bool | None = None
    storage_policy: dict[str, Any] | None = None
    observability_policy: dict[str, Any] | None = None
    owner: str | None = None


class ManagedInstanceRegister(StrictModel):
    """Idempotent self-registration payload for one runtime instance."""

    instance_id: str = Field(min_length=1, max_length=255)
    instance_type: ManagedInstanceType
    name: str = Field(min_length=1, max_length=255)
    environment: str = "development"
    region: str = "local"
    cluster: str = "default"
    namespace: str = "default"
    host: str
    management_url: str
    inference_url: str | None = None
    pra_version: str
    component_version: str
    engine_kind: str | None = None
    engine_version: str | None = None
    health: str = "healthy"
    started_at: float
    capabilities: dict[str, Any] = Field(default_factory=dict)
    models: list[dict[str, Any]] = Field(default_factory=list)
    runtime_summary: dict[str, Any] = Field(default_factory=dict)
    observability: dict[str, Any] = Field(default_factory=dict)
    desired_revision: int | None = None
    observed_revision: int = Field(default=1, ge=0)
    in_sync: bool = True
    drift_fields: list[str] = Field(default_factory=list)
    labels: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    registration_source: RegistrationSource = RegistrationSource.SELF

    @field_validator("management_url", "inference_url")
    @classmethod
    def require_http_url(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith(("http://", "https://")):
            raise ValueError("runtime URLs must use http:// or https://")
        return value.rstrip("/") if value else value

    @field_validator("metadata")
    @classmethod
    def reject_secret_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        if _contains_secret_field(value):
            raise ValueError("credentials must not appear in instance metadata")
        return value

    @model_validator(mode="after")
    def reject_secret_observed_state(self) -> "ManagedInstanceRegister":
        if _contains_secret_field({
            "capabilities": self.capabilities, "models": self.models,
            "runtime_summary": self.runtime_summary, "observability": self.observability,
            "metadata": self.metadata,
        }):
            raise ValueError("credentials must not appear in instance observed state")
        return self


class ManagedInstanceHeartbeat(StrictModel):
    health: str = "healthy"
    uptime_seconds: float = Field(ge=0)
    observed_revision: int = Field(ge=0)
    runtime_summary: dict[str, Any] = Field(default_factory=dict)
    capabilities: dict[str, Any] | None = None
    timestamp: float


class ManagedInstanceObservedPatch(StrictModel):
    health: str | None = None
    management_url: str | None = None
    inference_url: str | None = None
    component_version: str | None = None
    engine_version: str | None = None
    capabilities: dict[str, Any] | None = None
    models: list[dict[str, Any]] | None = None
    runtime_summary: dict[str, Any] | None = None
    observability: dict[str, Any] | None = None
    observed_revision: int | None = Field(default=None, ge=0)
    in_sync: bool | None = None
    drift_fields: list[str] | None = None
    labels: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("management_url", "inference_url")
    @classmethod
    def require_http_url(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith(("http://", "https://")):
            raise ValueError("runtime URLs must use http:// or https://")
        return value.rstrip("/") if value else value

    @field_validator("metadata")
    @classmethod
    def reject_secret_metadata(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        if _contains_secret_field(value):
            raise ValueError("credentials must not appear in instance metadata")
        return value

    @model_validator(mode="after")
    def reject_secret_observed_state(self) -> "ManagedInstanceObservedPatch":
        values = self.model_dump(exclude_none=True)
        if _contains_secret_field(values):
            raise ValueError("credentials must not appear in instance observed state")
        return self


class PolicyCreate(ResourceBase):
    version: str
    scope: str
    selection_policy: dict[str, Any] = Field(default_factory=dict)
    storage_policy: dict[str, Any] = Field(default_factory=dict)
    session_policy: dict[str, Any] = Field(default_factory=dict)
    observability_policy: dict[str, Any] = Field(default_factory=dict)
    allowed_bundle_ids: list[str] = Field(default_factory=list)
    allowed_model_ids: list[str] = Field(default_factory=list)
    allowed_profile_ids: list[str] = Field(default_factory=list)


class PolicyPatch(StrictModel):
    selection_policy: dict[str, Any] | None = None
    storage_policy: dict[str, Any] | None = None
    session_policy: dict[str, Any] | None = None
    observability_policy: dict[str, Any] | None = None
    allowed_bundle_ids: list[str] | None = None
    allowed_model_ids: list[str] | None = None
    allowed_profile_ids: list[str] | None = None


class ApprovalCreate(StrictModel):
    resource_type: str
    resource_id: str
    version: str
    state: ApprovalState
    approver: str
    reason: str


class BundleResolveRequest(StrictModel):
    model: str
    model_revision: str | None = None
    engine: str | None = None
    engine_version: str | None = None
    hardware: dict[str, Any] = Field(default_factory=dict)
    trust: str | None = None


class ProfileResolveRequest(StrictModel):
    model_id: str
    bundle_id: str | None = None
    engine: str | None = None
    workload: str | None = None


class DeploymentResolveRequest(StrictModel):
    environment: str
    cluster: str
    engine_labels: dict[str, str] = Field(default_factory=dict)


class HuggingFaceImportRequest(StrictModel):
    repo_id: str
    revision: str | None = None


class HuggingFaceCollectionSyncRequest(StrictModel):
    collection: str


class Page(StrictModel):
    items: list[dict[str, Any]]
    total: int
    limit: int
    offset: int
