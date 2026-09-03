"""SQLAlchemy persistence for the PRA Registry.

The schema intentionally stores policy and evidence payloads as JSON while
keeping identities, revisions, approvals, and relationships queryable. This
works unchanged on PostgreSQL and on SQLite for local development.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterator

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from sqlalchemy.pool import StaticPool


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class ModelRecord(TimestampMixin, Base):
    __tablename__ = "registry_models"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    provider: Mapped[str] = mapped_column(String(128), index=True)
    repo: Mapped[str] = mapped_column(String(512), index=True)
    revision: Mapped[str] = mapped_column(String(255))
    architecture: Mapped[str | None] = mapped_column(String(255))
    tokenizer: Mapped[str | None] = mapped_column(String(512))
    fingerprint: Mapped[str | None] = mapped_column(String(255))
    license_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    parameter_class: Mapped[str | None] = mapped_column(String(128))
    approval_state: Mapped[str] = mapped_column(String(32), default="DRAFT", index=True)
    owner: Mapped[str | None] = mapped_column(String(255))
    provenance: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    deleted: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


class BundleRecord(TimestampMixin, Base):
    __tablename__ = "registry_bundles"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    immutable_revision: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    base_model_id: Mapped[str] = mapped_column(ForeignKey("registry_models.id"), index=True)
    base_model_revision: Mapped[str] = mapped_column(String(255))
    schema_version: Mapped[int] = mapped_column(Integer, default=2)
    structural_adapter_status: Mapped[str] = mapped_column(String(64))
    learned_adapters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    profile_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    engine_compatibility: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    qualification_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    trust: Mapped[str] = mapped_column(String(128), index=True)
    publisher: Mapped[str | None] = mapped_column(String(255))
    checksums: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)
    lifecycle_state: Mapped[str] = mapped_column(String(64), default="active", index=True)
    approval_state: Mapped[str] = mapped_column(String(32), default="DRAFT", index=True)
    owner: Mapped[str | None] = mapped_column(String(255))
    provenance: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    sources: Mapped[list["ArtifactSourceRecord"]] = relationship(cascade="all, delete-orphan")


class ArtifactSourceRecord(TimestampMixin, Base):
    __tablename__ = "registry_artifact_sources"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    bundle_id: Mapped[str] = mapped_column(ForeignKey("registry_bundles.id"), index=True)
    source_type: Mapped[str] = mapped_column(String(64), index=True)
    locator: Mapped[str] = mapped_column(String(1024))
    credential_reference: Mapped[str | None] = mapped_column(String(255))
    immutable_revision: Mapped[str] = mapped_column(String(255))
    digest: Mapped[str | None] = mapped_column(String(255))
    mirrors: Mapped[list[str]] = mapped_column(JSON, default=list)


class ProfileRecord(TimestampMixin, Base):
    __tablename__ = "registry_profiles"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    version: Mapped[str] = mapped_column(String(128))
    immutable_revision: Mapped[str] = mapped_column(String(255), nullable=False)
    model_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    bundle_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    policy_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    qualification_status: Mapped[str] = mapped_column(String(64))
    approval_state: Mapped[str] = mapped_column(String(32), default="DRAFT", index=True)
    owner: Mapped[str | None] = mapped_column(String(255))
    provenance: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class CompatibilityRecord(TimestampMixin, Base):
    __tablename__ = "registry_engine_compatibility"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    engine: Mapped[str] = mapped_column(String(128), index=True)
    engine_version_range: Mapped[str] = mapped_column(String(128), default="*")
    model_id: Mapped[str | None] = mapped_column(String(255), index=True)
    bundle_id: Mapped[str | None] = mapped_column(String(255), index=True)
    execution_mode: Mapped[str] = mapped_column(String(64), index=True)
    mechanism_status: Mapped[str] = mapped_column(String(64))
    quality_status: Mapped[str] = mapped_column(String(64))
    economics_status: Mapped[str] = mapped_column(String(64))
    recommendation_status: Mapped[str] = mapped_column(String(64), index=True)
    limitations: Mapped[list[str]] = mapped_column(JSON, default=list)
    evidence_references: Mapped[list[str]] = mapped_column(JSON, default=list)


class QualificationRecord(TimestampMixin, Base):
    __tablename__ = "registry_qualifications"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    model_id: Mapped[str] = mapped_column(String(255), index=True)
    model_revision: Mapped[str] = mapped_column(String(255))
    bundle_id: Mapped[str | None] = mapped_column(String(255), index=True)
    bundle_revision: Mapped[str | None] = mapped_column(String(255))
    engine: Mapped[str] = mapped_column(String(128), index=True)
    engine_version: Mapped[str | None] = mapped_column(String(128))
    hardware: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    workload: Mapped[str] = mapped_column(String(255), index=True)
    profile_id: Mapped[str | None] = mapped_column(String(255), index=True)
    mode: Mapped[str] = mapped_column(String(64), index=True)
    cohort_size: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    quality_gate: Mapped[str] = mapped_column(String(64))
    recommendation: Mapped[str] = mapped_column(String(64))
    evidence_level: Mapped[str] = mapped_column(String(64))
    pra_commit: Mapped[str | None] = mapped_column(String(64))
    artifact_links: Mapped[list[str]] = mapped_column(JSON, default=list)
    annotations: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    finalized: Mapped[bool] = mapped_column(Boolean, default=True)


class DeploymentRecord(TimestampMixin, Base):
    __tablename__ = "registry_deployments"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    environment: Mapped[str] = mapped_column(String(128), index=True)
    cluster: Mapped[str] = mapped_column(String(255), index=True)
    engine_instance_selector: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    desired_model_id: Mapped[str] = mapped_column(String(255))
    desired_bundle_id: Mapped[str | None] = mapped_column(String(255))
    desired_profile_id: Mapped[str | None] = mapped_column(String(255))
    desired_mode: Mapped[str] = mapped_column(String(64))
    desired_models: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    allow_extra_models: Mapped[bool] = mapped_column(Boolean, default=True)
    storage_policy: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    observability_policy: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    desired_revision: Mapped[int] = mapped_column(Integer, default=1)
    approval_state: Mapped[str] = mapped_column(String(32), default="DRAFT", index=True)
    owner: Mapped[str | None] = mapped_column(String(255))
    provenance: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class ManagedInstanceRecord(TimestampMixin, Base):
    """Observed runtime identity and compact liveness state."""

    __tablename__ = "registry_managed_instances"
    instance_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    instance_type: Mapped[str] = mapped_column(String(32), index=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    environment: Mapped[str] = mapped_column(String(128), index=True)
    region: Mapped[str] = mapped_column(String(128), index=True)
    cluster: Mapped[str] = mapped_column(String(255), index=True)
    namespace: Mapped[str] = mapped_column(String(255), index=True)
    host: Mapped[str] = mapped_column(String(255), index=True)
    management_url: Mapped[str] = mapped_column(String(1024))
    inference_url: Mapped[str | None] = mapped_column(String(1024))
    pra_version: Mapped[str] = mapped_column(String(128))
    component_version: Mapped[str] = mapped_column(String(128))
    engine_kind: Mapped[str | None] = mapped_column(String(128), index=True)
    engine_version: Mapped[str | None] = mapped_column(String(128))
    health: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), index=True)
    started_at: Mapped[float] = mapped_column(Float)
    last_heartbeat: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    capabilities: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    models: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    runtime_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    observability: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    desired_revision: Mapped[int | None] = mapped_column(Integer)
    observed_revision: Mapped[int] = mapped_column(Integer, default=1)
    in_sync: Mapped[bool] = mapped_column(Boolean, default=True)
    drift_fields: Mapped[list[str]] = mapped_column(JSON, default=list)
    labels: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)
    metadata_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    registration_source: Mapped[str] = mapped_column(String(32))
    credential_identity: Mapped[str | None] = mapped_column(String(255))
    deregistered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PolicyRecord(TimestampMixin, Base):
    __tablename__ = "registry_policies"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    version: Mapped[str] = mapped_column(String(128))
    scope: Mapped[str] = mapped_column(String(255), index=True)
    selection_policy: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    storage_policy: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    session_policy: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    observability_policy: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    allowed_bundle_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    allowed_model_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    allowed_profile_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    approval_state: Mapped[str] = mapped_column(String(32), default="DRAFT", index=True)
    owner: Mapped[str | None] = mapped_column(String(255))
    provenance: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class RouterInstanceRecord(TimestampMixin, Base):
    """Desired and observed state for an external routing data plane."""

    __tablename__ = "registry_router_instances"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[str | None] = mapped_column(String(128))
    management_url: Mapped[str] = mapped_column(String(1024))
    inference_url: Mapped[str | None] = mapped_column(String(1024))
    credential_reference: Mapped[str | None] = mapped_column(String(255))
    region: Mapped[str] = mapped_column(String(128), index=True)
    cluster: Mapped[str] = mapped_column(String(255), index=True)
    health: Mapped[str] = mapped_column(String(64), index=True)
    desired_revision: Mapped[int] = mapped_column(Integer, default=1)
    observed_revision: Mapped[int] = mapped_column(Integer, default=0)
    supported_features: Mapped[list[str]] = mapped_column(JSON, default=list)
    labels: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)
    metadata_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    last_sync: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)


class RouteRecord(TimestampMixin, Base):
    __tablename__ = "registry_routes"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    public_model: Mapped[str] = mapped_column(String(512), index=True)
    route_kind: Mapped[str] = mapped_column(String(32), index=True)
    policy_id: Mapped[str] = mapped_column(String(255), index=True)
    pool_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    fallback_pool_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    tenant_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    metadata_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    desired_revision: Mapped[int] = mapped_column(Integer, default=1)


class ModelPoolRecord(TimestampMixin, Base):
    __tablename__ = "registry_model_pools"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    model_id: Mapped[str] = mapped_column(String(512), index=True)
    model_revision: Mapped[str | None] = mapped_column(String(255))
    selectors: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    metadata_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    desired_revision: Mapped[int] = mapped_column(Integer, default=1)


class BackendEndpointRecord(TimestampMixin, Base):
    __tablename__ = "registry_backend_endpoints"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    pool_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    engine_instance_id: Mapped[str | None] = mapped_column(String(255), index=True)
    runtime_model_id: Mapped[str] = mapped_column(String(255), default="default")
    inference_url: Mapped[str] = mapped_column(String(1024))
    engine: Mapped[str] = mapped_column(String(128), index=True)
    engine_version: Mapped[str | None] = mapped_column(String(128))
    model_id: Mapped[str] = mapped_column(String(512), index=True)
    model_revision: Mapped[str | None] = mapped_column(String(255))
    model_fingerprint: Mapped[str | None] = mapped_column(String(255))
    bundle_id: Mapped[str | None] = mapped_column(String(512), index=True)
    bundle_revision: Mapped[str | None] = mapped_column(String(255))
    profile: Mapped[str | None] = mapped_column(String(128), index=True)
    modes: Mapped[list[str]] = mapped_column(JSON, default=list)
    qualification_tier: Mapped[str] = mapped_column(String(64), index=True)
    approval_state: Mapped[str] = mapped_column(String(32), index=True)
    region: Mapped[str] = mapped_column(String(128), index=True)
    cluster: Mapped[str] = mapped_column(String(255), index=True)
    health: Mapped[str] = mapped_column(String(64), index=True)
    maintenance: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    cost: Mapped[float | None] = mapped_column(Float)
    labels: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)
    metadata_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class RoutingPolicyRecord(TimestampMixin, Base):
    __tablename__ = "registry_routing_policies"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    strategy: Mapped[str] = mapped_column(String(64), index=True)
    constraints: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    preferences: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    fallback: Mapped[list[str]] = mapped_column(JSON, default=list)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    metadata_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    desired_revision: Mapped[int] = mapped_column(Integer, default=1)


class RouteBindingRecord(TimestampMixin, Base):
    __tablename__ = "registry_route_bindings"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    route_id: Mapped[str] = mapped_column(String(255), index=True)
    router_id: Mapped[str] = mapped_column(String(255), index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    metadata_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    desired_revision: Mapped[int] = mapped_column(Integer, default=1)


class ApprovalRecord(Base):
    __tablename__ = "registry_approvals"
    sequence: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    resource_type: Mapped[str] = mapped_column(String(64), index=True)
    resource_id: Mapped[str] = mapped_column(String(255), index=True)
    version: Mapped[str] = mapped_column(String(255))
    state: Mapped[str] = mapped_column(String(32))
    approver: Mapped[str] = mapped_column(String(255))
    reason: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditRecord(Base):
    __tablename__ = "registry_audit_events"
    sequence: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    actor: Mapped[str] = mapped_column(String(255))
    action: Mapped[str] = mapped_column(String(128), index=True)
    resource_type: Mapped[str] = mapped_column(String(64), index=True)
    resource_id: Mapped[str] = mapped_column(String(255), index=True)
    version: Mapped[str | None] = mapped_column(String(255))
    before_summary: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    after_summary: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    trace_id: Mapped[str | None] = mapped_column(String(64), index=True)


class RegistryDatabase:
    """Own an engine and short-lived sessions without global connection state."""

    def __init__(self, database_url: str, *, create_schema: bool = False) -> None:
        kwargs: dict[str, Any] = {"pool_pre_ping": True}
        if database_url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        if database_url in {"sqlite://", "sqlite:///:memory:"}:
            kwargs["poolclass"] = StaticPool
        self.engine = create_engine(database_url, **kwargs)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)
        if create_schema:
            Base.metadata.create_all(self.engine)

    def session(self) -> Iterator[Session]:
        with self.session_factory() as session:
            yield session

    def close(self) -> None:
        self.engine.dispose()
