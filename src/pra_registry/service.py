"""Transactional registry operations and deterministic resolution policy."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
import json
from typing import Any, TypeVar

from sqlalchemy import func, inspect, select
from sqlalchemy.orm import Session
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from .contracts import (
    ApprovalCreate,
    ApprovalState,
    BundleCreate,
    BundleResolveRequest,
    CompatibilityCreate,
    DeploymentCreate,
    DeploymentPatch,
    DeploymentResolveRequest,
    ManagedInstanceHeartbeat,
    ManagedInstanceObservedPatch,
    ManagedInstanceRegister,
    ManagedInstanceStatus,
    ModelCreate,
    ModelPatch,
    PolicyCreate,
    PolicyPatch,
    ProfileCreate,
    ProfilePatch,
    ProfileResolveRequest,
    QualificationCreate,
    BackendEndpointCreate,
    BackendEndpointPatch,
    ModelPoolCreate,
    ModelPoolPatch,
    RouteBindingCreate,
    RouteBindingPatch,
    RouteCreate,
    RoutePatch,
    RouterInstanceCreate,
    RouterInstancePatch,
    RoutingPolicyCreate,
    RoutingPolicyPatch,
)
from .database import (
    ApprovalRecord,
    ArtifactSourceRecord,
    AuditRecord,
    BundleRecord,
    CompatibilityRecord,
    DeploymentRecord,
    ManagedInstanceRecord,
    ModelRecord,
    PolicyRecord,
    ProfileRecord,
    QualificationRecord,
    BackendEndpointRecord,
    ModelPoolRecord,
    RouteBindingRecord,
    RouteRecord,
    RouterInstanceRecord,
    RoutingPolicyRecord,
)


class RegistryError(RuntimeError):
    status_code = 400


class RegistryNotFound(RegistryError):
    status_code = 404


class RegistryConflict(RegistryError):
    status_code = 409


Record = TypeVar("Record")


def record_dict(record: Any) -> dict[str, Any]:
    return {column.key: getattr(record, column.key) for column in inspect(record).mapper.column_attrs}


RESOURCE_TABLES = {
    "model": ModelRecord,
    "bundle": BundleRecord,
    "profile": ProfileRecord,
    "deployment": DeploymentRecord,
    "policy": PolicyRecord,
}


QUALIFICATION_ORDER = {
    "NOT_MEASURED": 0,
    "SMOKE": 1,
    "RESEARCH": 2,
    "CONTROLLED": 3,
    "ENGINE_QUALIFIED": 4,
    "PRODUCTION_QUALIFIED": 5,
}


def _routing_payload(value: Any, *, patch: bool = False) -> dict[str, Any]:
    payload = value.model_dump(mode="json", exclude_unset=patch)
    if "metadata" in payload:
        payload["metadata_payload"] = payload.pop("metadata")
    return payload


def _routing_record(record: Any) -> dict[str, Any]:
    value = record_dict(record)
    if "metadata_payload" in value:
        value["metadata"] = value.pop("metadata_payload")
    return value


class RegistryService:
    """Apply registry invariants inside a caller-owned SQLAlchemy session."""

    def __init__(self, session: Session, *, actor: str = "registry") -> None:
        self.session = session
        self.actor = actor

    def _get(self, table: type[Record], resource_id: str) -> Record:
        value = self.session.get(table, resource_id)
        if value is None:
            raise RegistryNotFound(f"{table.__tablename__} resource {resource_id!r} was not found")
        return value

    def _create(self, table: type[Record], payload: Mapping[str, Any], resource_type: str) -> dict[str, Any]:
        resource_id = str(payload["id"])
        if self.session.get(table, resource_id) is not None:
            raise RegistryConflict(f"{resource_type} {resource_id!r} already exists")
        row = table(**dict(payload))
        self.session.add(row)
        self.session.flush()
        self._audit("create", resource_type, resource_id, None, record_dict(row))
        self.session.commit()
        return record_dict(row)

    def _patch(self, table: type[Record], resource_id: str, changes: Mapping[str, Any], resource_type: str) -> dict[str, Any]:
        row = self._get(table, resource_id)
        before = record_dict(row)
        for key, value in changes.items():
            setattr(row, key, value)
        self.session.flush()
        self._audit("patch", resource_type, resource_id, before, record_dict(row))
        self.session.commit()
        return record_dict(row)

    def _patch_routing(self, table: type[Record], resource_id: str, value: Any, resource_type: str) -> dict[str, Any]:
        row = self._get(table, resource_id)
        before = _routing_record(row)
        for key, item in _routing_payload(value, patch=True).items():
            setattr(row, key, item)
        if hasattr(row, "desired_revision") and "desired_revision" not in value.model_fields_set:
            row.desired_revision += 1
        self._touch_bound_routers(resource_type, row)
        self.session.flush()
        self._audit("patch", resource_type, resource_id, before, _routing_record(row))
        self.session.commit()
        return _routing_record(row)

    def _list(
        self, table: type[Record], *, limit: int, offset: int, filters: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        filters = {key: value for key, value in (filters or {}).items() if value is not None}
        query = select(table)
        count_query = select(func.count()).select_from(table)
        for key, value in filters.items():
            query = query.where(getattr(table, key) == value)
            count_query = count_query.where(getattr(table, key) == value)
        if table in {ApprovalRecord, AuditRecord}:
            query = query.order_by(getattr(table, "sequence"))
        elif hasattr(table, "created_at"):
            query = query.order_by(getattr(table, "created_at").desc(), getattr(table, "id"))
        items = self.session.scalars(query.offset(offset).limit(limit)).all()
        return {
            "items": [record_dict(item) for item in items],
            "total": self.session.scalar(count_query) or 0,
            "limit": limit,
            "offset": offset,
        }

    def _list_routing(
        self, table: type[Record], *, limit: int, offset: int,
        filters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        filters = {key: value for key, value in (filters or {}).items() if value is not None}
        query = select(table)
        count_query = select(func.count()).select_from(table)
        for key, value in filters.items():
            query = query.where(getattr(table, key) == value)
            count_query = count_query.where(getattr(table, key) == value)
        query = query.order_by(table.id)
        rows = self.session.scalars(query.offset(offset).limit(limit)).all()
        return {
            "items": [_routing_record(row) for row in rows],
            "total": self.session.scalar(count_query) or 0,
            "limit": limit,
            "offset": offset,
        }

    def _audit(
        self, action: str, resource_type: str, resource_id: str,
        before: dict[str, Any] | None, after: dict[str, Any] | None,
        *, version: str | None = None, trace_id: str | None = None,
    ) -> None:
        # Normalize datetime/enums before JSON persistence and exclude credentials.
        before = self._audit_summary(before)
        after = self._audit_summary(after)
        self.session.add(AuditRecord(
            actor=self.actor, action=action, resource_type=resource_type,
            resource_id=resource_id, version=version,
            before_summary=before, after_summary=after, trace_id=trace_id,
        ))

    @staticmethod
    def _audit_summary(value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        safe = json.loads(json.dumps(value, default=str))
        safe.pop("credential_reference", None)
        safe.pop("sources", None)
        return safe

    # Models
    def create_model(self, value: ModelCreate) -> dict[str, Any]:
        return self._create(ModelRecord, value.model_dump(mode="json"), "model")

    def import_bundle_metadata(self, model: ModelCreate, bundle: BundleCreate) -> dict[str, Any]:
        """Idempotently register connector-normalized metadata at pinned revisions."""

        existing_model = self.session.get(ModelRecord, model.id)
        model_result = record_dict(existing_model) if existing_model else self.create_model(model)
        existing_bundle = self.session.get(BundleRecord, bundle.id)
        if existing_bundle:
            if existing_bundle.immutable_revision != bundle.immutable_revision:
                raise RegistryConflict("bundle identity already refers to another immutable revision")
            bundle_result = self.get_bundle(bundle.id)
        else:
            bundle_result = self.create_bundle(bundle)
        return {"model": model_result, "bundle": bundle_result}

    def get_model(self, resource_id: str) -> dict[str, Any]:
        return record_dict(self._get(ModelRecord, resource_id))

    def list_models(self, limit: int, offset: int, *, provider: str | None = None, state: str | None = None) -> dict[str, Any]:
        return self._list(ModelRecord, limit=limit, offset=offset, filters={"provider": provider, "approval_state": state})

    def patch_model(self, resource_id: str, value: ModelPatch) -> dict[str, Any]:
        return self._patch(ModelRecord, resource_id, value.model_dump(exclude_unset=True), "model")

    def delete_model(self, resource_id: str) -> dict[str, Any]:
        row = self._get(ModelRecord, resource_id)
        before = record_dict(row)
        row.deleted = True
        row.approval_state = ApprovalState.DEPRECATED.value
        self._audit("soft_delete", "model", resource_id, before, record_dict(row))
        self.session.commit()
        return record_dict(row)

    # Bundles and immutable artifact sources
    def create_bundle(self, value: BundleCreate) -> dict[str, Any]:
        if self.session.get(ModelRecord, value.base_model_id) is None:
            raise RegistryConflict("base_model_id must reference a registered model")
        if self.session.get(BundleRecord, value.id) is not None:
            raise RegistryConflict(f"bundle {value.id!r} already exists")
        for source in value.artifact_sources:
            if self.session.get(ArtifactSourceRecord, source.id):
                raise RegistryConflict(f"artifact source {source.id!r} already exists")
        payload = value.model_dump(mode="json", exclude={"artifact_sources"})
        row = BundleRecord(**payload)
        self.session.add(row)
        self.session.flush()
        for source in value.artifact_sources:
            self.session.add(ArtifactSourceRecord(bundle_id=value.id, **source.model_dump(mode="json")))
        self._audit("create", "bundle", value.id, None, record_dict(row))
        self.session.commit()
        return self.get_bundle(value.id)

    def get_bundle(self, resource_id: str) -> dict[str, Any]:
        row = self._get(BundleRecord, resource_id)
        value = record_dict(row)
        value["artifact_sources"] = [record_dict(source) for source in row.sources]
        return value

    def list_bundles(self, limit: int, offset: int, **filters: Any) -> dict[str, Any]:
        return self._list(BundleRecord, limit=limit, offset=offset, filters=filters)

    def patch_bundle(self, resource_id: str, value: Any) -> dict[str, Any]:
        return self._patch(BundleRecord, resource_id, value.model_dump(exclude_unset=True), "bundle")

    # Profiles
    def create_profile(self, value: ProfileCreate) -> dict[str, Any]:
        return self._create(ProfileRecord, value.model_dump(mode="json"), "profile")

    def get_profile(self, resource_id: str) -> dict[str, Any]:
        return record_dict(self._get(ProfileRecord, resource_id))

    def list_profiles(self, limit: int, offset: int, **filters: Any) -> dict[str, Any]:
        return self._list(ProfileRecord, limit=limit, offset=offset, filters=filters)

    def patch_profile(self, resource_id: str, value: ProfilePatch) -> dict[str, Any]:
        row = self._get(ProfileRecord, resource_id)
        if row.approval_state == ApprovalState.APPROVED.value:
            raise RegistryConflict("approved profiles are immutable; create a new version")
        return self._patch(ProfileRecord, resource_id, value.model_dump(exclude_unset=True), "profile")

    # Compatibility and immutable qualifications
    def create_compatibility(self, value: CompatibilityCreate) -> dict[str, Any]:
        return self._create(CompatibilityRecord, value.model_dump(mode="json"), "compatibility")

    def list_compatibility(self, limit: int, offset: int, **filters: Any) -> dict[str, Any]:
        return self._list(CompatibilityRecord, limit=limit, offset=offset, filters=filters)

    def resolve_compatibility(
        self, *, model_id: str | None = None, model_revision: str | None = None,
        bundle_id: str | None = None, engine: str | None = None,
        engine_version: str | None = None, hardware_class: str | None = None,
        execution_mode: str | None = None,
    ) -> dict[str, Any]:
        rows = self.session.scalars(select(CompatibilityRecord)).all()
        rows = [row for row in rows if (
            (not model_id or row.model_id == model_id)
            and (not bundle_id or row.bundle_id == bundle_id)
            and (not engine or row.engine == engine)
            and (not execution_mode or row.execution_mode == execution_mode)
            and _version_matches(engine_version, row.engine_version_range)
        )]
        qualifications = self.session.scalars(select(QualificationRecord)).all()
        evidence = []
        for item in qualifications:
            accelerator = str((item.hardware or {}).get("accelerator", (item.hardware or {}).get("class", "")))
            if (
                (not model_id or item.model_id == model_id)
                and (not model_revision or item.model_revision == model_revision)
                and (not bundle_id or item.bundle_id == bundle_id)
                and (not engine or item.engine == engine)
                and (not execution_mode or item.mode == execution_mode)
                and (not hardware_class or hardware_class.lower() in accelerator.lower())
            ):
                evidence.append(record_dict(item))
        return {
            "items": [record_dict(row) for row in sorted(rows, key=lambda row: row.id)],
            "evidence": sorted(evidence, key=lambda row: row["id"]),
            "constraints": {
                "model_id": model_id, "model_revision": model_revision,
                "bundle_id": bundle_id, "engine": engine,
                "engine_version": engine_version, "hardware_class": hardware_class,
                "execution_mode": execution_mode,
            },
        }

    def create_qualification(self, value: QualificationCreate) -> dict[str, Any]:
        return self._create(QualificationRecord, value.model_dump(mode="json"), "qualification")

    def get_qualification(self, resource_id: str) -> dict[str, Any]:
        return record_dict(self._get(QualificationRecord, resource_id))

    def list_qualifications(self, limit: int, offset: int, **filters: Any) -> dict[str, Any]:
        return self._list(QualificationRecord, limit=limit, offset=offset, filters=filters)

    # Desired deployment state
    def create_deployment(self, value: DeploymentCreate) -> dict[str, Any]:
        return self._create(DeploymentRecord, value.model_dump(mode="json"), "deployment")

    def get_deployment(self, resource_id: str) -> dict[str, Any]:
        return record_dict(self._get(DeploymentRecord, resource_id))

    def list_deployments(self, limit: int, offset: int, **filters: Any) -> dict[str, Any]:
        return self._list(DeploymentRecord, limit=limit, offset=offset, filters=filters)

    def patch_deployment(self, resource_id: str, value: DeploymentPatch) -> dict[str, Any]:
        changes = value.model_dump(exclude_unset=True)
        desired_models = changes.get("desired_models")
        if desired_models:
            first = desired_models[0]
            changes.update({
                "desired_model_id": first["model_id"],
                "desired_bundle_id": first.get("bundle_id"),
                "desired_profile_id": first.get("profile_id"),
                "desired_mode": first.get("mode", "E0"),
            })
        if changes:
            row = self._get(DeploymentRecord, resource_id)
            changes["desired_revision"] = row.desired_revision + 1
        return self._patch(DeploymentRecord, resource_id, changes, "deployment")

    # Runtime instances advertise observed state; deployments continue to own intent.
    def register_instance(
        self, value: ManagedInstanceRegister, *, credential_identity: str | None = None,
    ) -> dict[str, Any]:
        payload = value.model_dump(mode="json")
        instance_id = value.instance_id
        row = self.session.get(ManagedInstanceRecord, instance_id)
        now = datetime.now(timezone.utc)
        if row is not None:
            incompatible = {
                "instance_type": (row.instance_type, value.instance_type.value),
                "name": (row.name, value.name),
                "engine_kind": (row.engine_kind, value.engine_kind),
            }
            conflicts = {
                key: pair for key, pair in incompatible.items()
                if pair[0] is not None and pair[1] is not None and pair[0] != pair[1]
            }
            if conflicts:
                self._audit(
                    "INSTANCE_IDENTITY_CONFLICT", "instance", instance_id,
                    record_dict(row), {"conflicts": conflicts},
                )
                self.session.commit()
                raise RegistryConflict(
                    f"instance {instance_id!r} is already registered with incompatible identity"
                )
            before = record_dict(row)
            for key, item in payload.items():
                target = "metadata_payload" if key == "metadata" else key
                if key not in {"instance_id", "registration_source"}:
                    setattr(row, target, item)
            row.health = value.health
            row.status = self._status(value.health)
            row.last_heartbeat = now
            row.deregistered_at = None
            row.credential_identity = credential_identity or row.credential_identity
            self.session.flush()
            self._audit("INSTANCE_REGISTERED", "instance", instance_id, before, record_dict(row))
        else:
            payload["metadata_payload"] = payload.pop("metadata")
            payload.update({
                "status": self._status(value.health),
                "last_heartbeat": now,
                "registered_at": now,
                "credential_identity": credential_identity,
                "deregistered_at": None,
            })
            row = ManagedInstanceRecord(**payload)
            self.session.add(row)
            self.session.flush()
            self._audit("INSTANCE_REGISTERED", "instance", instance_id, None, record_dict(row))
        self.session.commit()
        return self._instance_dict(row)

    def heartbeat_instance(
        self, instance_id: str, value: ManagedInstanceHeartbeat,
    ) -> dict[str, Any]:
        row = self._get(ManagedInstanceRecord, instance_id)
        if row.deregistered_at is not None:
            raise RegistryConflict("deregistered instance must register again before heartbeat")
        before_status = row.status
        row.health = value.health
        row.status = self._status(value.health)
        row.last_heartbeat = datetime.now(timezone.utc)
        row.observed_revision = value.observed_revision
        row.runtime_summary = dict(value.runtime_summary)
        if value.capabilities is not None:
            row.capabilities = dict(value.capabilities)
        self.session.flush()
        # Heartbeats retain only a compact status transition in the audit stream.
        self._audit(
            "INSTANCE_HEARTBEAT", "instance", instance_id,
            {"status": before_status},
            {"status": row.status, "observed_revision": row.observed_revision},
        )
        self.session.commit()
        return self._instance_dict(row)

    def patch_instance_observed(
        self, instance_id: str, value: ManagedInstanceObservedPatch,
    ) -> dict[str, Any]:
        row = self._get(ManagedInstanceRecord, instance_id)
        before = self._instance_dict(row)
        for key, item in value.model_dump(exclude_unset=True, mode="json").items():
            setattr(row, "metadata_payload" if key == "metadata" else key, item)
        if value.health is not None:
            row.status = self._status(value.health)
        row.last_heartbeat = datetime.now(timezone.utc)
        self.session.flush()
        self._audit("INSTANCE_UPDATED", "instance", instance_id, before, self._instance_dict(row))
        self.session.commit()
        return self._instance_dict(row)

    def deregister_instance(self, instance_id: str) -> dict[str, Any]:
        row = self._get(ManagedInstanceRecord, instance_id)
        before = self._instance_dict(row)
        now = datetime.now(timezone.utc)
        row.status = ManagedInstanceStatus.OFFLINE.value
        row.health = "offline"
        row.deregistered_at = now
        row.last_heartbeat = now
        self.session.flush()
        self._audit("INSTANCE_DEREGISTERED", "instance", instance_id, before, self._instance_dict(row))
        self.session.commit()
        return self._instance_dict(row)

    def get_instance(self, instance_id: str, *, offline_after_seconds: int = 90) -> dict[str, Any]:
        row = self._get(ManagedInstanceRecord, instance_id)
        self._mark_offline((row,), offline_after_seconds)
        self.session.commit()
        return self._instance_dict(row)

    def list_instances(
        self, limit: int, offset: int, *, offline_after_seconds: int = 90,
        instance_type: str | None = None, environment: str | None = None,
        cluster: str | None = None, status: str | None = None,
    ) -> dict[str, Any]:
        rows = self.session.scalars(select(ManagedInstanceRecord)).all()
        self._mark_offline(rows, offline_after_seconds)
        self.session.commit()
        filtered = [row for row in rows if (
            (instance_type is None or row.instance_type == instance_type)
            and (environment is None or row.environment == environment)
            and (cluster is None or row.cluster == cluster)
            and (status is None or row.status == status)
        )]
        filtered.sort(key=lambda row: (row.environment, row.cluster, row.name, row.instance_id))
        return {
            "items": [self._instance_dict(row) for row in filtered[offset:offset + limit]],
            "total": len(filtered), "limit": limit, "offset": offset,
        }

    def desired_instance(self, instance_id: str) -> dict[str, Any]:
        instance = self._get(ManagedInstanceRecord, instance_id)
        deployments = self.session.scalars(select(DeploymentRecord).where(
            DeploymentRecord.environment == instance.environment,
            DeploymentRecord.cluster == instance.cluster,
        )).all()
        matches = []
        for deployment in deployments:
            selector = dict(deployment.engine_instance_selector or {})
            labels = dict(selector.get("labels") or {})
            if selector.get("name") and selector["name"] != instance.name:
                continue
            if selector.get("host") and selector["host"] != instance.host:
                continue
            if any((instance.labels or {}).get(key) != item for key, item in labels.items()):
                continue
            matches.append(deployment)
        selected = sorted(matches, key=lambda row: (-row.desired_revision, row.id))[0] if matches else None
        return {
            "instance_id": instance_id,
            "desired_revision": selected.desired_revision if selected else None,
            "observed_revision": instance.observed_revision,
            "in_sync": instance.in_sync,
            "drift_fields": list(instance.drift_fields or []),
            "desired": record_dict(selected) if selected else None,
        }

    def _mark_offline(
        self, rows: Any, offline_after_seconds: int,
    ) -> None:
        threshold = datetime.now(timezone.utc) - timedelta(seconds=offline_after_seconds)
        for row in rows:
            heartbeat = row.last_heartbeat
            if heartbeat is not None and heartbeat.tzinfo is None:
                heartbeat = heartbeat.replace(tzinfo=timezone.utc)
            if (
                row.deregistered_at is None
                and row.status != ManagedInstanceStatus.OFFLINE.value
                and heartbeat is not None
                and heartbeat < threshold
            ):
                before = {"status": row.status, "last_heartbeat": row.last_heartbeat}
                row.status = ManagedInstanceStatus.OFFLINE.value
                row.health = "offline"
                self._audit(
                    "INSTANCE_OFFLINE", "instance", row.instance_id,
                    before, {"status": row.status, "last_heartbeat": row.last_heartbeat},
                )

    @staticmethod
    def _status(health: str) -> str:
        value = health.lower()
        if value in {"healthy", "ok", "online", "ready"}:
            return ManagedInstanceStatus.ONLINE.value
        if value in {"offline", "stopped", "unavailable"}:
            return ManagedInstanceStatus.OFFLINE.value
        return ManagedInstanceStatus.DEGRADED.value

    @staticmethod
    def _instance_dict(row: ManagedInstanceRecord) -> dict[str, Any]:
        value = record_dict(row)
        value["metadata"] = value.pop("metadata_payload", {})
        value["registration"] = {
            "source": value.pop("registration_source"),
            "credential_identity": value.pop("credential_identity"),
        }
        return value

    # Policies
    def create_policy(self, value: PolicyCreate) -> dict[str, Any]:
        return self._create(PolicyRecord, value.model_dump(mode="json"), "policy")

    def get_policy(self, resource_id: str) -> dict[str, Any]:
        return record_dict(self._get(PolicyRecord, resource_id))

    def list_policies(self, limit: int, offset: int, **filters: Any) -> dict[str, Any]:
        return self._list(PolicyRecord, limit=limit, offset=offset, filters=filters)

    def patch_policy(self, resource_id: str, value: PolicyPatch) -> dict[str, Any]:
        row = self._get(PolicyRecord, resource_id)
        if row.approval_state == ApprovalState.APPROVED.value:
            raise RegistryConflict("approved policies are immutable; create a new version")
        return self._patch(PolicyRecord, resource_id, value.model_dump(exclude_unset=True), "policy")

    # Router desired state. These records configure external data planes; they
    # never participate directly in an inference request.
    def create_router(self, value: RouterInstanceCreate) -> dict[str, Any]:
        payload = _routing_payload(value)
        credential_reference = payload.pop("credential_reference", None)
        result = self._create(RouterInstanceRecord, {
            **payload, "credential_reference": credential_reference,
        }, "router")
        result["metadata"] = result.pop("metadata_payload", {})
        return result

    def get_router(self, resource_id: str) -> dict[str, Any]:
        return _routing_record(self._get(RouterInstanceRecord, resource_id))

    def list_routers(self, limit: int, offset: int, **filters: Any) -> dict[str, Any]:
        return self._list_routing(RouterInstanceRecord, limit=limit, offset=offset, filters=filters)

    def patch_router(self, resource_id: str, value: RouterInstancePatch) -> dict[str, Any]:
        row = self._get(RouterInstanceRecord, resource_id)
        before = _routing_record(row)
        changes = _routing_payload(value, patch=True)
        for key, item in changes.items():
            setattr(row, key, item)
        desired_fields = {"management_url", "inference_url", "region", "cluster", "labels", "metadata_payload"}
        if desired_fields.intersection(changes) and "desired_revision" not in changes:
            row.desired_revision += 1
        self.session.flush()
        self._audit("patch", "router", resource_id, before, _routing_record(row))
        self.session.commit()
        return _routing_record(row)

    def create_route(self, value: RouteCreate) -> dict[str, Any]:
        if self.session.get(RoutingPolicyRecord, value.policy_id) is None:
            raise RegistryConflict("policy_id must reference a routing policy")
        missing = [item for item in value.pool_ids + value.fallback_pool_ids if self.session.get(ModelPoolRecord, item) is None]
        if missing:
            raise RegistryConflict(f"route references unknown pools: {', '.join(sorted(set(missing)))}")
        return self._create_routing(RouteRecord, value, "route")

    def get_route(self, resource_id: str) -> dict[str, Any]:
        return _routing_record(self._get(RouteRecord, resource_id))

    def list_routes(self, limit: int, offset: int, **filters: Any) -> dict[str, Any]:
        return self._list_routing(RouteRecord, limit=limit, offset=offset, filters=filters)

    def patch_route(self, resource_id: str, value: RoutePatch) -> dict[str, Any]:
        if value.policy_id is not None and self.session.get(RoutingPolicyRecord, value.policy_id) is None:
            raise RegistryConflict("policy_id must reference a routing policy")
        pool_ids = list(value.pool_ids or []) + list(value.fallback_pool_ids or [])
        missing = [item for item in pool_ids if self.session.get(ModelPoolRecord, item) is None]
        if missing:
            raise RegistryConflict(f"route references unknown pools: {', '.join(sorted(set(missing)))}")
        return self._patch_routing(RouteRecord, resource_id, value, "route")

    def create_model_pool(self, value: ModelPoolCreate) -> dict[str, Any]:
        return self._create_routing(ModelPoolRecord, value, "model_pool")

    def get_model_pool(self, resource_id: str) -> dict[str, Any]:
        return _routing_record(self._get(ModelPoolRecord, resource_id))

    def list_model_pools(self, limit: int, offset: int, **filters: Any) -> dict[str, Any]:
        return self._list_routing(ModelPoolRecord, limit=limit, offset=offset, filters=filters)

    def patch_model_pool(self, resource_id: str, value: ModelPoolPatch) -> dict[str, Any]:
        return self._patch_routing(ModelPoolRecord, resource_id, value, "model_pool")

    def create_backend_endpoint(self, value: BackendEndpointCreate) -> dict[str, Any]:
        missing = [item for item in value.pool_ids if self.session.get(ModelPoolRecord, item) is None]
        if missing:
            raise RegistryConflict(f"backend references unknown pools: {', '.join(sorted(missing))}")
        return self._create_routing(BackendEndpointRecord, value, "backend_endpoint")

    def get_backend_endpoint(self, resource_id: str) -> dict[str, Any]:
        return _routing_record(self._get(BackendEndpointRecord, resource_id))

    def list_backend_endpoints(self, limit: int, offset: int, **filters: Any) -> dict[str, Any]:
        return self._list_routing(BackendEndpointRecord, limit=limit, offset=offset, filters=filters)

    def patch_backend_endpoint(self, resource_id: str, value: BackendEndpointPatch) -> dict[str, Any]:
        missing = [item for item in (value.pool_ids or []) if self.session.get(ModelPoolRecord, item) is None]
        if missing:
            raise RegistryConflict(f"backend references unknown pools: {', '.join(sorted(missing))}")
        return self._patch_routing(BackendEndpointRecord, resource_id, value, "backend_endpoint")

    def create_routing_policy(self, value: RoutingPolicyCreate) -> dict[str, Any]:
        return self._create_routing(RoutingPolicyRecord, value, "routing_policy")

    def get_routing_policy(self, resource_id: str) -> dict[str, Any]:
        return _routing_record(self._get(RoutingPolicyRecord, resource_id))

    def list_routing_policies(self, limit: int, offset: int, **filters: Any) -> dict[str, Any]:
        return self._list_routing(RoutingPolicyRecord, limit=limit, offset=offset, filters=filters)

    def patch_routing_policy(self, resource_id: str, value: RoutingPolicyPatch) -> dict[str, Any]:
        return self._patch_routing(RoutingPolicyRecord, resource_id, value, "routing_policy")

    def create_route_binding(self, value: RouteBindingCreate) -> dict[str, Any]:
        if self.session.get(RouteRecord, value.route_id) is None:
            raise RegistryConflict("route_id must reference a route")
        if self.session.get(RouterInstanceRecord, value.router_id) is None:
            raise RegistryConflict("router_id must reference a router")
        return self._create_routing(RouteBindingRecord, value, "route_binding")

    def get_route_binding(self, resource_id: str) -> dict[str, Any]:
        return _routing_record(self._get(RouteBindingRecord, resource_id))

    def list_route_bindings(self, limit: int, offset: int, **filters: Any) -> dict[str, Any]:
        return self._list_routing(RouteBindingRecord, limit=limit, offset=offset, filters=filters)

    def patch_route_binding(self, resource_id: str, value: RouteBindingPatch) -> dict[str, Any]:
        return self._patch_routing(RouteBindingRecord, resource_id, value, "route_binding")

    def _create_routing(self, table: type[Record], value: Any, resource_type: str) -> dict[str, Any]:
        resource_id = value.id
        if self.session.get(table, resource_id) is not None:
            raise RegistryConflict(f"{resource_type} {resource_id!r} already exists")
        row = table(**_routing_payload(value))
        self.session.add(row)
        self.session.flush()
        self._touch_bound_routers(resource_type, row)
        public = _routing_record(row)
        self._audit("create", resource_type, resource_id, None, public)
        self.session.commit()
        return public

    def _touch_bound_routers(self, resource_type: str, row: Any) -> None:
        """Advance each affected router's monotonic desired revision."""

        router_ids: set[str] = set()
        route_ids: set[str] = set()
        if resource_type == "route_binding":
            router_ids.add(row.router_id)
        elif resource_type == "route":
            route_ids.add(row.id)
        elif resource_type == "routing_policy":
            route_ids.update(item.id for item in self.session.scalars(
                select(RouteRecord).where(RouteRecord.policy_id == row.id)
            ).all())
        elif resource_type in {"model_pool", "backend_endpoint"}:
            pool_ids = {row.id} if resource_type == "model_pool" else set(row.pool_ids or [])
            route_ids.update(item.id for item in self.session.scalars(select(RouteRecord)).all() if (
                pool_ids.intersection(set(item.pool_ids or []) | set(item.fallback_pool_ids or []))
            ))
        if route_ids:
            router_ids.update(item.router_id for item in self.session.scalars(select(RouteBindingRecord)).all() if item.route_id in route_ids)
        for router_id in router_ids:
            router = self.session.get(RouterInstanceRecord, router_id)
            if router is not None:
                router.desired_revision += 1

    def router_desired_state(self, router_id: str) -> dict[str, Any]:
        """Compile one router's deterministic, qualification-aware desired state."""

        router = self._get(RouterInstanceRecord, router_id)
        bindings = self.session.scalars(select(RouteBindingRecord).where(
            RouteBindingRecord.router_id == router_id,
            RouteBindingRecord.enabled.is_(True),
        )).all()
        compiled_routes: list[dict[str, Any]] = []
        for binding in sorted(bindings, key=lambda row: (row.priority, row.id)):
            route = self.session.get(RouteRecord, binding.route_id)
            if route is None or not route.enabled:
                continue
            policy = self.session.get(RoutingPolicyRecord, route.policy_id)
            if policy is None or not policy.enabled:
                continue
            pools = []
            for pool_id in route.pool_ids + route.fallback_pool_ids:
                pool = self.session.get(ModelPoolRecord, pool_id)
                if pool is None or not pool.enabled:
                    continue
                eligible, excluded = self._eligible_backends(pool, policy)
                pools.append({
                    **_routing_record(pool),
                    "fallback": pool_id in route.fallback_pool_ids,
                    "backends": eligible,
                    "excluded": excluded,
                })
            compiled_routes.append({
                **_routing_record(route),
                "binding": _routing_record(binding),
                "policy": _routing_record(policy),
                "pools": pools,
            })
        desired_revision = router.desired_revision
        return {
            "router": _routing_record(router),
            "desired_revision": desired_revision,
            "routes": compiled_routes,
            "in_sync": router.observed_revision == desired_revision and not router.last_error,
        }

    def _eligible_backends(
        self, pool: ModelPoolRecord, policy: RoutingPolicyRecord,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        rows = self.session.scalars(select(BackendEndpointRecord)).all()
        constraints = {**dict(pool.selectors or {}), **dict(policy.constraints or {})}
        eligible: list[dict[str, Any]] = []
        excluded: list[dict[str, str]] = []
        for row in sorted(rows, key=lambda item: item.id):
            if pool.id not in (row.pool_ids or []):
                continue
            reason = self._backend_exclusion(row, pool, constraints)
            if reason:
                excluded.append({"id": row.id, "reason": reason})
            else:
                eligible.append(_routing_record(row))
        preferences = dict(policy.preferences or {})
        if preferences.get("qualified_first"):
            eligible.sort(key=lambda row: (-QUALIFICATION_ORDER.get(row["qualification_tier"], -1), row["id"]))
        elif preferences.get("region_local"):
            eligible.sort(key=lambda row: (row["region"] != pool.selectors.get("region"), row["id"]))
        elif policy.strategy in {"cost-preferred", "lowest-cost"}:
            eligible.sort(key=lambda row: (row["cost"] is None, row["cost"] or 0, row["id"]))
        return eligible, excluded

    @staticmethod
    def _backend_exclusion(
        row: BackendEndpointRecord, pool: ModelPoolRecord, constraints: dict[str, Any],
    ) -> str | None:
        if row.model_id != pool.model_id:
            return "model identity mismatch"
        if pool.model_revision and row.model_revision != pool.model_revision:
            return "model revision mismatch"
        for field in ("model_revision", "model_fingerprint", "bundle_id", "bundle_revision", "profile", "region", "cluster"):
            expected = constraints.get(field)
            if expected and getattr(row, field) != expected:
                return f"{field} constraint"
        engine_versions = constraints.get("engine_version") or constraints.get("engine_versions")
        if engine_versions and not _version_matches(row.engine_version, engine_versions):
            return "engine version constraint"
        hardware = constraints.get("hardware")
        observed_hardware = (row.metadata_payload or {}).get("hardware")
        if hardware and observed_hardware not in ([hardware] if isinstance(hardware, str) else hardware):
            return "hardware constraint"
        tenants = set(constraints.get("tenant_ids") or constraints.get("tenants") or [])
        endpoint_tenants = set((row.metadata_payload or {}).get("tenant_ids") or [])
        if tenants and not tenants.intersection(endpoint_tenants):
            return "tenant constraint"
        if row.maintenance:
            return "maintenance"
        allowed_health = constraints.get("health", ["READY", "healthy", "online"])
        if isinstance(allowed_health, str):
            allowed_health = [allowed_health]
        if row.health.casefold() not in {str(value).casefold() for value in allowed_health}:
            return "health constraint"
        engines = constraints.get("engine")
        if engines and row.engine not in ([engines] if isinstance(engines, str) else engines):
            return "engine constraint"
        required_modes = set(constraints.get("required_modes") or constraints.get("modes") or [])
        if required_modes and not required_modes.issubset(set(row.modes or [])):
            return "mode constraint"
        minimum = str(constraints.get("minimum_evidence", "NOT_MEASURED"))
        if QUALIFICATION_ORDER.get(row.qualification_tier, -1) < QUALIFICATION_ORDER.get(minimum, 0):
            return "qualification constraint"
        if constraints.get("approved_only") and row.approval_state != ApprovalState.APPROVED.value:
            return "approval constraint"
        labels = dict(constraints.get("labels") or {})
        if any((row.labels or {}).get(key) != value for key, value in labels.items()):
            return "label constraint"
        return None

    # Approval state changes are append-only and always audited.
    def approve(self, value: ApprovalCreate) -> dict[str, Any]:
        table = RESOURCE_TABLES.get(value.resource_type)
        if table is None:
            raise RegistryConflict(f"resource type {value.resource_type!r} cannot be approved")
        row = self._get(table, value.resource_id)
        expected_version = {
            "model": getattr(row, "revision", None),
            "bundle": getattr(row, "immutable_revision", None),
            "profile": getattr(row, "version", None),
            "deployment": str(getattr(row, "desired_revision", "")),
            "policy": getattr(row, "version", None),
        }[value.resource_type]
        if value.version != "current" and expected_version and value.version != expected_version:
            raise RegistryConflict(
                f"approval version {value.version!r} does not match current {expected_version!r}"
            )
        before = record_dict(row)
        row.approval_state = value.state.value
        approval = ApprovalRecord(**value.model_dump(mode="json"))
        self.session.add(approval)
        self.session.flush()
        self._audit("approval", value.resource_type, value.resource_id, before, record_dict(row), version=value.version)
        self.session.commit()
        return record_dict(approval)

    def list_approvals(self, limit: int, offset: int, **filters: Any) -> dict[str, Any]:
        return self._list(ApprovalRecord, limit=limit, offset=offset, filters=filters)

    def list_audit(self, limit: int, offset: int, **filters: Any) -> dict[str, Any]:
        return self._list(AuditRecord, limit=limit, offset=offset, filters=filters)

    # Resolver ordering is total and stable across database implementations.
    def resolve_bundle(self, request: BundleResolveRequest) -> dict[str, Any]:
        query = select(BundleRecord).join(ModelRecord, BundleRecord.base_model_id == ModelRecord.id).where(
            ModelRecord.repo == request.model,
            BundleRecord.lifecycle_state == "active",
        )
        if request.model_revision:
            query = query.where(BundleRecord.base_model_revision == request.model_revision)
        if request.trust:
            query = query.where(BundleRecord.trust == request.trust)
        rows = self.session.scalars(query).all()
        compatibility = self.session.scalars(select(CompatibilityRecord)).all()
        state_rank = {"APPROVED": 0, "CANDIDATE": 1, "DRAFT": 2, "DEPRECATED": 3, "REVOKED": 4}

        def rank(bundle: BundleRecord) -> tuple[Any, ...]:
            matching = [item for item in compatibility if item.bundle_id == bundle.id and (not request.engine or item.engine == request.engine) and _version_matches(request.engine_version, item.engine_version_range)]
            recommended = any(item.recommendation_status in {"RECOMMENDED", "VALIDATED", "CONTROLLED"} for item in matching)
            exact = request.model_revision is None or bundle.base_model_revision == request.model_revision
            return (state_rank.get(bundle.approval_state, 9), not exact, not recommended, bundle.id, bundle.immutable_revision)

        if not rows:
            raise RegistryNotFound("no compatible active bundle was found")
        selected = sorted(rows, key=rank)[0]
        evidence = self.session.scalars(select(QualificationRecord).where(QualificationRecord.bundle_id == selected.id)).all()
        profiles = self.session.scalars(select(ProfileRecord)).all()
        matching_profiles = [record_dict(item) for item in profiles if selected.id in (item.bundle_ids or [])]
        matching_compat = [record_dict(item) for item in compatibility if item.bundle_id == selected.id and (not request.engine or item.engine == request.engine)]
        return {
            "selected_bundle": self.get_bundle(selected.id),
            "immutable_revision": selected.immutable_revision,
            "profile_options": matching_profiles,
            "evidence": [record_dict(item) for item in evidence],
            "reason": "highest approval, exact-revision, engine recommendation, then immutable identity",
            "limitations": sorted({limit for item in matching_compat for limit in item["limitations"]}),
        }

    def resolve_profile(self, request: ProfileResolveRequest) -> dict[str, Any]:
        rows = self.session.scalars(select(ProfileRecord)).all()
        candidates = [row for row in rows if request.model_id in (row.model_ids or []) and (not request.bundle_id or request.bundle_id in (row.bundle_ids or []))]
        if not candidates:
            raise RegistryNotFound("no matching profile was found")
        rank = {"APPROVED": 0, "CANDIDATE": 1, "DRAFT": 2}
        selected = sorted(candidates, key=lambda row: (rank.get(row.approval_state, 9), row.id, row.immutable_revision))[0]
        return {"selected_profile": record_dict(selected), "reason": "highest approval then immutable identity"}

    def resolve_deployment(self, request: DeploymentResolveRequest) -> dict[str, Any]:
        rows = self.session.scalars(select(DeploymentRecord).where(
            DeploymentRecord.environment == request.environment,
            DeploymentRecord.cluster == request.cluster,
        )).all()
        if not rows:
            raise RegistryNotFound("no desired deployment was found")
        selected = sorted(rows, key=lambda row: (-row.desired_revision, row.id))[0]
        return {"desired": record_dict(selected), "reason": "highest desired revision then deployment identity"}


def _version_matches(version: str | None, version_range: str | list[str] | None) -> bool:
    if isinstance(version_range, list):
        return any(_version_matches(version, item) for item in version_range)
    if not version or not version_range or version_range in {"*", "any"}:
        return True
    try:
        return Version(version) in SpecifierSet(version_range)
    except (InvalidVersion, InvalidSpecifier):
        return version == version_range
