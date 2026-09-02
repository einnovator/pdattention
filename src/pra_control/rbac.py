"""Role-based authorization shared by HTTP handlers and agent tools."""

from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    VIEWER = "Viewer"
    OPERATOR = "Operator"
    APPROVER = "Approver"
    ADMINISTRATOR = "Administrator"


class Permission(str, Enum):
    FLEET_READ = "fleet:read"
    FLEET_ADMIN = "fleet:admin"
    ENGINE_READ = "engine:read"
    REGISTRY_READ = "registry:read"
    REGISTRY_WRITE = "registry:write"
    APPROVE = "governance:approve"
    ENGINE_ACTION = "engine:action"
    ENGINE_HIGH_IMPACT = "engine:high-impact"
    ENGINE_CONFIGURE = "engine:configure"
    QUALIFICATION_READ = "qualification:read"
    QUALIFICATION_APPROVE = "qualification:approve"
    DEPLOYMENT_READ = "deployment:read"
    DEPLOYMENT_WRITE = "deployment:write"
    DEPLOYMENT_APPLY = "deployment:apply"
    OBSERVABILITY_READ = "observability:read"
    AUDIT_READ = "audit:read"
    EXPERIMENT_READ = "experiment:read"
    EXPERIMENT_RUN = "experiment:run"
    IDENTITY_ADMIN = "identity:admin"
    ADMIN = "admin"


ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.VIEWER: frozenset({
        Permission.FLEET_READ, Permission.ENGINE_READ, Permission.REGISTRY_READ,
        Permission.QUALIFICATION_READ, Permission.DEPLOYMENT_READ, Permission.EXPERIMENT_READ,
        Permission.OBSERVABILITY_READ, Permission.AUDIT_READ,
    }),
    Role.OPERATOR: frozenset({
        Permission.FLEET_READ, Permission.ENGINE_READ, Permission.REGISTRY_READ,
        Permission.QUALIFICATION_READ, Permission.DEPLOYMENT_READ, Permission.EXPERIMENT_READ,
        Permission.OBSERVABILITY_READ, Permission.AUDIT_READ,
        Permission.ENGINE_ACTION, Permission.ENGINE_CONFIGURE, Permission.DEPLOYMENT_WRITE,
    }),
    Role.APPROVER: frozenset({
        Permission.FLEET_READ, Permission.ENGINE_READ, Permission.REGISTRY_READ, Permission.REGISTRY_WRITE,
        Permission.QUALIFICATION_READ, Permission.QUALIFICATION_APPROVE,
        Permission.DEPLOYMENT_READ, Permission.DEPLOYMENT_WRITE, Permission.DEPLOYMENT_APPLY,
        Permission.OBSERVABILITY_READ, Permission.AUDIT_READ, Permission.EXPERIMENT_READ,
        Permission.APPROVE, Permission.ENGINE_HIGH_IMPACT,
    }),
    Role.ADMINISTRATOR: frozenset(Permission),
}


def permits(role: Role, permission: Permission) -> bool:
    return permission in ROLE_PERMISSIONS[role]


def permissions_for_role(role: Role) -> frozenset[str]:
    """Return canonical string permissions for transport-neutral callers."""
    return frozenset(permission.value for permission in ROLE_PERMISSIONS[role])
