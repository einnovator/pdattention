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
    REGISTRY_READ = "registry:read"
    REGISTRY_WRITE = "registry:write"
    APPROVE = "governance:approve"
    ENGINE_ACTION = "engine:action"
    ENGINE_CONFIGURE = "engine:configure"
    OBSERVABILITY_READ = "observability:read"
    AUDIT_READ = "audit:read"
    IDENTITY_ADMIN = "identity:admin"


ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.VIEWER: frozenset({
        Permission.FLEET_READ, Permission.REGISTRY_READ,
        Permission.OBSERVABILITY_READ, Permission.AUDIT_READ,
    }),
    Role.OPERATOR: frozenset({
        Permission.FLEET_READ, Permission.REGISTRY_READ,
        Permission.OBSERVABILITY_READ, Permission.AUDIT_READ,
        Permission.ENGINE_ACTION, Permission.ENGINE_CONFIGURE,
    }),
    Role.APPROVER: frozenset({
        Permission.FLEET_READ, Permission.REGISTRY_READ, Permission.REGISTRY_WRITE,
        Permission.OBSERVABILITY_READ, Permission.AUDIT_READ, Permission.APPROVE,
    }),
    Role.ADMINISTRATOR: frozenset(Permission),
}


def permits(role: Role, permission: Permission) -> bool:
    return permission in ROLE_PERMISSIONS[role]
