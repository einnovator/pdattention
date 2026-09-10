"""Exercise tenant-scoped llama.cpp resource-slot lifecycle invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pra_llamacpp import (
    LlamaCppResourceIdentity,
    LlamaCppResourceSlotAllocator,
    LlamaCppSlotCapacityError,
    LlamaCppSlotIsolationError,
)


def identity(tenant: str, session: str, digest: str) -> LlamaCppResourceIdentity:
    return LlamaCppResourceIdentity(tenant, session, "qwen2.5-0.5b-q4km", digest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    allocator = LlamaCppResourceSlotAllocator(
        resource_slots=(0, 1), request_slots=(2, 3)
    )
    alpha = identity("tenant-a", "session-a", "document-alpha-v1")
    beta = identity("tenant-b", "session-b", "document-beta-v1")
    first = allocator.acquire(
        request_id="request-a1", tenant_id="tenant-a", identity=alpha
    )
    second = allocator.acquire(
        request_id="request-b1", tenant_id="tenant-b", identity=beta
    )

    active_eviction_blocked = False
    try:
        allocator.acquire(
            request_id="request-c1",
            tenant_id="tenant-c",
            identity=identity("tenant-c", "session-c", "document-gamma-v1"),
        )
    except LlamaCppSlotCapacityError:
        active_eviction_blocked = True

    cross_tenant_blocked = False
    try:
        allocator.acquire(
            request_id="request-x1", tenant_id="tenant-b", identity=alpha
        )
    except LlamaCppSlotIsolationError:
        cross_tenant_blocked = True

    allocator.release(first.request_id)
    allocator.release(second.request_id)
    released_slots = allocator.invalidate_session("session-a")
    replacement = allocator.acquire(
        request_id="request-c2",
        tenant_id="tenant-c",
        identity=identity("tenant-c", "session-c", "document-gamma-v1"),
    )
    allocator.release(replacement.request_id)

    checks = {
        "resource_request_pools_disjoint": not {
            first.resource_slot,
            second.resource_slot,
        }.intersection({first.request_slot, second.request_slot}),
        "distinct_active_resource_slots": first.resource_slot != second.resource_slot,
        "distinct_active_request_slots": first.request_slot != second.request_slot,
        "cross_tenant_attach_blocked": cross_tenant_blocked,
        "active_resource_eviction_blocked": active_eviction_blocked,
        "session_release_returned_owned_slot": released_slots
        == (first.resource_slot,),
        "released_slot_reused": replacement.resource_slot == first.resource_slot,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Resource-slot allocator audit failed: {checks}")
    payload = {
        "schema_version": "paper6.7-resource-slot-allocator-v1",
        "evidence_tier": "CONTROL_PLANE_LIFECYCLE_TEST",
        "server_transport_scope": "one selected resource sequence per request",
        "checks": checks,
        "allocator": allocator.snapshot(),
        "claims": {
            "tenant_isolation": "control-plane enforced",
            "session_invalidation": "fail-closed while active",
            "active_resource_eviction": "forbidden",
            "multi_resource_native_geometry": "not implemented",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
