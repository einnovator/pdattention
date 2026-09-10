from __future__ import annotations

import pytest

from pra_llamacpp import (
    LlamaCppResourceIdentity,
    LlamaCppResourceSlotAllocator,
    LlamaCppSlotCapacityError,
    LlamaCppSlotIsolationError,
)


def identity(
    tenant: str = "tenant-a",
    session: str = "session-a",
    digest: str = "resource-a",
) -> LlamaCppResourceIdentity:
    return LlamaCppResourceIdentity(tenant, session, "model-sha", digest)


def test_allocator_reuses_identity_and_separates_resource_and_request_slots() -> None:
    allocator = LlamaCppResourceSlotAllocator(
        resource_slots=(0, 1), request_slots=(2, 3)
    )
    first = allocator.acquire(
        request_id="request-1", tenant_id="tenant-a", identity=identity()
    )
    allocator.release("request-1")
    second = allocator.acquire(
        request_id="request-2", tenant_id="tenant-a", identity=identity()
    )

    assert first.resource_slot == second.resource_slot == 0
    assert first.request_slot == second.request_slot == 2
    assert first.resource_slot != first.request_slot


def test_allocator_rejects_cross_tenant_attachment() -> None:
    allocator = LlamaCppResourceSlotAllocator(
        resource_slots=(0,), request_slots=(1,)
    )
    with pytest.raises(LlamaCppSlotIsolationError, match="does not own"):
        allocator.acquire(
            request_id="request-1", tenant_id="tenant-b", identity=identity()
        )


def test_allocator_never_evicts_an_active_resource() -> None:
    allocator = LlamaCppResourceSlotAllocator(
        resource_slots=(0,), request_slots=(1, 2)
    )
    allocator.acquire(
        request_id="request-1", tenant_id="tenant-a", identity=identity()
    )
    with pytest.raises(LlamaCppSlotCapacityError, match="active leases"):
        allocator.acquire(
            request_id="request-2",
            tenant_id="tenant-b",
            identity=identity("tenant-b", "session-b", "resource-b"),
        )


def test_session_invalidation_is_blocked_until_request_releases() -> None:
    allocator = LlamaCppResourceSlotAllocator(
        resource_slots=(0,), request_slots=(1,)
    )
    allocator.acquire(
        request_id="request-1", tenant_id="tenant-a", identity=identity()
    )
    with pytest.raises(LlamaCppSlotIsolationError, match="active request"):
        allocator.invalidate_session("session-a")
    allocator.release("request-1")
    assert allocator.invalidate_session("session-a") == (0,)
    assert allocator.snapshot()["resident_resources"] == 0


def test_allocator_evicts_only_the_least_recent_inactive_resource() -> None:
    allocator = LlamaCppResourceSlotAllocator(
        resource_slots=(0, 1), request_slots=(2,)
    )
    for request_id, item in (
        ("request-a", identity(digest="a")),
        ("request-b", identity(digest="b")),
        ("request-a2", identity(digest="a")),
        ("request-c", identity(digest="c")),
    ):
        lease = allocator.acquire(
            request_id=request_id, tenant_id=item.tenant_id, identity=item
        )
        allocator.release(request_id)
    assert lease.resource_slot == 1
    assert allocator.snapshot()["resident_resources"] == 2
