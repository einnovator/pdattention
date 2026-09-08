from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from pra_hf.engine_invariants import EnginePRAIsolationGuard


def test_request_a_cannot_leak_selected_detail_to_request_b() -> None:
    guard = EnginePRAIsolationGuard()
    guard.open_request("A", ("resource-R",))
    guard.attach_once("A")
    guard.close_request("A")

    guard.open_request("B", ())
    assert guard.visible_keys("A") == ()
    assert guard.visible_keys("B") == ()
    guard.attach_once("B")
    guard.close_request("B")


def test_request_c_receives_exactly_one_copy_of_selected_detail() -> None:
    guard = EnginePRAIsolationGuard()
    guard.open_request("C", ("resource-R",))
    guard.attach_once("C", ("resource-R",))

    with pytest.raises(RuntimeError, match="already attached"):
        guard.attach_once("C", ("resource-R",))

    guard.close_request("C")


def test_selected_detail_is_rejected_by_ordinary_cache_pool() -> None:
    guard = EnginePRAIsolationGuard()

    guard.assert_ordinary_pool_safe(())
    with pytest.raises(RuntimeError, match="ordinary sequential or prefix"):
        guard.assert_ordinary_pool_safe(("resource-R",))


def test_duplicate_selected_keys_are_rejected_before_attachment() -> None:
    guard = EnginePRAIsolationGuard()

    with pytest.raises(ValueError, match="unique"):
        guard.open_request("A", ("resource-R", "resource-R"))


def test_concurrent_tenants_keep_request_visibility_and_attachment_disjoint() -> None:
    guard = EnginePRAIsolationGuard()

    def execute(index: int) -> tuple[str, tuple[str, ...], str | None, str | None]:
        request_id = f"request-{index}"
        tenant_id = f"tenant-{index % 3}"
        session_id = f"session-{index}"
        keys = (f"{tenant_id}/resource-{index}",)
        guard.open_request(
            request_id, keys, tenant_id=tenant_id, session_id=session_id
        )
        guard.assert_request_scope(
            request_id, tenant_id=tenant_id, session_id=session_id
        )
        guard.attach_once(request_id, keys)
        view = guard.view(request_id)
        assert view is not None
        return request_id, view.logical_keys, view.tenant_id, view.session_id

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = tuple(pool.map(execute, range(24)))

    assert len({row[0] for row in rows}) == 24
    assert all(row[1][0].startswith(f"{row[2]}/") for row in rows)
    for request_id, _, _, _ in rows:
        guard.close_request(request_id)
        assert guard.visible_keys(request_id) == ()


def test_request_scope_rejects_cross_tenant_or_cross_session_reuse() -> None:
    guard = EnginePRAIsolationGuard()
    guard.open_request(
        "request", ("tenant-a/resource",), tenant_id="tenant-a", session_id="one"
    )

    with pytest.raises(RuntimeError, match="scope mismatch"):
        guard.assert_request_scope(
            "request", tenant_id="tenant-b", session_id="one"
        )
    with pytest.raises(RuntimeError, match="scope mismatch"):
        guard.assert_request_scope(
            "request", tenant_id="tenant-a", session_id="two"
        )
