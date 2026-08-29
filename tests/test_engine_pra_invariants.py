from __future__ import annotations

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
