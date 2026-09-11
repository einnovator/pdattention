from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from experiments.paper4_5_agent.run_vllm_cuda_live_agent_kv_gate import (
    _selected_page_indices,
)
from pra_hf.live_history import LiveKVInterval, LiveKVSelectionPlan
from pra_vllm.cuda_sparse_connector import PRASparseConnector, _position_delta
from pra_vllm.cuda_sparse_protocol import SparseCudaConnectorCommand


def test_sparse_command_round_trip_preserves_original_position_base() -> None:
    command = SparseCudaConnectorCommand(
        "load", "agent-selection", 144, 160, "hot", "turn-4"
    )

    assert SparseCudaConnectorCommand.parse(command.cache_salt()) == command
    assert _position_delta(command.source_tokens, command.source_position_base) == 16


def test_complete_page_materialization_keeps_retention_as_a_floor() -> None:
    plan = LiveKVSelectionPlan.create(
        160,
        (
            LiveKVInterval(0, 64, "preamble", "preamble"),
            LiveKVInterval(80, 160, "recent", "turn:recent"),
        ),
        source_position_base=160,
    )

    pages = _selected_page_indices(plan, 16)
    realized = len(pages) * 16 / plan.source_tokens

    assert pages == (0, 1, 2, 3, 5, 6, 7, 8, 9)
    assert realized == pytest.approx(0.9)
    assert realized >= 0.9


def _connector_with_two_borrowers() -> PRASparseConnector:
    connector = PRASparseConnector.__new__(PRASparseConnector)
    key = ("agent-selection", "hot")
    connector._commands = {}
    connector._loads = {}
    connector._detached_active_requests = {"candidate": key, "reference": key}
    connector._detached_refcounts = {key: 2}
    connector._detached_handles = {key: (100, 101)}
    connector._detached_materialized = {key}
    connector._detached_tensor_bytes = {key: 4096}
    connector._detached_free = []
    return connector


def test_shared_borrowers_release_independently_and_eviction_fails_closed() -> None:
    connector = _connector_with_two_borrowers()

    with pytest.raises(RuntimeError, match="2 borrower"):
        connector.evict_detached_resource("agent-selection")

    connector.request_finished(SimpleNamespace(request_id="candidate"), [])
    assert connector._detached_refcounts[("agent-selection", "hot")] == 1
    with pytest.raises(RuntimeError, match="1 borrower"):
        connector.evict_detached_resource("agent-selection")

    # The same callback is used after cancellation/termination.  Releasing the
    # final request keeps HOT state resident until an explicit safe eviction.
    connector.request_finished(SimpleNamespace(request_id="reference"), [])
    assert ("agent-selection", "hot") not in connector._detached_refcounts
    assert connector.evict_detached_resource("agent-selection") == (100, 101)
    assert connector._detached_free == [100, 101]
    assert not connector._detached_handles
    assert not connector._detached_materialized


def test_eviction_rejects_inconsistent_live_owner_without_freeing_pages() -> None:
    connector = _connector_with_two_borrowers()
    connector._detached_refcounts.clear()

    with pytest.raises(RuntimeError, match="ownership corruption"):
        connector.evict_detached_resource("agent-selection")

    assert connector._detached_handles[("agent-selection", "hot")] == (100, 101)
    assert connector._detached_free == []


def test_authoritative_active_snapshot_reconciles_finished_borrowers() -> None:
    connector = _connector_with_two_borrowers()

    assert connector.reconcile_detached_requests({"reference"}) == ("candidate",)
    assert connector._detached_refcounts[("agent-selection", "hot")] == 1
    assert connector.reconcile_detached_requests(set()) == ("reference",)
    assert ("agent-selection", "hot") not in connector._detached_refcounts
