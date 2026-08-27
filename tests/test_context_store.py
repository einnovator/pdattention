import json
import time

import pytest

from pra_hf.context_records import RecordType
from pra_hf.context_store import (
    BackingStoreError,
    LocalBackingStore,
    RecordAccessDenied,
    RecordNotFound,
    RecordScope,
)


def test_store_round_trip_is_content_addressed_and_scope_checked(tmp_path):
    store = LocalBackingStore(tmp_path, persistent=True)
    scope = RecordScope("tenant-a", "session-a")
    payload = {"rows": [[1, "alpha"], [2, "beta"]], "ok": True}

    first = store.put(payload, record_type=RecordType.DB_RESULT, scope=scope)
    second = store.put(payload, record_type=RecordType.DB_RESULT, scope=scope)

    assert first.record_id == second.record_id
    assert store.get(first.record_id, scope=scope) == payload
    assert store.stats().records == 1
    with pytest.raises(RecordAccessDenied):
        store.get(first.record_id, scope=RecordScope("tenant-a", "session-b"))


def test_logical_identity_preserves_type_and_provenance(tmp_path):
    store = LocalBackingStore(tmp_path, persistent=True)
    scope = RecordScope("tenant", "session")

    first = store.put(
        {"value": 1}, record_type=RecordType.API_RESULT, scope=scope,
        provenance={"call_id": "call-1"},
    )
    second = store.put(
        {"value": 1}, record_type=RecordType.TOOL_RESPONSE, scope=scope,
        provenance={"call_id": "call-1"},
    )
    third = store.put(
        {"value": 1}, record_type=RecordType.API_RESULT, scope=scope,
        provenance={"call_id": "call-2"},
    )

    assert len({first.record_id, second.record_id, third.record_id}) == 3
    assert first.content_hash == second.content_hash == third.content_hash
    assert store.stats().records == 3


def test_store_verifies_hash_and_expires_records(tmp_path):
    store = LocalBackingStore(tmp_path, persistent=True)
    scope = RecordScope("tenant", "session")
    record = store.put("exact", record_type=RecordType.GENERIC_TEXT, scope=scope)
    payload_path = next(tmp_path.glob("*/*.payload"))
    payload_path.write_text("changed", encoding="utf-8")

    with pytest.raises(BackingStoreError, match="hash verification"):
        store.get(record.record_id, scope=scope)

    expiring = store.put(
        "short lived",
        record_type=RecordType.GENERIC_TEXT,
        scope=scope,
        ttl_seconds=0.01,
    )
    time.sleep(0.02)
    with pytest.raises(RecordNotFound):
        store.get(expiring.record_id, scope=scope)


def test_persistent_manifest_reopens_exact_payload(tmp_path):
    scope = RecordScope("tenant", "session")
    first = LocalBackingStore(tmp_path, persistent=True)
    record = first.put(b"\x00\x01exact", record_type=RecordType.API_RESULT, scope=scope)

    reopened = LocalBackingStore(tmp_path, persistent=True)

    assert reopened.descriptor(record.record_id, scope=scope).content_hash == record.content_hash
    assert reopened.get(record.record_id, scope=scope) == b"\x00\x01exact"
