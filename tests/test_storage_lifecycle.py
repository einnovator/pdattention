"""Shared PRA HOT/WARM/COLD/SOURCE lifecycle and policy contracts."""

from dataclasses import replace
from pathlib import Path

import pytest
from click.testing import CliRunner

from pra_hf.cli import cli
from pra_hf.storage_lifecycle import (
    FileKVStore,
    InMemoryHotBridge,
    MemoryKVStore,
    PRARetentionClass,
    PRAStorageEntry,
    PRAStorageEvictionPolicy,
    PRAStorageManager,
    PRAStoragePolicy,
    PRAStorageTier,
    PRAStorageTierConfig,
    dequantize_int8_array,
    parse_byte_size,
    parse_duration,
    quantize_int8_array,
)


def _policy(tmp_path, **changes):
    values = {
        "profile": "balanced",
        "hot": PRAStorageTierConfig(max_bytes=1024),
        "warm": PRAStorageTierConfig(path=str(tmp_path / "warm"), max_bytes=1024, cold_grace_seconds=10),
        "cold": PRAStorageTierConfig(path=str(tmp_path / "cold"), max_bytes=1024, compression="gzip"),
    }
    values.update(changes)
    return PRAStoragePolicy(**values)


def _entry(key, record_type="generic_document", **changes):
    values = dict(
        logical_key=key,
        record_type=record_type,
        retention_class=PRARetentionClass.RECONSTRUCTABLE,
        tenant_id="tenant-a",
        security_scope=None,
        session_id="session-a",
        task_id="task-a",
        task_status="active",
        resource_version="v1",
        detail_bytes=0,
        created_ns=0,
        last_access_ns=0,
    )
    values.update(changes)
    return PRAStorageEntry(**values)


def test_human_sizes_durations_and_named_profiles(tmp_path):
    assert parse_byte_size("8GiB") == 8 * 1024**3
    assert parse_duration("15m") == 900
    policy = PRAStoragePolicy.named("minimal", home=tmp_path)
    assert policy.profile == "minimal"
    assert policy.warm.max_bytes == 4 * 1024**3
    assert policy.cold.compression == "gzip"
    assert Path(PRAStoragePolicy().warm.path).is_absolute()


def test_cli_inspect_resolves_storage_yaml_and_emits_machine_readable_policy(tmp_path):
    config = tmp_path / "storage.yaml"
    config.write_text(
        "storage:\n  profile: minimal\n  warm:\n    max_size: 12MiB\n"
        "    cold_grace_period: 3m\n  cold:\n    kv_quantization: int8\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        cli,
        ["runtime", "inspect", "org/model", "--storage-config", str(config), "--yaml"],
    )

    assert result.exit_code == 0, result.output
    assert "profile: minimal" in result.output
    assert "max_bytes: 12582912" in result.output
    assert "kv_quantization: int8" in result.output


def test_int8_quantization_is_independent_from_backend_compression():
    import numpy as np

    values = np.linspace(-2.0, 2.0, 4096, dtype=np.float32).reshape(2, 8, 256)
    payload, metadata = quantize_int8_array(values)
    restored = dequantize_int8_array(payload, metadata)

    assert len(payload) == values.size
    assert values.nbytes / len(payload) == 4.0
    assert np.max(np.abs(restored - values)) <= metadata["scale"] / 2 + 1e-6


def test_lossless_tier_round_trip_and_strict_fingerprint(tmp_path):
    manager = PRAStorageManager(_policy(tmp_path))
    payload = b"native-kv" * 20
    manager.register(_entry("block:a", task_status=None), payload, fingerprint="model-layout-v1", now_ns=0)
    manager.demote_hot("block:a", now_ns=1)
    assert manager.entries["block:a"].current_tier == PRAStorageTier.WARM
    manager.run_maintenance(now_ns=8 * 86400 * 1_000_000_000)
    assert manager.entries["block:a"].current_tier == PRAStorageTier.COLD
    assert manager.promote("block:a") == payload

    store = FileKVStore(tmp_path / "strict")
    store.put("x", payload, {"fingerprint": "one"})
    with pytest.raises(ValueError, match="fingerprint"):
        store.get("x", {"fingerprint": "two"})


def test_promotion_enforces_tenant_and_security_scope(tmp_path):
    manager = PRAStorageManager(_policy(tmp_path))
    manager.register(_entry("private", security_scope="case:7"), b"kv")
    manager.demote_hot("private")

    with pytest.raises(PermissionError, match="Cross-tenant"):
        manager.promote("private", tenant_id="tenant-b", authorization_scopes=("case:7",))
    with pytest.raises(PermissionError, match="not authorized"):
        manager.promote("private", tenant_id="tenant-a")
    assert manager.promote("private", tenant_id="tenant-a", authorization_scopes=("case:7",)) == b"kv"


def test_hot_budget_demotes_lower_value_unpinned_object(tmp_path):
    policy = _policy(tmp_path, hot=PRAStorageTierConfig(max_bytes=12))
    manager = PRAStorageManager(policy, warm=MemoryKVStore(), cold=MemoryKVStore())
    manager.register(_entry("log", "terminal_output"), b"L" * 8, now_ns=0)
    manager.register(_entry("doc", shared_reference_count=5), b"D" * 8, now_ns=1)

    assert manager.entries["log"].current_tier == PRAStorageTier.WARM
    assert manager.entries["doc"].current_tier == PRAStorageTier.HOT


def test_grace_avoids_transient_persistence_write(tmp_path):
    manager = PRAStorageManager(_policy(tmp_path))
    manager.register(_entry("tool:a", "tool_response"), b"temporary", now_ns=0)
    manager.update_task_status("task-a", "completed", now_ns=0)
    manager.run_maintenance(now_ns=5 * 60 * 1_000_000_000)

    assert manager.entries["tool:a"].current_tier == PRAStorageTier.SOURCE
    assert manager.metrics.persistence_writes == 0


def test_open_task_and_dependency_delay_compaction(tmp_path):
    manager = PRAStorageManager(_policy(tmp_path))
    manager.register(_entry("open", "tool_response"), b"open", now_ns=0)
    manager.run_maintenance(now_ns=3600 * 1_000_000_000)
    assert manager.entries["open"].current_tier == PRAStorageTier.HOT
    assert manager.metrics.task_aware_retention_hits == 1

    manager.entries["open"] = replace(manager.entries["open"], dependent_record_count=1)
    manager.update_task_status("task-a", "completed", now_ns=0)
    manager.run_maintenance(now_ns=400 * 1_000_000_000)
    assert manager.entries["open"].current_tier != PRAStorageTier.SOURCE

    manager.entries["open"] = replace(manager.entries["open"], dependent_record_count=0)
    manager.run_maintenance(now_ns=401 * 1_000_000_000)
    assert manager.entries["open"].current_tier == PRAStorageTier.SOURCE


def test_weighted_quota_prefers_shared_document_over_transient_log(tmp_path):
    policy = _policy(
        tmp_path,
        warm=PRAStorageTierConfig(max_bytes=12, cold_grace_seconds="1d"),
        cold=PRAStorageTierConfig(enabled=False),
        eviction_policy=PRAStorageEvictionPolicy.WEIGHTED_LRU,
    )
    manager = PRAStorageManager(policy, warm=MemoryKVStore())
    manager.register(_entry("doc", shared_reference_count=8, task_status=None), b"D" * 10, now_ns=0)
    manager.register(_entry("log", "terminal_output", task_status=None), b"L" * 10, now_ns=0)
    manager.demote_hot("doc", now_ns=1)
    manager.demote_hot("log", now_ns=1)
    manager.run_maintenance(now_ns=2)

    assert manager.entries["doc"].current_tier == PRAStorageTier.WARM
    assert manager.entries["log"].current_tier == PRAStorageTier.SOURCE


def test_session_close_preserves_shared_but_drops_transient_native_detail(tmp_path):
    manager = PRAStorageManager(_policy(tmp_path), warm=MemoryKVStore(), cold=MemoryKVStore())
    manager.register(_entry("shared", session_id="session-a"), b"shared")
    manager.register(_entry("terminal", "terminal_output", session_id="session-a"), b"scratch")
    freed = manager.close_session("session-a")

    assert manager.entries["shared"].current_tier == PRAStorageTier.HOT
    assert manager.entries["terminal"].current_tier == PRAStorageTier.SOURCE
    assert freed == len(b"scratch")


@pytest.mark.parametrize("engine", ["hf", "vllm", "sglang", "mlx"])
def test_engine_hot_realizations_share_identical_semantic_policy(tmp_path, engine):
    bridge = InMemoryHotBridge()
    manager = PRAStorageManager(_policy(tmp_path / engine), hot=bridge, warm=MemoryKVStore(), cold=MemoryKVStore())
    manager.register(_entry(f"{engine}:block"), b"kv", now_ns=0)
    manager.demote_hot(f"{engine}:block", now_ns=1)

    assert manager.entries[f"{engine}:block"].current_tier == PRAStorageTier.WARM
    assert manager.inspect()["objects"] == {"hot": 0, "warm": 1, "cold": 0, "source": 0}
