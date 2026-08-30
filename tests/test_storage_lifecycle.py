"""Shared PRA HOT/WARM/COLD/SOURCE lifecycle and policy contracts."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from click.testing import CliRunner

from pra_hf.cli import cli
from pra_hf.storage_lifecycle import (
    FileKVStore,
    Float32Int8ColdCodec,
    InMemoryHotBridge,
    MemoryKVStore,
    MemoryMappedKVStore,
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


def test_restart_rehydrates_warm_entry_without_process_local_loader(tmp_path):
    policy = _policy(tmp_path)
    state_path = tmp_path / "lifecycle.json"
    first = PRAStorageManager(policy, state_path=state_path)
    first.register(_entry("restart", task_status=None), b"durable-native", fingerprint="fp")
    first.demote_hot("restart")
    first.close()

    recovered = PRAStorageManager(policy)

    assert recovered.entries["restart"].current_tier == PRAStorageTier.WARM
    assert recovered.promote("restart") == b"durable-native"


def test_source_resolver_reconstructs_source_only_entry_after_restart(tmp_path):
    policy = _policy(tmp_path, warm=PRAStorageTierConfig(enabled=False), cold=PRAStorageTierConfig(enabled=False))
    state_path = tmp_path / "lifecycle.json"
    first = PRAStorageManager(policy, state_path=state_path)
    first.register_source(_entry("source-only"), lambda: b"original")
    first.close()

    recovered = PRAStorageManager(
        policy,
        source_resolver=lambda entry: f"rebuilt:{entry.logical_key}".encode(),
        state_path=state_path,
    )

    assert recovered.promote("source-only") == b"rebuilt:source-only"


def test_restart_restores_metrics_task_dependency_session_and_file_source(tmp_path):
    source = tmp_path / "authoritative.kv"
    source.write_bytes(b"reconstructed-native")
    policy = _policy(
        tmp_path,
        warm=PRAStorageTierConfig(enabled=False),
        cold=PRAStorageTierConfig(enabled=False),
    )
    state_path = tmp_path / "lifecycle.json"
    first = PRAStorageManager(policy, state_path=state_path)
    entry = _entry(
        "durable-source",
        source_uri=source.as_uri(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        dependent_record_count=3,
    )
    first.register_source(entry, source.read_bytes)
    first.metrics.bytes_read = 17
    first.close()

    recovered = PRAStorageManager(policy, state_path=state_path)

    restored = recovered.entries["durable-source"]
    assert restored.session_id == "session-a"
    assert restored.task_id == "task-a"
    assert restored.dependent_record_count == 3
    assert recovered.metrics.bytes_read == 17
    assert recovered.promote("durable-source") == b"reconstructed-native"


def test_file_source_checksum_rejects_changed_authoritative_content(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"original")
    policy = _policy(
        tmp_path,
        warm=PRAStorageTierConfig(enabled=False),
        cold=PRAStorageTierConfig(enabled=False),
    )
    manager = PRAStorageManager(policy, state_path=tmp_path / "state.json")
    manager.register_source(
        _entry(
            "checked",
            source_uri=str(source),
            source_sha256=hashlib.sha256(b"original").hexdigest(),
        ),
        source.read_bytes,
    )
    manager.close()
    source.write_bytes(b"changed")

    recovered = PRAStorageManager(policy, state_path=tmp_path / "state.json")
    with pytest.raises(ValueError, match="checksum"):
        recovered.promote("checked")


def test_concurrent_promotions_share_one_hot_value_without_state_race(tmp_path):
    manager = PRAStorageManager(_policy(tmp_path))
    manager.register(_entry("shared-promotion", task_status=None), b"native-kv")
    manager.demote_hot("shared-promotion")

    with ThreadPoolExecutor(max_workers=8) as pool:
        values = tuple(
            pool.map(lambda _index: manager.promote("shared-promotion"), range(32))
        )

    assert values == (b"native-kv",) * 32
    assert manager.entries["shared-promotion"].current_tier == PRAStorageTier.HOT
    assert manager.metrics.reloads == 1


def test_hot_promotion_retains_durable_warm_copy_and_avoids_rewrite(tmp_path):
    policy = _policy(tmp_path)
    manager = PRAStorageManager(policy)
    manager.register(_entry("crash-safe", task_status=None), b"native")
    manager.demote_hot("crash-safe")
    writes = manager.metrics.persistence_writes

    assert manager.promote("crash-safe") == b"native"
    assert manager.warm.contains("crash-safe")
    assert manager.entries["crash-safe"].warm_bytes > 0
    manager.demote_hot("crash-safe")

    assert manager.metrics.persistence_writes == writes
    manager.close()
    recovered = PRAStorageManager(policy)
    assert recovered.promote("crash-safe") == b"native"


def test_mmap_store_reads_named_segments_without_loading_neighbor(tmp_path):
    store = MemoryMappedKVStore(tmp_path / "segments")
    store.put_segments(
        "memory",
        {"layer-0-k": b"keys", "layer-0-v": b"values"},
        {"fingerprint": "model-v1"},
    )

    assert store.get_segments(
        "memory", ("layer-0-v",), {"fingerprint": "model-v1"}
    ) == {"layer-0-v": b"values"}


def test_cold_codec_quantizes_and_restores_float_payload(tmp_path):
    import numpy as np

    policy = _policy(
        tmp_path,
        warm=PRAStorageTierConfig(path=str(tmp_path / "warm"), max_bytes=4096, cold_grace_seconds=0),
        cold=PRAStorageTierConfig(path=str(tmp_path / "cold"), max_bytes=4096, kv_quantization="int8"),
    )
    values = np.linspace(-1, 1, 128, dtype=np.float32)
    manager = PRAStorageManager(policy, cold_codec=Float32Int8ColdCodec())
    manager.register(_entry("quantized", task_status=None), values.tobytes(), now_ns=0)
    manager.demote_hot("quantized", now_ns=1)
    manager.run_maintenance(now_ns=8 * 86400 * 1_000_000_000)

    assert manager.entries["quantized"].current_tier == PRAStorageTier.COLD
    assert manager.entries["quantized"].cold_bytes < values.nbytes
    restored = np.frombuffer(manager.promote("quantized"), dtype=np.float32)
    assert np.max(np.abs(restored - values)) < 0.01


def test_per_tenant_quota_cannot_be_consumed_by_another_tenant(tmp_path):
    policy = _policy(
        tmp_path,
        warm=PRAStorageTierConfig(
            max_bytes=100,
            per_tenant_max_bytes=12,
            cold_grace_seconds="1d",
        ),
        cold=PRAStorageTierConfig(enabled=False),
    )
    manager = PRAStorageManager(policy, warm=MemoryKVStore())
    manager.register(_entry("a1", tenant_id="tenant-a", task_status=None), b"a" * 8)
    manager.register(_entry("a2", "terminal_output", tenant_id="tenant-a", task_status=None), b"b" * 8)
    manager.register(_entry("b1", tenant_id="tenant-b", task_status=None), b"c" * 8)
    for key in ("a1", "a2", "b1"):
        manager.demote_hot(key)
    manager.run_maintenance()

    assert sum(
        entry.warm_bytes
        for entry in manager.entries.values()
        if entry.tenant_id == "tenant-a"
    ) <= 12
    assert manager.entries["b1"].current_tier == PRAStorageTier.WARM
