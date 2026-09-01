from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "docs/papers/shared/results/paper6_7_llamacpp/native_sequence_attach.json"
)
SOURCE = (
    ROOT
    / "engine-patches/llamacpp/examples/pra-native/pra-native.cpp"
)
SERVER_RESULT = (
    ROOT / "docs/papers/shared/results/paper6_7_llamacpp/native_server_attach.json"
)
SERVER_PATCH = ROOT / "engine-patches/llamacpp/llamacpp-pra-server.patch"


def test_native_sequence_attach_is_exact_and_persistent_on_cpu_and_metal():
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["upstream_commit"] == "458681e1d5d4a29a1463c4732e03226cf384b997"
    assert payload["summary"]["runs"] == 10
    assert payload["summary"]["schedule_matched_exact_logits"] == 10
    assert payload["summary"]["persistent_decode_exact"] == 10
    assert payload["summary"]["absent_request_exact"] == 5
    assert payload["summary"]["absent_request_bounded_1e_2"] == 10
    assert payload["summary"]["absent_request_top_token_equal"] == 10
    assert payload["summary"]["warm_resource_reuse_exact"] == 10
    assert payload["summary"]["physical_kv_copy"] is False
    assert {row["backend"] for row in payload["rows"]} == {"cpu", "metal"}
    assert all(row["decode_steps"] == 4 for row in payload["rows"])
    assert max(
        row["absent_request_isolation_max_logit_error"]
        for row in payload["rows"]
    ) < 0.006


def test_native_probe_requires_unified_cache_and_sequence_membership_copy():
    source = SOURCE.read_text(encoding="utf-8")
    assert "context_params.kv_unified = true" in source
    assert "llama_memory_seq_cp" in source
    assert "persistent_decode_max_logit_error" in source


def test_native_server_cohort_closes_parity_reuse_and_isolation_gates():
    payload = json.loads(SERVER_RESULT.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert summary["runs"] == 25
    assert summary["e0_e2_exact"] == 25
    assert summary["e2_warm_exact"] == 25
    assert summary["absent_differs"] == 25
    assert summary["concurrent_exact"] == summary["concurrent_requests"] == 15
    assert summary["physical_kv_copy"] is False
    assert summary["e2_warm_over_e0"] < 1.0


def test_native_server_patch_pins_attaches_and_releases_resources():
    patch = SERVER_PATCH.read_text(encoding="utf-8")
    assert "pra_resource_slot" in patch
    assert "pra_resource_pinned" in patch
    assert "slot.mem.seq_cp(source.id, slot.id, -1, -1)" in patch
    assert '"/pra/resources/:id_slot"' in patch
