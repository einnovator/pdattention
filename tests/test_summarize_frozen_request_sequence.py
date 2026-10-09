import json

import pytest

from experiments.paper4_5_agent.summarize_frozen_request_sequence import summarize


def _artifact(tmp_path, index, retention=1.0, digest=None, *, hf_pack=False):
    path = tmp_path / f"request_{index}.json"
    path.write_text(
        json.dumps(
            {
                "engine": "test-engine",
                "model": "test-model",
                "request_index": index,
                "request_input_sha256": digest or f"digest-{index}",
                "source_policy": "frozen-policy",
                "source_tokens": 100,
                "wire_suffix_tokens": 10,
                "selected_kv_tokens": int(100 * retention),
                "realized_retention_fraction": retention,
                "selected_text_reencoded_tokens": 0,
                **({"interval_pack_bytes": 0} if hf_pack else {"selection_pack_bytes": 0}),
                "physical_kv_copy": False,
                "transient_attention_bytes": 200 + index,
                "offloaded_payload_bytes": 1000 + index,
                "engine_lifecycle_qualified": True,
                "qualification_blockers": [],
                "checks": {
                    "same_subset_token_exact": True,
                    "dense_engine_oracle_token_exact": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_full_sequence_requires_complete_identity_and_ordinary_parity(tmp_path):
    paths = [_artifact(tmp_path, index) for index in (1, 2, 3)]
    result = summarize(paths, (1, 2, 3), expect_full_retention=True)
    assert result["sequence_qualified"] is True
    assert result["qualified_requests"] == 3
    assert result["weighted_realized_retention_fraction"] == 1.0
    assert all(row["request_qualified"] for row in result["rows"])


def test_reduced_sequence_preserves_weighted_accounting(tmp_path):
    paths = [_artifact(tmp_path, 1, 0.5), _artifact(tmp_path, 2, 0.9)]
    result = summarize(paths, (1, 2), expect_full_retention=False)
    assert result["sequence_qualified"] is True
    assert result["weighted_realized_retention_fraction"] == pytest.approx(160 / 220)
    assert result["visible_context_omission_fraction"] == pytest.approx(60 / 220)
    assert result["cumulative_source_kv_tokens"] == 200
    assert result["cumulative_selected_kv_tokens"] == 140
    assert result["cumulative_wire_suffix_tokens"] == 20
    assert result["weighted_historical_kv_retention_fraction"] == pytest.approx(0.7)
    assert result["historical_kv_omission_fraction"] == pytest.approx(0.3)
    assert result["consumer_peak_temporary_bytes_max"] == 202
    assert result["offloaded_payload_bytes_cumulative"] == 2003


def test_duplicate_request_digest_is_rejected(tmp_path):
    paths = [
        _artifact(tmp_path, 1, digest="same"),
        _artifact(tmp_path, 2, digest="same"),
    ]
    with pytest.raises(ValueError, match="not unique"):
        summarize(paths, (1, 2), expect_full_retention=False)


def test_hf_interval_pack_bytes_maps_to_common_selection_pack_gate(tmp_path):
    paths = [_artifact(tmp_path, 1, 0.5, hf_pack=True)]
    result = summarize(paths, (1,), expect_full_retention=False)
    assert result["sequence_qualified"] is True
    assert result["selection_pack_bytes"] == 0


def _ledger(tmp_path, request_digest="digest-1"):
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(
        json.dumps(
            {
                "request_index": 1,
                "request_input_sha256": request_digest,
                "source_plan_digest": "plan-digest",
                "selected_resource_digest": "resource-digest",
                "source_wire_plan_digest": "wire-digest",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    replay = tmp_path / "replay.jsonl"
    replay.write_text(
        json.dumps(
            {"request_index": 1, "request_input_sha256": request_digest}
        )
        + "\n",
        encoding="utf-8",
    )
    return fixture, replay


def test_summary_binds_frozen_selector_identity(tmp_path):
    artifact = _artifact(tmp_path, 1)
    fixture, replay = _ledger(tmp_path)
    result = summarize(
        [artifact],
        (1,),
        expect_full_retention=False,
        selection_fixture=fixture,
        request_replay=replay,
    )
    assert result["frozen_ledger_bound"] is True
    assert result["rows"][0]["source_plan_digest"] == "plan-digest"
    assert result["frozen_ledger_files"]["selection_fixture_sha256"]


def test_summary_rejects_artifact_outside_frozen_ledger(tmp_path):
    artifact = _artifact(tmp_path, 1)
    fixture, replay = _ledger(tmp_path, request_digest="other-request")
    with pytest.raises(ValueError, match="does not match frozen ledger"):
        summarize(
            [artifact],
            (1,),
            expect_full_retention=False,
            selection_fixture=fixture,
            request_replay=replay,
        )


def test_summary_requires_fixture_and_replay_together(tmp_path):
    artifact = _artifact(tmp_path, 1)
    fixture, _replay = _ledger(tmp_path)
    with pytest.raises(ValueError, match="supplied together"):
        summarize(
            [artifact],
            (1,),
            expect_full_retention=False,
            selection_fixture=fixture,
        )


def test_partial_summary_can_bind_prefix_of_complete_frozen_ledger(tmp_path):
    artifact = _artifact(tmp_path, 1)
    fixture_rows = []
    replay_rows = []
    for index in (1, 2):
        digest = f"digest-{index}"
        fixture_rows.append(
            {
                "request_index": index,
                "request_input_sha256": digest,
                "source_plan_digest": f"plan-{index}",
                "selected_resource_digest": f"resource-{index}",
                "source_wire_plan_digest": f"wire-{index}",
            }
        )
        replay_rows.append(
            {"request_index": index, "request_input_sha256": digest}
        )
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(
        "\n".join(json.dumps(row) for row in fixture_rows) + "\n",
        encoding="utf-8",
    )
    replay = tmp_path / "replay.jsonl"
    replay.write_text(
        "\n".join(json.dumps(row) for row in replay_rows) + "\n",
        encoding="utf-8",
    )
    result = summarize(
        [artifact],
        (1,),
        expect_full_retention=False,
        selection_fixture=fixture,
        request_replay=replay,
    )
    assert result["sequence_qualified"] is True
    assert result["rows"][0]["source_plan_digest"] == "plan-1"
