import json

import pytest

from experiments.paper4_5_agent.summarize_frozen_request_sequence import summarize


def _artifact(tmp_path, index, retention=1.0, digest=None):
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
                "selection_pack_bytes": 0,
                "physical_kv_copy": False,
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


def test_reduced_sequence_preserves_weighted_accounting(tmp_path):
    paths = [_artifact(tmp_path, 1, 0.5), _artifact(tmp_path, 2, 0.9)]
    result = summarize(paths, (1, 2), expect_full_retention=False)
    assert result["sequence_qualified"] is True
    assert result["weighted_realized_retention_fraction"] == pytest.approx(160 / 220)


def test_duplicate_request_digest_is_rejected(tmp_path):
    paths = [
        _artifact(tmp_path, 1, digest="same"),
        _artifact(tmp_path, 2, digest="same"),
    ]
    with pytest.raises(ValueError, match="not unique"):
        summarize(paths, (1, 2), expect_full_retention=False)
