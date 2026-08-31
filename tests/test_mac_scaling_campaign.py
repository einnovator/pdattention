from __future__ import annotations

import json
from pathlib import Path

from experiments.mac_scaling.run_mlx_profile_scaling import resolve_consumer_layers


ROOT = Path(__file__).resolve().parents[1]


def test_consumer_profiles_are_model_normalized_suffixes() -> None:
    assert len(resolve_consumer_layers(36, "last_3_4")) == 27
    assert len(resolve_consumer_layers(36, "last_2_3")) == 24
    assert resolve_consumer_layers(36, "last_1_4") == tuple(range(27, 36))


def test_campaign_pins_large_model_revisions_and_evidence_tier() -> None:
    payload = json.loads(
        (ROOT / "experiments/mac_scaling/campaign.json").read_text(encoding="utf-8")
    )
    models = payload["primary_models"]

    assert [row["key"] for row in models] == [
        "qwen3_8b_q4",
        "qwen3_14b_q4",
        "qwen3_32b_q4",
        "qwen3_30b_a3b_q4",
    ]
    assert all(len(row["revision"]) == 40 for row in models)
    assert payload["evidence_tiers"]["promotion"].endswith("REQUIRED")
