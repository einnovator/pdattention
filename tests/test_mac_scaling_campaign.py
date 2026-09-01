from __future__ import annotations

import json
from pathlib import Path

from experiments.mac_scaling import run_mlx_profile_scaling
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


def test_dense_and_moe_models_share_the_preregistered_profile_space() -> None:
    payload = json.loads(
        (ROOT / "experiments/mac_scaling/campaign.json").read_text(encoding="utf-8")
    )
    dense = next(row for row in payload["primary_models"] if row["key"] == "qwen3_32b_q4")
    moe = next(
        row for row in payload["primary_models"] if row["key"] == "qwen3_30b_a3b_q4"
    )

    assert dense["parameters_billion"] > moe["parameters_billion"]
    assert moe["active_parameters_billion"] < 4
    assert payload["consumer_profiles"][0] == "all_layers"


def test_runtime_metadata_labels_declared_m4_pro_bandwidth(monkeypatch) -> None:
    values = {
        ("system_profiler", "SPHardwareDataType", "-detailLevel", "mini"):
            "Hardware:\n    Chip: Apple M4 Pro",
        ("sysctl", "-n", "hw.model"): "Mac16,7",
        ("sysctl", "-n", "hw.memsize"): "51539607552",
        ("sysctl", "-n", "iogpu.wired_limit_mb"): "0",
        ("git", "rev-parse", "HEAD"): "abc123",
    }
    monkeypatch.setattr(
        run_mlx_profile_scaling,
        "_command_value",
        lambda command: values.get(tuple(command)),
    )

    metadata = run_mlx_profile_scaling.runtime_metadata()

    assert metadata["chip"] == "Apple M4 Pro"
    assert metadata["declared_memory_bandwidth_gbps"] == 273
    assert metadata["memory_bandwidth_source"] == "Apple chip specification"
