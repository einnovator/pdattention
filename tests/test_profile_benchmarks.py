from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from pra_hf.cli import cli
from pra_hf.config import PRAConfig
from pra_hf.profile_benchmarks import (
    MeasurementStatus,
    ProfileBenchmarkRegistry,
    normalized_quality,
    normalized_saving,
)
from pra_hf.runtime import PRARuntimeConfig


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = (
    ROOT
    / "docs/papers/shared/results/paper4_5_runtime/pra_profile_benchmarks.json"
)
PACKAGED = ROOT / "src/pra_hf/model_profiles/pra_profile_benchmarks.json"


def test_packaged_registry_matches_canonical_evidence() -> None:
    assert PACKAGED.read_bytes() == CANONICAL.read_bytes()
    registry = ProfileBenchmarkRegistry.default()
    assert len(registry.rows) == 12
    assert {
        row["profile"] for row in registry.find("Qwen/Qwen3-0.6B")
    } == {
        "REFERENCE_CORRECTNESS",
        "QUALITY_MAX_CANDIDATE",
        "BALANCED",
        "ECONOMY",
    }
    candidate = registry.resolve(
        "Qwen/Qwen3-0.6B",
        workload="semantic_smoke",
        profile="QUALITY_MAX_CANDIDATE",
    ).row
    assert candidate["evidence_tier"] == "SMOKE"
    assert candidate["profile_status"] == "CALIBRATION_PENDING"
    assert candidate["measurement_status"] == "MEASURED"
    with pytest.raises(KeyError):
        registry.resolve(
            "Qwen/Qwen3-0.6B",
            workload="semantic_smoke",
            profile="QUALITY_MAX",
        )


def test_registry_recomputes_normalized_metrics_and_rejects_stale_values() -> None:
    retention, delta = normalized_quality(0.75, 0.50)
    assert retention == 1.5
    assert delta == 0.25
    assert normalized_saving(25, 100) == 0.75

    payload = json.loads(CANONICAL.read_text(encoding="utf-8"))
    stale = copy.deepcopy(payload)
    stale["benchmarks"][0]["quality_retention"] = 999
    with pytest.raises(ValueError, match="stale quality_retention"):
        ProfileBenchmarkRegistry(stale)


def test_missing_workload_is_explicitly_calibration_pending() -> None:
    report = ProfileBenchmarkRegistry.default().inspect(
        "Qwen/Qwen3-0.6B", workload="typed_records"
    )
    assert report["profiles"] == []
    assert report["measurement_status"] == MeasurementStatus.CALIBRATION_PENDING.value


def test_product_profile_alias_preserves_explicit_layer_overrides() -> None:
    config = PRAConfig(
        profile="economy",
        workload="semantic_smoke",
        model_id="Qwen/Qwen3-0.6B",
        routing_layers=(3,),
        detail_kv_layers=(3, 5),
        consumption_layers=(5,),
    )
    roles = config.resolved_layer_roles(8)
    assert config.layer_profile_objective == "economy"
    assert config.workload_class == "semantic_smoke"
    assert roles.routing_layers == (3,)
    assert roles.detail_kv_layers == (3, 5)
    assert roles.consumption_layers == (5,)
    trace = config.product_profile_trace()
    assert trace["profile_requested"] == "ECONOMY"
    assert trace["profile_resolved"] == "ECONOMY"
    assert trace["registry_version"] == "2026-08-product-profile-v2"


def test_quality_max_candidate_maps_to_internal_objective_but_production_label_is_reserved() -> None:
    candidate = PRAConfig(
        profile="quality_max_candidate",
        workload="semantic_smoke",
        model_id="Qwen/Qwen3-0.6B",
    )
    production = PRAConfig(
        profile="quality_max",
        workload="semantic_smoke",
        model_id="Qwen/Qwen3-0.6B",
    )

    assert candidate.layer_profile_objective == "quality_max"
    assert candidate.product_profile_trace()["profile_status"] == "CALIBRATION_PENDING"
    assert production.product_profile_trace()["profile_status"] == "CALIBRATION_PENDING"
    assert production.product_profile_trace()["profile_resolved"] == "QUALITY_MAX"


def test_runtime_convenience_aliases_round_trip() -> None:
    config = PRARuntimeConfig(profile="balanced", workload="semantic_smoke")
    restored = PRARuntimeConfig.from_dict(config.to_dict())
    assert restored.profile == "balanced"
    assert restored.pra.profile == "BALANCED"
    assert restored.pra.workload_class == "semantic_smoke"


def test_profiles_show_cli_reports_null_runtime_metrics_honestly() -> None:
    result = CliRunner().invoke(
        cli,
        [
            "profiles",
            "show",
            "Qwen/Qwen3-0.6B",
            "--workload",
            "semantic_smoke",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert len(payload["profiles"]) == 4
    assert payload["profiles"][0]["runtime"]["ttft_ms"] == "NOT_MEASURED"
    assert payload["profiles"][0]["evidence_tier"] == "SMOKE"
    candidate = next(
        row
        for row in payload["profiles"]
        if row["profile"] == "QUALITY_MAX_CANDIDATE"
    )
    assert candidate["profile_status"] == "CALIBRATION_PENDING"
