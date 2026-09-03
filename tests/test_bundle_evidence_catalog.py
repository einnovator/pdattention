from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pra_hf.bundle import BundleBuilder, BundleValidationError, PRAModelBundle
from pra_hf.bundle_catalog import (
    load_bundle_catalog,
    render_canonical_evidence_catalog,
    render_catalog,
    render_qualification_matrix,
    validate_collection_membership,
)
from pra_hf.bundle_evidence import (
    EvidenceIdentity,
    EvidenceValidationError,
    canonicalize_paired_transport_evidence,
    import_matched_e0_e2_evidence,
    import_mlx_paired_evidence,
    import_product_matrix_evidence,
    validate_selector_manifest,
)
from pra_hf.canonical_evidence import EvidenceCondition, MeasurementState
from experiments.paper4_5_runtime.build_canonical_evidence_audit import build_audit
from experiments.paper4_5_runtime.run_exact_identity_qualification import build_command


ROOT = Path(__file__).resolve().parents[1]
MLX_32B = ROOT / "docs/papers/shared/results/mac_scaling/qwen3_32b_mlx_profiles.json"


def _identity(**overrides: str) -> EvidenceIdentity:
    values = {
        "model_id": "mlx-community/Qwen3-32B-4bit",
        "model_revision": "bcaaf7f538adf166c1080a2befdb4f6019f66639",
        "quantization": "4bit",
        "engine": "mlx-lm",
        "engine_version": "0.31.3",
        "profile": "balanced",
        "execution_mode": "Native Memory",
    }
    values.update(overrides)
    return EvidenceIdentity(**values)


def test_mlx_importer_builds_paired_baseline_relative_evidence() -> None:
    rows = import_mlx_paired_evidence(MLX_32B, _identity())
    combined = next(row for row in rows if row["dataset"] == "combined")

    assert combined["metric_class"] == "END_TASK"
    assert combined["baseline"]["quality"] == combined["pra"]["quality"]
    assert combined["deltas"]["quality"] == 0.0
    assert combined["deltas"]["visible_tokens_pct"] == pytest.approx(-89.14008)
    assert combined["baseline"]["itl_ms"]["p95"] > 0
    assert combined["pra"]["output_tokens_per_second"] > 0
    assert combined["semantic_equivalence"] == {
        "exact_output_pairs": 15,
        "paired_examples": 15,
    }
    assert combined["evidence_tier"] == "ENGINE_QUALIFIED"
    assert len(combined["artifact_sha256"]) == 64

    canonical = canonicalize_paired_transport_evidence(combined)
    assert canonical.delta("visible_tokens", EvidenceCondition.PRA_NO_ADAPTOR).percent_delta == pytest.approx(-89.14008)
    assert canonical.conditions[EvidenceCondition.NO_PRA].metrics["itl_p95_ms"].value > 0
    assert canonical.conditions[EvidenceCondition.PRA_NO_ADAPTOR].metrics["output_tokens_per_second"].value > 0
    assert canonical.conditions[EvidenceCondition.PRA_ADAPTOR_BUNDLE].metrics["token_f1"].state == MeasurementState.NOT_MEASURED
    assert canonical.key.model_revision == _identity().model_revision


def test_mlx_importer_rejects_revision_and_mode_mismatch() -> None:
    with pytest.raises(EvidenceValidationError, match="model_revision"):
        import_mlx_paired_evidence(MLX_32B, _identity(model_revision="wrong"))
    with pytest.raises(EvidenceValidationError, match="BALANCED Native Memory"):
        import_mlx_paired_evidence(MLX_32B, _identity(execution_mode="Selected Context"))


def test_shared_matched_importer_uses_one_cold_row_per_selection(tmp_path: Path) -> None:
    def row(condition: str, selection_id: str, token_f1: float) -> dict:
        native = condition == "e2_native_kv"
        return {
            "condition": condition,
            "regime": "cold_one_shot",
            "request_ordinal": 0,
            "query_sha256": "q" * 64,
            "selection": {"selection_id": selection_id},
            "output": "answer",
            "metrics": {
                "quality": {
                    "exact_match": token_f1,
                    "token_f1": token_f1,
                    "gold_answer_logprob": -1.0,
                    "evidence_recall": 1.0,
                },
                "input": {"visible_prompt_tokens": 40 if native else 140},
                "pra": {
                    "selected_native_kv_tokens": 100 if native else 0,
                    "active_detail_bytes": 1024 if native else 0,
                    "retained_detail_bytes": 1024 if native else 0,
                },
                "serving": {
                    "ttft_ms": 10.0 if native else 20.0,
                    "itl_ms": 2.0,
                    "total_latency_ms": 30.0,
                    "tokens_per_second": 50.0,
                    "requests_per_second": None,
                },
            },
            "extra": {"seed": 11},
        }

    payload = {
        "schema_version": "2.0",
        "engine": "mlx-lm",
        "engine_version": "0.31.3",
        "model_id": _identity().model_id,
        "model_revision": _identity().model_revision,
        "dataset": "qasper",
        "rows": [
            row("e0_selected_text", "selection-1", 0.5),
            row("e2_native_kv", "selection-1", 0.5),
            # Reuse rows must not be counted as extra quality examples.
            {**row("e0_selected_text", "selection-1", 0.0), "regime": "warm_repeated"},
            {**row("e2_native_kv", "selection-1", 0.0), "regime": "warm_repeated"},
        ],
    }
    artifact = tmp_path / "matched.json"
    artifact.write_text(__import__("json").dumps(payload), encoding="utf-8")

    imported = import_matched_e0_e2_evidence(
        [artifact],
        _identity(),
        hardware="Apple M4 Pro, 48 GB",
        evidence_date="2026-09-03",
    )
    combined = next(item for item in imported if item["dataset"] == "combined")

    assert combined["sample_count"] == 1
    assert combined["baseline"]["quality"] == 0.5
    assert combined["pra"]["quality"] == 0.5
    assert combined["deltas"]["visible_tokens_pct"] == pytest.approx(-71.42857)
    assert combined["semantic_equivalence"] == {
        "exact_output_pairs": 1,
        "paired_examples": 1,
    }


def test_shared_matched_importer_rejects_unpaired_or_wrong_identity(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": "2.0",
        "engine": "mlx-lm",
        "engine_version": "0.31.3",
        "model_id": _identity().model_id,
        "model_revision": _identity().model_revision,
        "dataset": "qasper",
        "rows": [],
    }
    artifact = tmp_path / "matched.json"
    artifact.write_text(__import__("json").dumps(payload), encoding="utf-8")
    with pytest.raises(EvidenceValidationError, match="cohort mismatch"):
        import_matched_e0_e2_evidence(
            [artifact], _identity(), hardware="M4", evidence_date="2026-09-03"
        )
    with pytest.raises(EvidenceValidationError, match="model_revision"):
        import_matched_e0_e2_evidence(
            [artifact],
            _identity(model_revision="wrong"),
            hardware="M4",
            evidence_date="2026-09-03",
        )


def test_shared_product_matrix_and_selector_manifest_importers() -> None:
    rows = import_product_matrix_evidence(
        ROOT / "docs/papers/shared/results/pra_product_matrix_v2.json",
        EvidenceIdentity(
            model_id="Qwen/Qwen3-0.6B",
            model_revision="c1899de289a04d12100db370d81485cdf75e47ca",
            quantization="torch.float16",
            engine="huggingface_eager",
            engine_version="",
            profile="REFERENCE_CORRECTNESS",
            execution_mode="Native Memory",
        ),
    )
    assert rows
    assert all(row["model_revision"] == "c1899de289a04d12100db370d81485cdf75e47ca" for row in rows)
    manifest = validate_selector_manifest(
        ROOT / "docs/papers/shared/results/engine_qualification/qualification_manifest.json"
    )
    assert manifest["selections"][0]["digest"]


def test_release_gate_rejects_profile_and_headline_conflicts() -> None:
    row = import_mlx_paired_evidence(MLX_32B, _identity())[-1]
    bundle = PRAModelBundle(
        base_model={
            "id": _identity().model_id,
            "revision": _identity().model_revision,
            "fingerprint": "f" * 64,
            "quantization": {"bits": 4, "runtime": "MLX"},
        },
        structural_adapter={},
        profiles={"balanced": {"status": "QUALIFIED", "recommended": True}},
        qualification={"contract_version": 1, "status": "ENGINE_QUALIFIED", "headline": [row]},
    )
    assert bundle.validate(require_card=False)["status"] == "VALID"

    with pytest.raises(BundleValidationError, match="exactly one recommended profile"):
        replace(bundle, profiles={"balanced": {"status": "QUALIFIED"}}).validate(require_card=False)
    with pytest.raises(BundleValidationError, match="routing diagnostics cannot be headline"):
        replace(bundle, qualification={**bundle.qualification, "headline": [{**row, "metric_class": "ROUTING_DIAGNOSTIC"}]}).validate(require_card=False)

    canonical = {
        "schema_version": 1,
        "key": {
            "task": "qasper", "hardware": "m5", "engine": "mlx-lm",
            "engine_version": "0.31.3", "model_id": "wrong/model",
            "model_revision": _identity().model_revision, "mode": "native-memory",
            "profile": "balanced",
        },
        "metric_definitions": {},
        "conditions": {
            "no_pra": {"metrics": {}},
            "pra_no_adaptor": {"metrics": {}},
            "pra_adaptor_bundle": {"metrics": {}},
        },
        "provenance": {"cohort": "test", "date": "2026-09-03"},
        "evidence_tier": "CONTROLLED",
    }
    with pytest.raises(BundleValidationError, match="canonical evidence model_id"):
        replace(bundle, qualification={**bundle.qualification, "canonical_evidence": [canonical]}).validate(require_card=False)


def test_generated_32b_card_leads_with_pairing_not_router_recall() -> None:
    bundle = PRAModelBundle.from_pretrained(
        ROOT / "artifacts/pra_hf/bundles/pra-qwen3-32b-mlx-4bit"
    )
    text = BundleBuilder.model_card(bundle)

    assert "# PRA Runtime Bundle for" in text
    assert "Recommended profile: **BALANCED**" in text
    assert "15/15" in text
    assert "-89.1%" in text
    assert "## Evidence by engine, mode, and profile" in text
    assert "| mlx | Native Memory | BALANCED | MEASURED" in text
    assert "Output Tokens Per Second" in text
    assert "ITL p95 (ms)" in text
    assert "SHA-256" in text
    assert "70.32 MiB" in text
    assert "## Expected metrics" not in text
    assert "R@20%" not in text.split("## Research diagnostics", 1)[0]


def test_catalog_order_reference_role_and_collection_membership() -> None:
    catalog = load_bundle_catalog()
    rows = catalog["bundles"]

    assert [row["order"] for row in rows] == list(range(1, len(rows) + 1))
    assert rows[0]["repo"] == "EInnovator/pra-qwen3-14b-mlx-4bit"
    assert rows[1]["repo"] == "EInnovator/pra-qwen3-32b-mlx-4bit"
    reference = next(row for row in rows if row["repo"].endswith("pra-qwen3-0.6b"))
    assert reference["evidence_tier"] == "RESEARCH"
    assert "reference" in reference["role"].lower()
    assert any("coder" in row["model"].lower() for row in rows)
    assert any("instruct" in row["model"].lower() for row in rows)
    matrix = render_qualification_matrix(catalog)
    assert "Qualification Matrix" in matrix
    assert "Canonical condition audit" in matrix
    assert "PRA - No Adaptor" in matrix
    assert "PRA Runtime Bundle Catalog" in render_catalog(catalog)

    published = {row["repo"] for row in rows if row["publication_status"] == "PUBLISHED"}
    validate_collection_membership(catalog, published)
    with pytest.raises(ValueError, match="Collection is missing"):
        validate_collection_membership(catalog, published - {next(iter(published))})


def test_catalog_canonical_audit_never_encodes_missing_as_zero() -> None:
    catalog = load_bundle_catalog()
    audit = build_audit(catalog)
    assert len(audit["rows"]) == len(catalog["bundles"])
    assert audit["summary"]["AVAILABLE_EXISTING"] > 0
    assert audit["summary"]["NEEDS_RUN"] > audit["summary"]["AVAILABLE_EXISTING"]
    states = {
        state
        for row in audit["rows"]
        for conditions in row["metrics"].values()
        for state in conditions.values()
    }
    assert states <= {"AVAILABLE_EXISTING", "NEEDS_RUN", "NOT_APPLICABLE", "BLOCKED"}
    assert 0 not in states


@pytest.mark.parametrize(
    ("bundle_name", "bits", "runtime"),
    [
        ("pra-qwen3-8b-mlx-6bit", 6, "MLX"),
        ("pra-gemma3-1b-mlx-8bit", 8, "MLX"),
        ("pra-qwen2-5-1-5b-instruct-bnb-8bit", 8, "bitsandbytes/PyTorch"),
    ],
)
def test_quantized_bundles_preserve_exact_identity_without_transferred_evidence(
    bundle_name: str, bits: int, runtime: str
) -> None:
    bundle = PRAModelBundle.from_pretrained(
        ROOT / "artifacts/pra_hf/bundles" / bundle_name
    )

    assert bundle.base_model["quantization"]["bits"] == bits
    assert bundle.base_model["quantization"]["runtime"] == runtime
    assert bundle.qualification["status"] == "SMOKE"
    assert bundle.qualification["headline"] == []
    assert bundle.learned_adapters == {}
    assert bundle.validate()["status"] == "VALID"
    card = BundleBuilder.model_card(bundle)
    assert "No learned router is bundled" in card
    assert "NO_QUALIFIED_ADAPTER" in card
    assert "NEEDS_RUN" in card
    assert "Exact-identity runtime smoke" in card
    assert "RUNTIME_SMOKE_VALIDATED" in card
    assert bundle.qualification["runtime_smoke"]["status"] == "RUNTIME_SMOKE_VALIDATED"


def test_exact_8bit_learned_router_is_opt_in_and_keeps_smoke_scope_separate() -> None:
    bundle = PRAModelBundle.from_pretrained(
        ROOT / "artifacts/pra_hf/bundles/pra-qwen3-4b-mlx-8bit"
    )

    assert bundle.base_model["quantization"]["bits"] == 8
    assert bundle.qualification["status"] == "CONTROLLED"
    assert bundle.qualification["headline"] == []
    assert "combined-router-d128" in bundle.learned_adapters
    assert bundle.profiles["balanced"]["routing_adapter"] is None
    assert bundle.profiles["qasper-learned"]["routing_adapter"] == "combined-router-d128"
    assert bundle.qualification["runtime_smoke"]["status"] == "RUNTIME_SMOKE_VALIDATED"


def test_canonical_evidence_catalog_covers_every_model_profile_and_engine_metric() -> None:
    text = render_canonical_evidence_catalog(load_bundle_catalog())
    for row in load_bundle_catalog()["bundles"]:
        assert row["model"] in text
    assert "PRA - No Adaptor" in text
    assert "PRA - Adaptor Bundle" in text
    assert "Output Tokens Per Second" in text
    assert "TTFT p95 (ms)" in text
    assert "ITL p95 (ms)" in text
    assert "Delta Bundle" in text


def test_published_cards_use_actionable_coverage_states() -> None:
    cards = sorted((ROOT / "artifacts/pra_hf/bundles").glob("*/README.md"))
    assert cards
    for card in cards:
        text = card.read_text(encoding="utf-8")
        assert "NOT_MEASURED" not in text, card
        assert any(
            state in text
            for state in (
                "MEASURED",
                "NEEDS_RUN",
                "CALIBRATION_PENDING",
                "NO_QUALIFIED_ADAPTER",
            )
        ), card


def test_exact_identity_runner_pins_bundle_model_and_revision() -> None:
    command, output = build_command(
        "llama3.2-1b-mlx-8bit",
        "qasper",
        max_examples=20,
        concurrency=4,
    )

    assert command[command.index("--model") + 1] == (
        "mlx-community/Llama-3.2-1B-Instruct-8bit"
    )
    assert command[command.index("--revision") + 1] == (
        "d48cdf0a4ea22d893b7c63a99d6a693e24822795"
    )
    assert command[command.index("--max-examples") + 1] == "20"
    assert output.name == "matched_e0_e2_qasper.json"
