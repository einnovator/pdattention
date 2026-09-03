from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from pra_hf.cli import cli
from pra_hf.product_qualification import (
    EngineProductRegistry,
    QualificationService,
    load_run,
    render_report,
)


def _write_measurements(path, *, selected_success=True, native=False, native_speedup=True) -> None:
    modes = {
        "full_context": {
            "quality": {"f1": 0.80},
            "context": {"visible_input_tokens": 1000},
            "performance": {"ttft_p95_ms": 200.0, "successful_requests_per_second": 5.0},
        },
        "selected_context": {
            "quality": {"success": selected_success, "f1": 0.80},
            "context": {"visible_input_tokens": 250},
            "performance": {"ttft_p95_ms": 100.0, "successful_requests_per_second": 8.0},
        },
    }
    if native:
        hot_ttft = 80.0 if native_speedup else 130.0
        modes.update({
            "native_memory_hot": {
                "quality": {"success": True, "f1": 0.80},
                "performance": {"ttft_p95_ms": hot_ttft, "successful_requests_per_second": 9.0 if native_speedup else 7.0},
                "memory": {"native_memory_bytes": 4096},
                "lifecycle": {"reference_encoding_ms": 12.0, "reuse": 4},
            },
            "native_memory_warm": {
                "quality": {"success": True, "f1": 0.80},
                "performance": {"ttft_p95_ms": hot_ttft},
                "memory": {"native_memory_bytes": 4096},
                "lifecycle": {"reference_encoding_ms": 0.0, "reuse": 4},
            },
        })
    path.write_text(json.dumps({
        "selector_digest": "frozen-selection",
        "hardware": "test CPU",
        "evidence_tier": "Measured",
        "modes": modes,
    }), encoding="utf-8")


def test_engines_command_matches_shared_registry() -> None:
    result = CliRunner().invoke(cli, ["engines", "--json"])

    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    expected = {row["slug"] for row in EngineProductRegistry.default().engines}
    assert {row["engine"] for row in value["engines"]} == expected
    assert value["provenance"].endswith("engine_documentation_registry.json")

    details = CliRunner().invoke(cli, ["engines", "--details", "airllm", "--json"])
    detail_value = json.loads(details.output)
    assert details.exit_code == 0, details.output
    assert detail_value["capabilities"]["native_memory"] == "Research-only"
    assert detail_value["registry_version"] == value["registry_version"]


def test_evaluate_writes_stable_assessment_and_adjacent_gains(tmp_path) -> None:
    measurements = tmp_path / "measurements.json"
    _write_measurements(measurements)
    run = tmp_path / "run"

    result = CliRunner().invoke(cli, [
        "evaluate", "org/model", "--engine", "mlx", "--dataset", "qasper",
        "--measurements", str(measurements), "-o", str(run), "--json",
    ])

    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    assert value["recommendation"]["recommended_mode"] == "Selected Context"
    assert value["attribution"]["context_gain"]["visible_token_reduction"] == 0.75
    assert value["attribution"]["context_gain"]["ttft_speedup"] == 2.0
    assert value["attribution"]["native_gain"]["ttft_speedup"] is None
    for name in ("config.yaml", "environment.json", "quality.json", "metrics.json", "report.md", "recommendation.json"):
        assert (run / name).is_file()
    assert (run / "runs").is_dir()


def test_quality_failure_never_produces_a_recommendation(tmp_path) -> None:
    measurements = tmp_path / "failed.json"
    _write_measurements(measurements, selected_success=False)
    document = QualificationService().evaluate(
        "org/model", engine="mlx", dataset="qasper", output=tmp_path / "run",
        measurements=measurements,
    )

    assert document["quality_gate"]["passed"] is False
    assert document["recommendation"]["recommended_mode"] is None
    assert document["recommendation"]["status"] == "Blocked"


def test_native_requires_complete_positive_incremental_economics(tmp_path) -> None:
    positive = tmp_path / "positive.json"
    negative = tmp_path / "negative.json"
    _write_measurements(positive, native=True, native_speedup=True)
    _write_measurements(negative, native=True, native_speedup=False)

    promoted = QualificationService().evaluate(
        "org/model", engine="mlx", dataset="qasper", output=tmp_path / "positive-run",
        measurements=positive, include_native_memory=True,
    )
    retained = QualificationService().evaluate(
        "org/model", engine="mlx", dataset="qasper", output=tmp_path / "negative-run",
        measurements=negative, include_native_memory=True,
    )

    assert promoted["recommendation"]["recommended_mode"] == "Native Memory"
    assert retained["recommendation"]["recommended_mode"] == "Selected Context"


def test_airllm_never_promotes_native_from_negative_registry_policy(tmp_path) -> None:
    measurements = tmp_path / "airllm.json"
    _write_measurements(measurements, native=True, native_speedup=True)

    document = QualificationService().evaluate(
        "org/model", engine="airllm", dataset="qasper", output=tmp_path / "run",
        measurements=measurements, include_native_memory=True,
    )

    assert document["recommendation"]["recommended_mode"] == "Selected Context"
    assert "negative measured economics" in " ".join(document["recommendation"]["limitations"])


def test_missing_metrics_remain_null_and_report_as_not_measured(tmp_path) -> None:
    run = tmp_path / "empty"
    document = QualificationService().evaluate(
        "org/model", engine="openvino", dataset="demo", output=run,
    )

    loaded = load_run(run)
    assert loaded["modes"]["selected_context"]["performance"]["ttft_p95_ms"] is None
    assert document["recommendation"]["recommended_mode"] is None
    assert "NOT_MEASURED" in (run / "report.md").read_text(encoding="utf-8")


def test_assessment_wrapper_creates_and_runs_enterprise_layout(tmp_path) -> None:
    runner = CliRunner()
    root = tmp_path / "assessments"
    initialized = runner.invoke(cli, ["assess", "init", "customer", "--root", str(root)])
    assessment = root / "customer"
    config = assessment / "config.yaml"
    config.write_text(
        "schema_version: '1.0'\nname: customer\nmodel: org/model\nengine: mlx\ndataset: qasper\nprofile: recommended\n",
        encoding="utf-8",
    )
    measurements = tmp_path / "measurements.json"
    _write_measurements(measurements)
    executed = runner.invoke(cli, ["assess", "run", str(assessment), "--measurements", str(measurements), "--json"])

    assert initialized.exit_code == 0, initialized.output
    assert executed.exit_code == 0, executed.output
    assert (assessment / "report.md").is_file()
    assert (assessment / "recommendation.json").is_file()


def test_report_accepts_canonical_agent_evidence() -> None:
    artifact = (
        Path(__file__).parents[1]
        / "docs/papers/shared/results/paper4_5_runtime_productization/coding_agents/qwen3_14b_canonical_evidence.json"
    )
    document = load_run(artifact)
    markdown = render_report(document, "md")
    html = render_report(document, "html")

    assert "PRA - No Adaptor" in markdown
    assert "BLOCKED" in markdown
    assert "PRA - Adaptor Bundle" in html
    assert "No-PRA official success is zero" in artifact.read_text(encoding="utf-8")
