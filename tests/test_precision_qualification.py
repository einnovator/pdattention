from __future__ import annotations

import pytest

from experiments.paper4_5_runtime.build_precision_ladder import build_ladder, write_ladder
from experiments.paper4_5_runtime.analyze_precision_rag import build_comparison
from pra_hf.precision import (
    MemoryGateObservation,
    MemoryGateStatus,
    PrecisionDescriptor,
    classify_memory_gate,
    infer_precision,
)
from pra_hf.precision_qualification import (
    PrecisionQualificationRequest,
    PrecisionQualificationService,
)


class _Inspector:
    def inspect(self, model_id: str, *, revision: str | None = None) -> dict:
        return {
            "model": {"id": model_id, "revision": revision or "a" * 40},
            "attention": {"layers": 36},
            "pra": {"native_kv": True},
        }


def test_precision_identity_keeps_family_and_encoding_separate() -> None:
    mlx = infer_precision({"bits": 4, "runtime": "MLX"}, engine="mlx-lm")
    awq = infer_precision({"bits": 4, "runtime": "AWQ"}, engine="vllm")

    assert mlx.precision_family == awq.precision_family == "INT4"
    assert mlx.precision_encoding == "MLX-4bit"
    assert awq.precision_encoding == "AWQ-4bit"
    assert mlx != awq


def test_string_and_mapping_forms_resolve_same_mlx_identity() -> None:
    from_mapping = infer_precision({"bits": 8, "runtime": "MLX"}, engine="mlx-lm")
    from_string = infer_precision("8bit", engine="mlx-lm")
    assert from_mapping.precision_encoding == from_string.precision_encoding == "MLX-8bit"


def test_torch_dtype_has_one_runtime_prefix() -> None:
    precision = infer_precision("torch.float16", engine="hf")
    assert precision.precision_family == "FP16"
    assert precision.precision_encoding == "PyTorch-float16"


def test_precision_descriptor_rejects_unknown_family_or_missing_encoding() -> None:
    with pytest.raises(ValueError, match="Unknown precision family"):
        PrecisionDescriptor("INT3", "custom")
    with pytest.raises(ValueError, match="encoding"):
        PrecisionDescriptor("BF16", "")


def test_memory_gate_classifies_load_context_and_workload_outcomes() -> None:
    assert classify_memory_gate([]) == MemoryGateStatus.LOAD_ONLY
    assert classify_memory_gate(
        [MemoryGateObservation("load_only", False, note="OOM")]
    ) == MemoryGateStatus.BLOCKED_MEMORY
    assert classify_memory_gate(
        [
            MemoryGateObservation("load_only", True),
            MemoryGateObservation("context_2k", True),
            MemoryGateObservation("context_8k", False, note="OOM"),
        ]
    ) == MemoryGateStatus.CONTEXT_LIMITED
    assert classify_memory_gate(
        [
            MemoryGateObservation("load_only", True),
            MemoryGateObservation("target_rag_context", True),
        ]
    ) == MemoryGateStatus.QUALIFIABLE


def test_qualification_service_emits_all_publication_artifacts(tmp_path) -> None:
    gate = tmp_path / "memory.json"
    gate.write_text(
        '{"observations":[{"stage":"load_only","succeeded":true},'
        '{"stage":"target_rag_context","succeeded":true}]}',
        encoding="utf-8",
    )
    request = PrecisionQualificationRequest(
        model_id="Qwen/Qwen3-4B",
        revision="a" * 40,
        tokenizer_revision="b" * 40,
        engine="mlx-lm",
        engine_version="0.31.3",
        dataset="multihop-rag",
        profile="balanced",
        mode="native-memory",
        precision=PrecisionDescriptor("BF16", "MLX-bfloat16"),
        feature_extraction_precision="BF16",
        adaptor_parameter_precision="FP32",
        date="2026-09-04",
    )
    output = tmp_path / "qualification"

    result = PrecisionQualificationService(_Inspector()).qualify(
        request, output=output, memory_gate=gate
    )

    assert result["memory_gate"]["status"] == "QUALIFIABLE"
    assert result["canonical_evidence"]["key"]["precision_family"] == "BF16"
    assert result["canonical_evidence"]["conditions"]["no_pra"]["metrics"]["token_f1"]["state"] == "NEEDS_RUN"
    assert result["canonical_evidence"]["conditions"]["pra_adaptor_bundle"]["metrics"]["token_f1"]["state"] == "NO_QUALIFIED_ADAPTER"
    for name in (
        "evidence.json", "manifest.yaml", "precision_matrix.csv",
        "card_fragment.md", "table.tex",
    ):
        assert (output / name).is_file()


def test_qualification_rejects_cross_precision_evidence(tmp_path) -> None:
    request = PrecisionQualificationRequest(
        model_id="Qwen/Qwen3-4B",
        revision="a" * 40,
        tokenizer_revision=None,
        engine="mlx-lm",
        engine_version="0.31.3",
        dataset="multihop-rag",
        profile="balanced",
        mode="native-memory",
        precision=PrecisionDescriptor("INT8", "MLX-8bit"),
    )
    first = PrecisionQualificationService(_Inspector()).qualify(
        request, output=tmp_path / "first"
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(__import__("json").dumps(first), encoding="utf-8")
    mismatched = PrecisionQualificationRequest(
        **{
            **request.__dict__,
            "precision": PrecisionDescriptor("INT4", "MLX-4bit"),
        }
    )
    with pytest.raises(ValueError, match="precision evidence identity mismatch"):
        PrecisionQualificationService(_Inspector()).qualify(
            mismatched, output=tmp_path / "second", evidence=evidence
        )


def test_precision_ladder_keeps_missing_cells_explicit(tmp_path) -> None:
    payload = build_ladder()

    assert payload["summary"]["rows"] >= 50
    assert payload["summary"]["measured_variants"] > 0
    assert payload["summary"]["needs_run"] > payload["summary"]["measured_variants"]
    qwen4 = [
        row for row in payload["rows"]
        if row["base_model"] == "Qwen/Qwen3-4B"
    ]
    assert {row["precision_family"] for row in qwen4} == {
        "FP32", "BF16", "INT8", "INT4"
    }
    assert next(row for row in qwen4 if row["precision_family"] == "FP32")[
        "condition_no_pra"
    ] == "NEEDS_RUN"

    write_ladder(payload, tmp_path)
    assert (tmp_path / "precision_ladder.json").is_file()
    assert (tmp_path / "precision_matrix.csv").is_file()
    assert "MLX-8bit" in (tmp_path / "generated_precision_ladder.tex").read_text(
        encoding="utf-8"
    )


def test_matched_precision_rag_keeps_conversion_and_missing_adaptor_explicit() -> None:
    payload = build_comparison()
    rows = payload["rows"]

    assert len(rows) == 6
    assert {row["precision_encoding"] for row in rows} == {
        "MLX-bfloat16", "MLX-8bit", "MLX-4bit"
    }
    assert {row["examples"] for row in rows} == {10}
    assert {row["seed"] for row in rows} == {11}
    assert all(row["adaptor_condition"] == "NO_QUALIFIED_ADAPTER" for row in rows)
    assert all(row["peak_memory_bytes"] is None for row in rows)
