from __future__ import annotations

import dataclasses

import pytest
import torch

from pra_hf.crossdoc_composition import (
    CrossDocumentCompositionConfig,
    CrossDocumentCompositionMode,
    CrossDocumentCompositionReceipt,
    GistAttentionMask,
    build_gist_attention_mask,
    contextualize_gists,
    memory_identity_digest,
)
from pra_hf.precision_qualification import (
    PrecisionMode,
    build_precision_metadata,
    infer_precision_mode,
)
from experiments.paper3_2_rag.run_prerope_causal_decomposition import (
    _composition_modes,
    _ordered_records,
    _precision_mode,
)


def test_parameter_free_gist_sa_shapes_determinism_and_immutability() -> None:
    keys = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    values = keys.flip(-1)
    keys_before = keys.clone()
    values_before = values.clone()
    mask = build_gist_attention_mask(3, GistAttentionMask.ALL_TO_ALL)

    first = contextualize_gists(keys, values, mask)
    second = contextualize_gists(keys, values, mask)

    assert first[0].shape == first[1].shape == (2, 3, 4)
    assert first[2].shape == (2, 3, 3)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert torch.equal(keys, keys_before)
    assert torch.equal(values, values_before)


def test_all_to_all_gist_sa_is_permutation_equivariant() -> None:
    generator = torch.Generator().manual_seed(17)
    keys = torch.randn((2, 4, 8), generator=generator)
    values = torch.randn((2, 4, 8), generator=generator)
    permutation = torch.tensor([2, 0, 3, 1])
    mask = build_gist_attention_mask(4, "all_to_all")
    expected_k, expected_v, _ = contextualize_gists(keys, values, mask)
    actual_k, actual_v, _ = contextualize_gists(
        keys[:, permutation], values[:, permutation], mask
    )
    assert torch.allclose(actual_k, expected_k[:, permutation], atol=1e-6)
    assert torch.allclose(actual_v, expected_v[:, permutation], atol=1e-6)


def test_gist_attention_masks_have_declared_geometry() -> None:
    causal = build_gist_attention_mask(3, "rank_causal")
    hub = build_gist_attention_mask(3, "top_ranked_hub")
    same = build_gist_attention_mask(
        3, "same_document_only", document_ids=("D1", "D2", "D1")
    )
    assert causal.int().sum().item() == 6
    assert hub.int().sum().item() == 7
    assert same.tolist() == [[True, False, True], [False, True, False], [True, False, True]]


def test_composition_contract_rejects_independent_mode_and_is_reproducible() -> None:
    with pytest.raises(ValueError, match="does not require"):
        CrossDocumentCompositionConfig(mode=CrossDocumentCompositionMode.INDEPENDENT_PRA)
    digest = memory_identity_digest(
        record_ids=("D1", "D2"), source_tokens=(8, 9), layer_count=2
    )
    receipt = CrossDocumentCompositionReceipt(
        mode="GIST_SA_APPEND",
        gist_count=2,
        gist_dim=16,
        gist_attention_mask="all_to_all",
        gist_attention_edges=4,
        boundary_tokens_per_record=0,
        corrected_token_count=0,
        request_composition_ms=1.0,
        request_composition_bytes=128,
        persistent_native_tokens=17,
        request_local_native_tokens=2,
        gist_positions=(17, 18),
        record_ids=("D1", "D2"),
        source_memory_digest=digest,
        pooling_method="layerwise_mean_pre_rope_kv",
        normalization_policy="scaled_dot_product_softmax",
        position_policy="query_adjacent_compact_band",
    )
    assert receipt.receipt_id == dataclasses.replace(receipt).receipt_id
    assert receipt.to_dict()["receipt_id"] == receipt.receipt_id


@pytest.mark.parametrize(
    ("model_id", "expected"),
    (("mlx-community/Qwen3-4B-4bit", PrecisionMode.INT4),
     ("mlx-community/Qwen3-4B-8bit", PrecisionMode.INT8),
     ("org/model-fp16", PrecisionMode.FP16),
     ("org/model-fp32", PrecisionMode.FP32)),
)
def test_precision_loader_routing(model_id: str, expected: PrecisionMode) -> None:
    assert infer_precision_mode(model_id) is expected


def test_precision_metadata_keeps_weight_and_kv_dtype_separate() -> None:
    metadata = build_precision_metadata(
        model_id="mlx-community/Qwen3-4B-8bit",
        model_revision="a" * 40,
        mode="INT8",
        kv_dtype="float16",
        group_size=64,
    )
    assert metadata.quantization_bits == 8
    assert metadata.weight_dtype == "int8_groupwise"
    assert metadata.kv_dtype == "float16"
    assert metadata.group_size == 64
    assert metadata.source_checkpoint == "NOT_REPORTED_BY_QUANTIZED_CHECKPOINT"
    assert metadata.source_weight_dtype == "checkpoint_declared_or_unknown"
    assert metadata.weight_conversion == "mlx_groupwise_quantized_checkpoint"


def test_float_precision_metadata_marks_runtime_cast() -> None:
    metadata = build_precision_metadata(
        model_id="Qwen/Qwen3-0.6B",
        model_revision="b" * 40,
        mode="FP32",
        kv_dtype="float32",
        source_weight_dtype="bfloat16",
    )
    assert metadata.weight_dtype == "float32"
    assert metadata.source_weight_dtype == "bfloat16"
    assert metadata.weight_conversion == "runtime_dtype_cast_from_checkpoint"


def test_runner_parses_composition_aliases_and_rejects_precision_mismatch() -> None:
    assert _composition_modes("append,boundary8,boundary32") == (
        CrossDocumentCompositionMode.GIST_SA_APPEND,
        CrossDocumentCompositionMode.GIST_SA_BOUNDARY_8,
        CrossDocumentCompositionMode.GIST_SA_BOUNDARY_32,
    )
    assert _precision_mode("auto", "org/model-8bit") is PrecisionMode.INT8
    with pytest.raises(ValueError, match="declares INT4"):
        _precision_mode("FP16", "org/model-4bit")


def test_record_order_is_explicit_and_seeded() -> None:
    records = ("a", "b", "c", "d")
    assert _ordered_records(records, "canonical", seed=11, example_id="q") == records
    assert _ordered_records(records, "reverse", seed=11, example_id="q") == records[::-1]
    first = _ordered_records(records, "random", seed=11, example_id="q")
    assert first == _ordered_records(records, "random", seed=11, example_id="q")
    assert sorted(first) == sorted(records)
