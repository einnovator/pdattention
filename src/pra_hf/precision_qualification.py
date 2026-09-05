"""Explicit precision provenance for Paper 3.2 transport qualification."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum


class PrecisionMode(str, Enum):
    FP32 = "FP32"
    FP16 = "FP16"
    INT8 = "INT8"
    INT4 = "INT4"


@dataclass(frozen=True)
class PrecisionMetadata:
    """Separate weight, activation, compute, and K/V precision provenance."""

    precision_mode: str
    quantization_method: str
    quantization_bits: int | None
    group_size: int | None
    symmetric: bool | None
    zero_point_policy: str
    weight_dtype: str
    activation_dtype: str
    kv_dtype: str
    compute_dtype: str
    source_checkpoint: str
    source_weight_dtype: str
    quantized_checkpoint: str | None
    weight_conversion: str
    engine: str
    kernel_backend: str
    provenance: str
    schema_version: str = "paper3.2-precision-metadata-v2"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def infer_precision_mode(model_id: str) -> PrecisionMode:
    """Route explicit MLX checkpoint suffixes without guessing unknown formats."""

    normalized = model_id.casefold()
    match = re.search(r"(?:^|[-_/])(4|8)bit(?:$|[-_/])", normalized)
    if match:
        return PrecisionMode.INT4 if match.group(1) == "4" else PrecisionMode.INT8
    if "fp32" in normalized or "float32" in normalized:
        return PrecisionMode.FP32
    if any(marker in normalized for marker in ("fp16", "float16", "bf16", "bfloat16")):
        return PrecisionMode.FP16
    raise ValueError(
        "precision cannot be inferred from checkpoint identity; pass an explicit mode"
    )


def build_precision_metadata(
    *,
    model_id: str,
    model_revision: str,
    mode: PrecisionMode | str,
    kv_dtype: str,
    activation_dtype: str | None = None,
    compute_dtype: str | None = None,
    source_checkpoint: str | None = None,
    source_weight_dtype: str | None = None,
    group_size: int | None = None,
    symmetric: bool | None = None,
) -> PrecisionMetadata:
    """Build conservative metadata; unknown quantizer details stay explicit."""

    mode = PrecisionMode(mode)
    bits = {PrecisionMode.INT4: 4, PrecisionMode.INT8: 8}.get(mode)
    quantized = bits is not None
    weight_dtype = {
        PrecisionMode.FP32: "float32",
        PrecisionMode.FP16: "float16_or_bfloat16",
        PrecisionMode.INT8: "int8_groupwise",
        PrecisionMode.INT4: "int4_groupwise",
    }[mode]
    resolved_source = source_checkpoint or (
        "NOT_REPORTED_BY_QUANTIZED_CHECKPOINT" if quantized else model_id
    )
    conversion = (
        "mlx_groupwise_quantized_checkpoint"
        if quantized
        else "runtime_dtype_cast_from_checkpoint"
    )
    return PrecisionMetadata(
        precision_mode=mode.value,
        quantization_method="mlx_groupwise" if quantized else "none",
        quantization_bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        zero_point_policy="checkpoint_declared_or_unknown" if quantized else "not_applicable",
        weight_dtype=weight_dtype,
        activation_dtype=activation_dtype or kv_dtype,
        kv_dtype=kv_dtype,
        compute_dtype=compute_dtype or kv_dtype,
        source_checkpoint=resolved_source,
        source_weight_dtype=(
            source_weight_dtype or "checkpoint_declared_or_unknown"
        ),
        quantized_checkpoint=model_id if quantized else None,
        weight_conversion=conversion,
        engine="mlx-lm",
        kernel_backend="MLX Metal",
        provenance=(
            f"checkpoint={model_id}@{model_revision}; source={resolved_source}; "
            f"conversion={conversion}; mode explicitly recorded; "
            "group and zero-point details are not inferred when unavailable"
        ),
    )
