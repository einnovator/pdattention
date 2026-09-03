"""Precision identities and deterministic qualification memory gates.

Weight precision is part of a PRA qualification identity.  The broad family
(``INT4``) and concrete encoding (``MLX-4bit``) stay separate because two
four-bit formats are not interchangeable evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


PRECISION_FAMILIES = frozenset(
    {
        "FP32",
        "FP16",
        "BF16",
        "INT8",
        "INT6",
        "INT4",
        "MXFP4",
        "OTHER",
        "UNSPECIFIED",
    }
)


def _clean(value: object) -> str:
    return str(value or "").strip()


def _runtime_label(value: str) -> str:
    lowered = value.lower()
    if "mlx" in lowered:
        return "MLX"
    if "bitsandbytes" in lowered or "bnb" in lowered:
        return "bitsandbytes"
    if "gguf" in lowered or "llama.cpp" in lowered:
        return "GGUF"
    if "gptq" in lowered:
        return "GPTQ"
    if "awq" in lowered:
        return "AWQ"
    if "torch" in lowered or lowered in {"hf", "huggingface", "huggingface_eager"}:
        return "PyTorch"
    return value or "native"


@dataclass(frozen=True)
class PrecisionDescriptor:
    """Exact weight and adaptor precision dimensions for one measured run."""

    precision_family: str
    precision_encoding: str
    serving_precision: str | None = None
    feature_extraction_precision: str | None = None
    adaptor_parameter_precision: str | None = None

    def __post_init__(self) -> None:
        family = _clean(self.precision_family).upper()
        encoding = _clean(self.precision_encoding)
        if family not in PRECISION_FAMILIES:
            raise ValueError(f"Unknown precision family: {self.precision_family!r}")
        if not encoding:
            raise ValueError("precision_encoding is required")
        object.__setattr__(self, "precision_family", family)
        object.__setattr__(self, "precision_encoding", encoding)
        if self.serving_precision is None:
            object.__setattr__(self, "serving_precision", family)

    @property
    def is_explicit(self) -> bool:
        return self.precision_family != "UNSPECIFIED"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_precision(
    quantization: object,
    *,
    engine: str | None = None,
    precision_family: str | None = None,
    precision_encoding: str | None = None,
    feature_extraction_precision: str | None = None,
    adaptor_parameter_precision: str | None = None,
) -> PrecisionDescriptor:
    """Normalize model metadata without treating all equal-bit formats alike."""

    engine_name = _clean(engine)
    runtime = engine_name
    scheme = ""
    name = ""
    bits: int | None = None
    if isinstance(quantization, Mapping):
        runtime = _clean(quantization.get("runtime")) or engine_name
        scheme = _clean(quantization.get("scheme"))
        name = _clean(quantization.get("name"))
        raw_bits = quantization.get("bits")
        if raw_bits is not None:
            bits = int(raw_bits)
    else:
        name = _clean(quantization)

    lowered = " ".join((runtime, scheme, name)).lower().replace("_", "-")
    if precision_family:
        family = precision_family.upper()
    elif "bfloat16" in lowered or "bf16" in lowered:
        family = "BF16"
    elif "float32" in lowered or "fp32" in lowered:
        family = "FP32"
    elif "float16" in lowered or "fp16" in lowered or "half" in lowered:
        family = "FP16"
    elif "mxfp4" in lowered:
        family = "MXFP4"
    elif bits in {4, 6, 8}:
        family = f"INT{bits}"
    elif any(marker in lowered for marker in ("4bit", "4-bit", "q4_")):
        family = "INT4"
    elif any(marker in lowered for marker in ("6bit", "6-bit")):
        family = "INT6"
    elif any(marker in lowered for marker in ("8bit", "8-bit", "int8")):
        family = "INT8"
    else:
        family = "UNSPECIFIED"

    if precision_encoding:
        encoding = precision_encoding
    elif bits is not None:
        runtime_label = _runtime_label(runtime or engine_name)
        scheme_label = f"-{scheme}" if scheme else ""
        encoding = f"{runtime_label}-{bits}bit{scheme_label}"
    elif name:
        runtime_label = _runtime_label(runtime or engine_name)
        normalized_name = name.removeprefix("torch.") if runtime_label == "PyTorch" else name
        encoding = f"{runtime_label}-{normalized_name}" if runtime_label else normalized_name
    elif family != "UNSPECIFIED":
        encoding = f"{engine_name or 'native'}-{family.lower()}"
    else:
        encoding = "UNSPECIFIED"

    return PrecisionDescriptor(
        precision_family=family,
        precision_encoding=encoding,
        serving_precision=family,
        feature_extraction_precision=feature_extraction_precision,
        adaptor_parameter_precision=adaptor_parameter_precision,
    )


class MemoryGateStatus(str, Enum):
    QUALIFIABLE = "QUALIFIABLE"
    CONTEXT_LIMITED = "CONTEXT_LIMITED"
    LOAD_ONLY = "LOAD_ONLY"
    BLOCKED_MEMORY = "BLOCKED_MEMORY"


@dataclass(frozen=True)
class MemoryGateObservation:
    """Measured outcome for one increasingly demanding model-load stage."""

    stage: str
    succeeded: bool
    peak_device_memory_bytes: float | None = None
    peak_host_memory_bytes: float | None = None
    peak_unified_memory_bytes: float | None = None
    kv_bytes: float | None = None
    temporary_allocation_bytes: float | None = None
    load_seconds: float | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if not _clean(self.stage):
            raise ValueError("memory-gate stage is required")
        for name in (
            "peak_device_memory_bytes",
            "peak_host_memory_bytes",
            "peak_unified_memory_bytes",
            "kv_bytes",
            "temporary_allocation_bytes",
            "load_seconds",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_memory_gate(
    observations: Sequence[MemoryGateObservation],
) -> MemoryGateStatus:
    """Classify a load -> context -> workload preflight without guessing limits."""

    if not observations:
        return MemoryGateStatus.LOAD_ONLY
    ordered = tuple(observations)
    load = next((row for row in ordered if row.stage == "load_only"), ordered[0])
    if not load.succeeded:
        return MemoryGateStatus.BLOCKED_MEMORY
    failures = [row for row in ordered if not row.succeeded]
    if failures:
        return MemoryGateStatus.CONTEXT_LIMITED
    successful_stages = {row.stage for row in ordered if row.succeeded}
    if "target_rag_context" in successful_stages or "multi_query_warm" in successful_stages:
        return MemoryGateStatus.QUALIFIABLE
    return MemoryGateStatus.LOAD_ONLY
