"""Model-normalized layer profiles for PRA routing and native-K/V storage.

The public runtime distinguishes four layer roles:

* address layers persist compact routing representations;
* detail layers persist materializable native K/V;
* routing layers score addresses for a request; and
* consumption layers read selected detail K/V during model execution.

Keeping these roles separate prevents a consumer-profile change from silently
expanding the persistent K/V footprint.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class DetailKVEncodingPolicy(str, Enum):
    """How reference ingestion chooses layers that retain full native K/V."""

    MINIMAL = "minimal"
    PROFILE_UNION = "profile_union"
    ALL_LAYERS = "all_layers"
    EXPLICIT = "explicit"


class AddressEncodingPolicy(str, Enum):
    """How reference ingestion chooses layers that retain routing addresses."""

    ROUTING_ONLY = "routing_only"
    ALL_CANDIDATE_ROUTING_LAYERS = "all_candidate_routing_layers"
    EXTERNAL_ONLY = "external_only"
    EXPLICIT = "explicit"


class MissingDetailKVPolicy(str, Enum):
    """Required behavior when a requested consumer lacks stored detail K/V."""

    REENCODE_MISSING = "reencode_missing"
    FAIL = "fail"
    DOWNGRADE_PROFILE = "downgrade_profile"


class NativeIndexLifecycleState(str, Enum):
    """Observable construction state for one native index component."""

    NOT_BUILT = "NOT_BUILT"
    PARTIAL = "PARTIAL"
    BUILT = "BUILT"
    SKIPPED = "SKIPPED"
    DEFERRED = "DEFERRED"


class MissingDetailKVError(RuntimeError):
    """A requested consumer profile needs detail K/V that is not resident."""

    def __init__(self, missing_layers: Sequence[int], policy: str) -> None:
        self.missing_layers = tuple(int(layer) for layer in missing_layers)
        self.policy = str(policy)
        super().__init__(
            f"Consumer profile requires missing detail K/V at layers "
            f"{self.missing_layers} (policy={self.policy})."
        )


LAYER_PROFILE_OBJECTIVES = ("reference_correctness", "quality_max", "balanced", "economy")


@dataclass(frozen=True)
class LayerSelection:
    """A topology-normalized selection over a decoder stack."""

    mode: str = "explicit"
    layers: tuple[int, ...] = ()
    n: int | None = None
    fraction: float | None = None

    @classmethod
    def from_value(cls, value: Any) -> "LayerSelection":
        """Parse an explicit list, profile string, mapping, or existing selection."""

        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(
                mode=str(value.get("mode", "explicit")),
                layers=tuple(int(item) for item in value.get("layers", ())),
                n=(None if value.get("n") is None else int(value["n"])),
                fraction=(
                    None if value.get("fraction") is None else float(value["fraction"])
                ),
            )
        if isinstance(value, str):
            name, separator, argument = value.partition(":")
            name = name.strip().lower()
            if name in {"all", "all_layers"}:
                return cls(mode="all_layers")
            if name in {"last_n", "evenly_spaced_n"}:
                if not separator:
                    raise ValueError(f"Layer profile {name!r} requires ':N'.")
                return cls(mode=name, n=int(argument))
            if name == "last_fraction":
                if not separator:
                    raise ValueError("last_fraction requires ':FRACTION'.")
                return cls(mode=name, fraction=float(argument))
            if name == "explicit":
                values = () if not argument else tuple(int(item) for item in argument.split(","))
                return cls(mode=name, layers=values)
            raise ValueError(f"Unknown layer profile mode: {name}")
        if isinstance(value, Iterable):
            return cls(mode="explicit", layers=tuple(int(item) for item in value))
        raise TypeError(f"Cannot parse layer selection from {type(value).__name__}.")

    def resolve(
        self,
        layer_count: int,
        *,
        allowed_layers: Sequence[int] | None = None,
    ) -> tuple[int, ...]:
        """Resolve the profile to sorted physical layer IDs."""

        if layer_count <= 0:
            raise ValueError("layer_count must be positive.")
        allowed = tuple(range(layer_count)) if allowed_layers is None else tuple(allowed_layers)
        if not allowed:
            raise ValueError("No model layers are eligible for PRA.")

        def normalize(layer: int) -> int:
            resolved = layer_count + layer if layer < 0 else layer
            if resolved < 0 or resolved >= layer_count:
                raise ValueError(f"Layer {layer} is outside a {layer_count}-layer model.")
            if resolved not in allowed:
                raise ValueError(
                    f"Layer {resolved} is not eligible for this model topology "
                    "(for example, a sliding attention layer)."
                )
            return resolved

        if self.mode == "all_layers":
            result = allowed
        elif self.mode == "last_n":
            count = int(self.n or 0)
            if count <= 0:
                raise ValueError("last_n requires a positive n.")
            result = allowed[-min(count, len(allowed)) :]
        elif self.mode == "last_fraction":
            fraction = float(self.fraction or 0.0)
            if not 0 < fraction <= 1:
                raise ValueError("last_fraction requires a fraction in (0, 1].")
            result = allowed[-max(1, math.ceil(len(allowed) * fraction)) :]
        elif self.mode == "evenly_spaced_n":
            count = int(self.n or 0)
            if count <= 0:
                raise ValueError("evenly_spaced_n requires a positive n.")
            if count >= len(allowed):
                result = allowed
            elif count == 1:
                result = (allowed[-1],)
            else:
                result = tuple(
                    allowed[round(index * (len(allowed) - 1) / (count - 1))]
                    for index in range(count)
                )
        elif self.mode == "explicit":
            result = tuple(normalize(layer) for layer in self.layers)
        else:
            raise ValueError(f"Unsupported layer profile mode: {self.mode}")
        return tuple(sorted(set(result)))


@dataclass(frozen=True)
class ResolvedLayerRoles:
    """Validated physical layers and their storage/execution provenance."""

    address_layers: tuple[int, ...]
    detail_kv_layers: tuple[int, ...]
    routing_layers: tuple[int, ...]
    consumption_layers: tuple[int, ...]
    detail_kv_encoding_policy: str
    address_encoding_policy: str
    missing_detail_kv_policy: str
    address_mode: str
    profile_name: str | None = None
    profile_source: str = "request"
    registry_version: str | None = None
    model_family: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    workload_class: str | None = None
    materialization_class: str | None = None
    objective: str = "balanced"

    def __post_init__(self) -> None:
        if not self.routing_layers:
            raise ValueError("routing_layers cannot be empty.")
        if not self.consumption_layers:
            raise ValueError("consumption_layers cannot be empty.")
        if not set(self.consumption_layers).issubset(self.detail_kv_layers):
            raise ValueError("Every consumption layer must have detail K/V.")
        if self.address_mode != "external" and not set(self.routing_layers).issubset(
            self.address_layers
        ):
            raise ValueError("Every native routing layer must have an address index.")

    @property
    def injected_layers(self) -> tuple[int, ...]:
        """Return layers requiring an attention adapter for any role."""

        return tuple(
            sorted(
                set(self.address_layers)
                | set(self.detail_kv_layers)
                | set(self.routing_layers)
                | set(self.consumption_layers)
            )
        )

    @property
    def primary_routing_layer(self) -> int:
        """Return the compatibility routing layer used by one-shot SDK calls."""

        return self.routing_layers[-1]

    def to_dict(self) -> dict[str, Any]:
        """Serialize profile provenance for traces and persisted artifacts."""

        values = asdict(self)
        values["injected_layers"] = list(self.injected_layers)
        return values


def eligible_layers(model_config_or_layer_count: Any) -> tuple[int, tuple[int, ...], str]:
    """Return decoder depth, PRA-eligible layers, and normalized model family."""

    if isinstance(model_config_or_layer_count, int):
        count = int(model_config_or_layer_count)
        return count, tuple(range(count)), "generic"
    config = model_config_or_layer_count
    count = int(config.num_hidden_layers)
    model_type = str(getattr(config, "model_type", "generic"))
    if model_type == "gemma3_text":
        layer_types = tuple(getattr(config, "layer_types", ()) or ())
        allowed = tuple(
            index for index, layer_type in enumerate(layer_types)
            if layer_type == "full_attention"
        )
        if not allowed:
            raise ValueError("Gemma 3 exposes no native full-attention layers.")
        return count, allowed, "gemma3"
    if "qwen" in model_type:
        family = "qwen"
    elif "llama" in model_type:
        family = "llama"
    else:
        family = model_type
    return count, tuple(range(count)), family


class LayerProfileRegistry:
    """Versioned fixed-profile registry with deterministic fallback precedence."""

    def __init__(self, payload: Mapping[str, Any], *, source: str = "memory") -> None:
        self.payload = dict(payload)
        self.source = source
        self.version = str(self.payload.get("version", "unversioned"))

    @classmethod
    def from_path(cls, path: str | Path) -> "LayerProfileRegistry":
        path = Path(path)
        return cls(json.loads(path.read_text(encoding="utf-8")), source=str(path))

    @classmethod
    def default(cls) -> "LayerProfileRegistry":
        path = Path(__file__).with_name("model_profiles") / "layer_profile_registry.json"
        return cls.from_path(path)

    def resolve(
        self,
        *,
        family: str,
        model_id: str | None,
        workload: str | None,
        materialization: str | None,
        objective: str,
    ) -> tuple[Mapping[str, Any], str]:
        """Resolve request -> workload/model -> family balanced -> correctness."""

        profiles = list(self.payload.get("profiles", ()))
        candidates = [
            (
                row.get("model_id") == model_id,
                row.get("workload") == workload,
                row.get("materialization") == materialization,
                row,
            )
            for row in profiles
            if row.get("family") == family and row.get("objective") == objective
        ]
        has_specific_request = any(
            value is not None for value in (model_id, workload, materialization)
        )
        for model_match, workload_match, materialization_match, row in candidates:
            if (
                has_specific_request
                and model_match
                and workload_match
                and materialization_match
            ):
                return row, "workload_model_profile"
        for row in profiles:
            if (
                row.get("family") == family
                and row.get("objective") == "balanced"
                and row.get("workload") is None
            ):
                return row, "family_balanced_fallback"
        for row in profiles:
            if row.get("objective") == "reference_correctness":
                return row, "reference_correctness_fallback"
        raise KeyError(f"No layer profile for family={family!r}, objective={objective!r}.")


def native_index_lifecycle(
    roles: ResolvedLayerRoles,
    *,
    built_address_layers: Iterable[int],
    built_detail_layers: Iterable[int],
) -> dict[str, str]:
    """Derive independently observable address and detail index states."""

    def state(requested: tuple[int, ...], built: set[int], *, external: bool = False) -> str:
        if external:
            return NativeIndexLifecycleState.SKIPPED.value
        if not requested:
            return NativeIndexLifecycleState.DEFERRED.value
        overlap = set(requested) & built
        if not overlap:
            return NativeIndexLifecycleState.NOT_BUILT.value
        if set(requested).issubset(built):
            return NativeIndexLifecycleState.BUILT.value
        return NativeIndexLifecycleState.PARTIAL.value

    return {
        "address_state": state(
            roles.address_layers,
            set(int(layer) for layer in built_address_layers),
            external=roles.address_mode == "external",
        ),
        "detail_kv_state": state(
            roles.detail_kv_layers,
            set(int(layer) for layer in built_detail_layers),
        ),
    }


def resolve_detail_availability(
    requested_layers: Iterable[int],
    available_layers: Iterable[int],
    *,
    policy: str,
) -> tuple[int, ...]:
    """Apply an explicit missing-detail policy without a silent profile change.

    ``reencode_missing`` raises a typed exception so the SDK can republish the
    source before generation. ``fail`` rejects immediately. Only an explicitly
    requested ``downgrade_profile`` returns the available subset.
    """

    requested = tuple(sorted({int(layer) for layer in requested_layers}))
    available = set(int(layer) for layer in available_layers)
    missing = tuple(layer for layer in requested if layer not in available)
    if not missing:
        return requested
    resolved_policy = MissingDetailKVPolicy(policy)
    if resolved_policy == MissingDetailKVPolicy.REENCODE_MISSING:
        raise MissingDetailKVError(missing, resolved_policy.value)
    if resolved_policy == MissingDetailKVPolicy.FAIL:
        raise ValueError(f"Missing detail K/V at consumer layers {missing}.")
    downgraded = tuple(layer for layer in requested if layer in available)
    if not downgraded:
        raise ValueError("Explicit profile downgrade would leave no consumer layers.")
    return downgraded


def common_calibration_candidates(layer_count: int) -> dict[str, LayerSelection]:
    """Return the preregistered contiguous and sparse profile search space."""

    counts = tuple(count for count in (1, 4, 8, 12, 14, 16, 20, 24) if count <= layer_count)
    result = {"all_layers": LayerSelection(mode="all_layers")}
    result.update({f"last_{count}": LayerSelection(mode="last_n", n=count) for count in counts})
    result.update(
        {
            f"even_{count}": LayerSelection(mode="evenly_spaced_n", n=count)
            for count in (4, 8)
            if count <= layer_count
        }
    )
    return result
