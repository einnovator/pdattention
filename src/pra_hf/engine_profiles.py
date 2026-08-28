"""Versioned inference-engine defaults for gateway capability resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class EngineType(str, Enum):
    """Stable physical engine family independent of PRA integration depth."""

    OPENAI_GENERIC = "openai_generic"
    VLLM = "vllm"
    SGLANG = "sglang"
    FREETOKEN = "freetoken"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    LLAMA_CPP = "llama_cpp"
    MLX = "mlx"
    CUSTOM = "custom"


class PrefixCacheMode(str, Enum):
    """Ordinary sequential-prefix cache/session behavior."""

    UNKNOWN = "unknown"
    STATELESS = "stateless"
    AUTOMATIC_PREFIX_CACHE = "automatic_prefix_cache"
    EXPLICIT_PREFIX_HANDLE = "explicit_prefix_handle"
    SESSION_STATE = "session_state"


@dataclass(frozen=True)
class EngineProfile:
    """Conservative defaults overridden by explicit config or probing."""

    engine_type: EngineType
    default_pra_level: str
    default_prefix_cache_mode: PrefixCacheMode
    streaming: bool
    incremental_messages: bool
    explicit_session: bool
    resource_delta: bool
    cache_affinity: bool

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "EngineProfile":
        return cls(
            EngineType(values["engine_type"]),
            str(values["default_pra_level"]),
            PrefixCacheMode(values["default_prefix_cache_mode"]),
            bool(values["streaming"]),
            bool(values["incremental_messages"]),
            bool(values["explicit_session"]),
            bool(values["resource_delta"]),
            bool(values["cache_affinity"]),
        )


class EngineProfileRegistry:
    """Validated packaged mapping from engine type to conservative defaults."""

    def __init__(self, payload: Mapping[str, Any], *, source: str = "memory") -> None:
        self.schema_version = str(payload.get("schema_version", ""))
        self.registry_version = str(payload.get("registry_version", ""))
        self.source = source
        if self.schema_version != "1.0" or not self.registry_version:
            raise ValueError("Unsupported or unversioned engine profile registry.")
        profiles = [EngineProfile.from_dict(row) for row in payload.get("profiles", ())]
        self.profiles = {profile.engine_type: profile for profile in profiles}
        if set(self.profiles) != set(EngineType):
            missing = set(EngineType) - set(self.profiles)
            raise ValueError(f"Engine profile registry is incomplete: {missing}")

    @classmethod
    def from_path(cls, path: str | Path) -> "EngineProfileRegistry":
        path = Path(path)
        return cls(json.loads(path.read_text(encoding="utf-8")), source=str(path))

    @classmethod
    def default(cls) -> "EngineProfileRegistry":
        return cls.from_path(
            Path(__file__).with_name("model_profiles") / "engine_profile_registry.json"
        )

    def resolve(self, engine_type: EngineType | str) -> EngineProfile:
        return self.profiles[EngineType(engine_type)]
