"""Service layer for structural model onboarding and profile calibration."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from .product_config import pra_home
from .profile_benchmarks import ProfileBenchmarkRegistry


KNOWN_STRUCTURAL_MAPPINGS: dict[str, dict[str, Any]] = {
    "qwen2": {"family": "qwen", "attention": "self_attn", "position": "rope", "status": "VALIDATED"},
    "qwen3": {"family": "qwen", "attention": "self_attn", "position": "rope", "status": "VALIDATED"},
    "llama": {"family": "llama", "attention": "self_attn", "position": "rope", "status": "VALIDATED"},
    "gemma3_text": {"family": "gemma3", "attention": "self_attn", "position": "rope", "status": "PARTIAL_TOPOLOGY"},
}

VALIDATION_STAGES = (
    "V0_config_inspection",
    "V1_model_load",
    "V2_disabled_pra_parity",
    "V3_native_kv_capture",
    "V4_visible_prefix_native_equivalence",
    "V5_source_offset_equivalence",
    "V6_cached_decode_lifetime",
    "V7_gqa_mqa_layout",
    "V8_selected_region_consumption",
    "V9_generation_smoke",
)


@dataclass(frozen=True)
class StructuralAdapterSpec:
    """Versioned, training-free mapping from an HF decoder to PRA hooks."""

    base_model: Mapping[str, Any]
    architecture: Mapping[str, Any]
    mapping: Mapping[str, str]
    heads: Mapping[str, Any]
    position: Mapping[str, str]
    topology: Mapping[str, str]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_pretrained(self, directory: str | Path) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / "pra_adapter.yaml"
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return path


class ModelInspector:
    """Inspect HF metadata without loading model weights by default."""

    def inspect(self, model_id: str, *, revision: str | None = None) -> dict[str, Any]:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_id, revision=revision)
        model_type = str(getattr(config, "model_type", "unknown"))
        mapping = KNOWN_STRUCTURAL_MAPPINGS.get(model_type)
        layers = int(getattr(config, "num_hidden_layers", 0) or 0)
        hidden = int(getattr(config, "hidden_size", 0) or 0)
        vocab = int(getattr(config, "vocab_size", 0) or 0)
        intermediate = int(getattr(config, "intermediate_size", hidden * 4) or hidden * 4)
        approximate_parameters = vocab * hidden + layers * (
            4 * hidden * hidden + 3 * hidden * intermediate
        )
        return {
            "model": {
                "id": model_id,
                "revision": revision or getattr(config, "_commit_hash", None) or "unresolved",
                "architecture": (getattr(config, "architectures", None) or [model_type])[0],
                "family": mapping["family"] if mapping else model_type,
                "variant": self._variant(model_id),
                "parameter_count_approx": approximate_parameters or None,
            },
            "attention": {
                "layers": layers or None,
                "query_heads": getattr(config, "num_attention_heads", None),
                "kv_heads": getattr(config, "num_key_value_heads", None),
                "head_dim": getattr(config, "head_dim", None) or (
                    hidden // int(getattr(config, "num_attention_heads", 1) or 1) if hidden else None
                ),
                "topology": "heterogeneous" if getattr(config, "layer_types", None) else "homogeneous_global",
                "position_encoding": "rope" if hasattr(config, "rope_theta") or mapping else "unknown",
            },
            "pra": {
                "structural_adapter": {
                    "status": mapping["status"] if mapping else "UNSUPPORTED",
                    "source": f"builtin:{mapping['family']}" if mapping else None,
                },
                "native_kv": bool(mapping),
                "routing_capture": bool(mapping),
                "profile_status": "registry_lookup_required",
            },
        }

    @staticmethod
    def _variant(model_id: str) -> str:
        lowered = model_id.lower()
        for name in ("instruct", "reasoning", "code", "tool", "multimodal"):
            if name in lowered:
                return name
        return "base"


class StructuralAdapterBuilder:
    """Generate a declarative adapter for conventional supported decoders."""

    def __init__(self, inspector: ModelInspector | None = None) -> None:
        self.inspector = inspector or ModelInspector()

    def build(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        output: str | Path,
        force: bool = False,
    ) -> tuple[StructuralAdapterSpec, Path]:
        target = Path(output)
        if target.exists() and any(target.iterdir()) and not force:
            raise FileExistsError(f"Adapter output is not empty: {target}")
        inspected = self.inspector.inspect(model_id, revision=revision)
        status = inspected["pra"]["structural_adapter"]["status"]
        if status == "UNSUPPORTED":
            raise ValueError("No safe declarative mapping was detected; a reviewed Python plugin is required.")
        spec = StructuralAdapterSpec(
            base_model={"id": model_id, "revision": inspected["model"]["revision"]},
            architecture={
                "model_type": inspected["model"]["family"],
                "attention_class": inspected["attention"]["topology"],
            },
            mapping={
                "layers": "model.layers",
                "attention": "self_attn",
                "q_proj": "q_proj",
                "k_proj": "k_proj",
                "v_proj": "v_proj",
                "o_proj": "o_proj",
            },
            heads={
                "query": "config.num_attention_heads",
                "kv": "config.num_key_value_heads",
                "head_dim": "config.head_dim",
            },
            position={"type": inspected["attention"]["position_encoding"], "implementation": "native"},
            topology={"type": inspected["attention"]["topology"]},
        )
        return spec, spec.save_pretrained(target)


class ModelValidator:
    """Run and persist the bounded structural-adapter validation ladder."""

    def validate(
        self,
        model_id: str,
        *,
        adapter: str | Path | None = None,
        revision: str | None = None,
        suite: str = "smoke",
        output: str | Path | None = None,
        load_weights: bool = False,
        device: str = "auto",
    ) -> dict[str, Any]:
        inspected = ModelInspector().inspect(model_id, revision=revision)
        supported = inspected["pra"]["structural_adapter"]["status"] != "UNSUPPORTED"
        if not supported:
            stages = [
                {"stage": name, "status": "FAIL" if index == 0 else "BLOCKED"}
                for index, name in enumerate(VALIDATION_STAGES)
            ]
        elif load_weights:
            stages = self._validate_loaded(model_id, revision=revision, device=device)
        else:
            stages = [
                {"stage": name, "status": "PASS" if index == 0 else "DEFERRED_WEIGHT_LOAD"}
                for index, name in enumerate(VALIDATION_STAGES)
            ]
        failures = any(row["status"] == "FAIL" for row in stages)
        unresolved = any(row["status"] not in {"PASS", "FAIL"} for row in stages)
        if not supported:
            overall = "UNSUPPORTED"
        elif failures:
            overall = "FAILED"
        elif unresolved:
            overall = "PARTIAL_TOPOLOGY"
        else:
            overall = "VALIDATED"
        result = {
            "schema_version": 1,
            "model": model_id,
            "revision": inspected["model"]["revision"],
            "adapter": str(adapter) if adapter else inspected["pra"]["structural_adapter"]["source"],
            "suite": suite,
            "status": overall,
            "stages": stages,
            "note": (
                "V4/V5 require published equivalence fixtures and are never inferred from projection discovery."
                if load_weights
                else "Weight-bearing semantic stages are deferred until --validate is requested."
            ),
        }
        if output is not None:
            path = Path(output)
            path.mkdir(parents=True, exist_ok=True)
            (path / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    @staticmethod
    def _validate_loaded(model_id: str, *, revision: str | None, device: str) -> list[dict[str, Any]]:
        """Execute bounded semantic checks and preserve unsupported fixture gates."""

        from .model import PRAForCausalLM

        rows: dict[str, dict[str, Any]] = {
            name: {"stage": name, "status": "BLOCKED"} for name in VALIDATION_STAGES
        }
        rows[VALIDATION_STAGES[0]] = {"stage": VALIDATION_STAGES[0], "status": "PASS"}
        try:
            kwargs = {"revision": revision} if revision else {}
            pra = PRAForCausalLM.from_pretrained(model_id, **kwargs)
            resolved_device = (
                "cuda" if device == "auto" and torch.cuda.is_available()
                else "cpu" if device == "auto"
                else device
            )
            pra.model.to(resolved_device)
            rows[VALIDATION_STAGES[1]] = {
                "stage": VALIDATION_STAGES[1], "status": "PASS", "device": str(pra.device)
            }

            prompt = "PRA structural validation"
            generation = {"max_new_tokens": 1, "do_sample": False}
            pra.disable()
            disabled = pra.generate(prompt, **generation)
            pra.enable()
            enabled = pra.generate(prompt, **generation)
            rows[VALIDATION_STAGES[2]] = {
                "stage": VALIDATION_STAGES[2],
                "status": "PASS" if disabled == enabled else "FAIL",
                "detail": "No-reference enabled/disabled generation parity.",
            }

            handle = pra.add_reference(
                "pra://validation/reference",
                text="Progressive Retrieval Attention validation evidence. " * 8,
            )
            rows[VALIDATION_STAGES[3]] = {
                "stage": VALIDATION_STAGES[3],
                "status": "PASS" if handle.tokens > 0 and handle.chunks > 0 else "FAIL",
                "tokens": handle.tokens,
                "chunks": handle.chunks,
            }
            rows[VALIDATION_STAGES[4]] = {
                "stage": VALIDATION_STAGES[4],
                "status": "FIXTURE_REQUIRED",
                "detail": "Requires a visible-prefix/native-KV equivalence fixture.",
            }
            rows[VALIDATION_STAGES[5]] = {
                "stage": VALIDATION_STAGES[5],
                "status": "FIXTURE_REQUIRED",
                "detail": "Requires a source-offset/RoPE equivalence fixture.",
            }

            first = pra.generate("What mechanism is being validated?", **generation)
            second = pra.generate("What mechanism is being validated?", **generation)
            rows[VALIDATION_STAGES[6]] = {
                "stage": VALIDATION_STAGES[6],
                "status": "PASS" if isinstance(first, str) and isinstance(second, str) else "FAIL",
                "detail": "Two bounded requests reused the reference cache without lifecycle failure.",
            }
            query_heads = int(getattr(pra.model.config, "num_attention_heads", 0) or 0)
            kv_heads = int(getattr(pra.model.config, "num_key_value_heads", query_heads) or query_heads)
            layout_ok = query_heads > 0 and kv_heads > 0 and query_heads % kv_heads == 0
            rows[VALIDATION_STAGES[7]] = {
                "stage": VALIDATION_STAGES[7],
                "status": "PASS" if layout_ok else "FAIL",
                "query_heads": query_heads,
                "kv_heads": kv_heads,
            }
            routed = pra.route("Which reference discusses Progressive Retrieval Attention?")
            rows[VALIDATION_STAGES[8]] = {
                "stage": VALIDATION_STAGES[8],
                "status": "PASS" if routed.selected else "FAIL",
                "selected_regions": len(routed.selected),
            }
            generated = pra.generate("Name the attention mechanism.", **generation)
            rows[VALIDATION_STAGES[9]] = {
                "stage": VALIDATION_STAGES[9],
                "status": "PASS" if isinstance(generated, str) else "FAIL",
            }
        except Exception as error:
            blocked = next(
                (name for name in VALIDATION_STAGES if rows[name]["status"] == "BLOCKED"),
                VALIDATION_STAGES[-1],
            )
            rows[blocked] = {
                "stage": blocked,
                "status": "FAIL",
                "error": f"{type(error).__name__}: {error}",
            }
        finally:
            if "pra" in locals():
                del pra
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return [rows[name] for name in VALIDATION_STAGES]


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    detail: str


class DoctorService:
    """Inspect core and optional PRA runtime dependencies without importing them eagerly."""

    OPTIONAL = {"transformers": "transformers", "vLLM": "vllm", "SGLang": "sglang", "MLX": "mlx"}

    def run(self, *, verbose: bool = False) -> dict[str, Any]:
        checks = [
            DoctorCheck("Python", "AVAILABLE", platform.python_version()),
            DoctorCheck("PRA", "AVAILABLE", self._version("pra-hf")),
            DoctorCheck("Torch", "AVAILABLE", torch.__version__),
            DoctorCheck("CUDA", "AVAILABLE" if torch.cuda.is_available() else "UNSUPPORTED", str(torch.cuda.is_available())),
            DoctorCheck("MPS", "AVAILABLE" if torch.backends.mps.is_available() else "UNSUPPORTED", str(torch.backends.mps.is_available())),
        ]
        for label, package in self.OPTIONAL.items():
            if package == "transformers":
                continue
            version = self._version(package)
            checks.append(DoctorCheck(label, "AVAILABLE" if version != "not installed" else "NOT_INSTALLED", version))
        result: dict[str, Any] = {
            "checks": [asdict(check) for check in checks],
            "paths": {
                "home": str(pra_home()),
                "cache": str(pra_home() / "cache"),
                "sessions": str(pra_home() / "sessions"),
            },
            "known_engines": ["hf", "openai", "vllm", "sglang", "mlx"],
        }
        if verbose:
            result["executable"] = sys.executable
            result["platform"] = platform.platform()
            result["available_memory"] = self._available_memory()
        return result

    @staticmethod
    def _version(package: str) -> str:
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return "not installed"

    @staticmethod
    def _available_memory() -> int | None:
        try:
            import psutil

            return int(psutil.virtual_memory().available)
        except ImportError:
            return None


class OnboardingPipeline:
    """Compose inspection, adapter generation, validation, and runtime artifacts."""

    def run(
        self,
        model_id: str,
        *,
        output: str | Path,
        revision: str | None = None,
        suite: str = "standard",
        force: bool = False,
    ) -> dict[str, Any]:
        root = Path(output)
        root.mkdir(parents=True, exist_ok=True)
        inspected = ModelInspector().inspect(model_id, revision=revision)
        _, adapter_path = StructuralAdapterBuilder().build(
            model_id, revision=revision, output=root / "structural_adapter", force=force
        )
        validation = ModelValidator().validate(
            model_id, adapter=adapter_path.parent, revision=revision, suite=suite, output=root
        )
        runtime = {
            "schema_version": 1,
            "model": {"id": model_id, "revision": inspected["model"]["revision"]},
            "structural_adapter": {"source": str(adapter_path.parent)},
            "learned_adapters": {"routing": None, "memory": None},
            "default_profile": "REFERENCE_CORRECTNESS",
            "profiles": {"REFERENCE_CORRECTNESS": {"status": "VALIDATION_REQUIRED"}},
        }
        (root / "pra.yaml").write_text(yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "model": inspected["model"],
            "suite": suite,
            "validation": validation["status"],
            "timestamp": time.time(),
            "environment": {"python": platform.python_version(), "torch": torch.__version__},
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (root / "report.md").write_text(
            f"# PRA onboarding: {model_id}\n\nStructural status: **{validation['status']}**.\n\n"
            "Semantic weight-bearing validation remains explicit in `validation.json`.\n",
            encoding="utf-8",
        )
        return {"output": str(root), "inspection": inspected, "validation": validation, "runtime": runtime}


class ProfileCalibrator:
    """Package measured registry evidence into an evidence-aware runtime run."""

    def calibrate(
        self,
        model_id: str,
        *,
        output: str | Path,
        workload: str | None = None,
        suite: str = "standard",
        engine: str = "hf",
        registry: str | Path | None = None,
    ) -> dict[str, Any]:
        root = Path(output)
        root.mkdir(parents=True, exist_ok=True)
        source = ProfileBenchmarkRegistry.from_path(registry) if registry else ProfileBenchmarkRegistry.default()
        matching = list(source.find(model_id, workload=workload))
        profiles: dict[str, Any] = {}
        for row in matching:
            profiles[str(row["profile"])] = {
                "evidence_tier": row["evidence_tier"],
                "status": row["profile_status"],
                "engine": row["engine"],
                "workload": row["workload"],
                "quality": row["quality_absolute"],
                "materialized_tokens": row["materialized_tokens"],
                "address_layers": row["address_layers"],
                "detail_kv_layers": row["detail_kv_layers"],
            }
        if not profiles:
            profiles = {
                "REFERENCE_CORRECTNESS": {
                    "evidence_tier": "SMOKE",
                    "status": "CALIBRATION_PENDING",
                    "engine": engine,
                    "workload": workload or "standard",
                },
                "QUALITY_MAX_CANDIDATE": {
                    "evidence_tier": "SMOKE",
                    "status": "CALIBRATION_PENDING",
                    "engine": engine,
                    "workload": workload or "standard",
                },
            }
        validated = [
            name for name, value in profiles.items()
            if value.get("status") == "MEASURED" and value.get("evidence_tier") != "SMOKE"
        ]
        default_profile = "BALANCED" if "BALANCED" in validated else "REFERENCE_CORRECTNESS"
        runtime = {
            "schema_version": 1,
            "model": {"id": model_id, "revision": "resolved_at_runtime"},
            "structural_adapter": {"source": "auto"},
            "learned_adapters": {"routing": None, "memory": None},
            "default_profile": default_profile,
            "profiles": profiles,
        }
        benchmark_payload = {
            "schema_version": source.schema_version,
            "registry_version": source.registry_version,
            "benchmarks": matching,
        }
        metrics = {
            "model": model_id,
            "suite": suite,
            "workload": workload,
            "engine": engine,
            "measured_rows": len(matching),
            "training_recommended": False,
            "reason": "oracle_headroom_not_measured" if not matching else "calibration_reuses_frozen_evidence",
        }
        (root / "pra.yaml").write_text(yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8")
        (root / "profiles.yaml").write_text(yaml.safe_dump(profiles, sort_keys=False), encoding="utf-8")
        (root / "benchmarks.json").write_text(json.dumps(benchmark_payload, indent=2), encoding="utf-8")
        (root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (root / "environment.json").write_text(
            json.dumps({"python": platform.python_version(), "torch": torch.__version__, "engine": engine}, indent=2),
            encoding="utf-8",
        )
        (root / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "model": model_id, "suite": suite, "created_at": time.time()}, indent=2),
            encoding="utf-8",
        )
        (root / "report.md").write_text(
            f"# PRA profile calibration: {model_id}\n\n"
            f"Suite: `{suite}`. Registry rows: {len(matching)}. Default: `{default_profile}`.\n",
            encoding="utf-8",
        )
        return {"output": str(root), "default_profile": default_profile, "profiles": profiles, "metrics": metrics}
