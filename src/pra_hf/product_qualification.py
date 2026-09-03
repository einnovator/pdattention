"""Product qualification records shared by the PRA CLI, reports, and docs.

This module deliberately keeps product names separate from the historical
research codes.  It also treats missing measurements as data: an unavailable
number is ``None`` and cannot satisfy a recommendation gate.
"""

from __future__ import annotations

import html
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from .canonical_evidence import (
    CanonicalEvidenceRecord,
    EvidenceCondition,
    MeasurementState,
    render_markdown_table as render_canonical_markdown_table,
)
from .execution_modes import (
    ExecutionMode,
    ExecutionModeResolver,
    ModeEvidence,
    ModeStatus,
    normalize_status,
)
from .product_config import pra_home


SCHEMA_VERSION = "1.0"
NOT_MEASURED = "NOT_MEASURED"
PUBLIC_STATUSES = {
    "Measured", "Validated", "Candidate", "Qualification pending", "Blocked",
    "Not measured", "Not applicable", "Research-only", "Recommended", NOT_MEASURED,
}
PUBLIC_MODES = (
    "full_context",
    "selected_context",
    "native_memory_hot",
    "native_memory_warm",
    "native_serving",
)
MODE_LABELS = {
    "full_context": "Full Context",
    "selected_context": "Selected Context",
    "native_memory_hot": "Native Memory HOT",
    "native_memory_warm": "Native Memory WARM",
    "native_serving": "Native Serving",
}
ENGINE_ALIASES = {
    "hf": "hugging-face",
    "huggingface": "hugging-face",
    "hugging_face": "hugging-face",
    "llama_cpp": "llama-cpp",
    "tensorrt_llm": "tensorrt-llm",
    "free-token": "freetoken",
}
ENGINE_PACKAGES = {
    "hugging-face": "transformers",
    "mlx": "mlx",
    "vllm": "vllm",
    "sglang": "sglang",
    "openvino": "openvino",
    "tensorrt-llm": "tensorrt_llm",
    "airllm": "airllm",
    "ollama": None,
    "llama-cpp": "llama_cpp",
    "freetoken": "freetoken",
}
ENGINE_EXECUTABLES = {"ollama": "ollama"}
STATUS_ALIASES = {
    "Not qualified": "Qualification pending",
    "Research only": "Research-only",
    "MEASURED": "Measured",
    "VALIDATED": "Validated",
    "CANDIDATE": "Candidate",
    "CALIBRATION_PENDING": "Qualification pending",
    "BLOCKED": "Blocked",
    "NOT_MEASURED": "Not measured",
    "NOT_APPLICABLE": "Not applicable",
    "RESEARCH_ONLY": "Research-only",
    "RECOMMENDED": "Recommended",
}

QUALITY_FIELDS = ("success", "em", "f1", "evidence_recall", "gold_log_probability")
CONTEXT_FIELDS = ("full_input_tokens", "selected_input_tokens", "visible_input_tokens", "reduction")
PERFORMANCE_FIELDS = (
    "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
    "itl_p50_ms", "itl_p95_ms", "itl_p99_ms",
    "completion_p50_ms", "completion_p95_ms", "completion_p99_ms",
    "requests_per_second", "successful_requests_per_second",
)
MEMORY_FIELDS = ("peak_bytes", "local_kv_bytes", "native_memory_bytes")
LIFECYCLE_FIELDS = (
    "reloads", "evictions", "reuse", "reference_encoding_ms", "transfer_bytes",
    "queue_delay_ms", "prefetch_ready_before_demand", "shared_residency",
)


def _registry_path() -> Path:
    return Path(__file__).with_name("model_profiles") / "engine_documentation_registry.json"


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a JSON object: {path}")
    return dict(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _version(package: str | None) -> str | None:
    if package is None:
        return None
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _status(value: Any) -> str:
    return STATUS_ALIASES.get(str(value), str(value))


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class EngineProductRegistry:
    """Versioned engine capability and recommendation registry."""

    def __init__(self, payload: Mapping[str, Any], *, source: str) -> None:
        self.payload = dict(payload)
        self.source = source
        self.engines = tuple(dict(row) for row in self.payload.get("engines", ()))
        if not self.engines:
            raise ValueError("Engine registry contains no engines.")
        slugs = [str(row.get("slug", "")) for row in self.engines]
        if any(not slug for slug in slugs) or len(slugs) != len(set(slugs)):
            raise ValueError("Engine registry slugs must be non-empty and unique.")

    @classmethod
    def default(cls) -> "EngineProductRegistry":
        path = _registry_path()
        return cls(_read_json(path), source=str(path))

    def resolve(self, engine: str) -> dict[str, Any]:
        slug = ENGINE_ALIASES.get(engine.lower(), engine.lower())
        for row in self.engines:
            if row["slug"] == slug or str(row["name"]).lower() == engine.lower():
                return deepcopy(row)
        known = ", ".join(row["slug"] for row in self.engines)
        raise KeyError(f"Unknown engine '{engine}'. Choose one of: {known}")

    def matrix(self) -> dict[str, Any]:
        return {
            "schema_version": self.payload.get("schema_version"),
            "registry_version": self.payload.get("evidence_as_of"),
            "provenance": self.source,
            "engines": [self.summary(row) for row in self.engines],
        }

    def details(self, engine: str) -> dict[str, Any]:
        row = self.resolve(engine)
        row["capabilities"] = {
            key: _status(value) for key, value in row["capabilities"].items()
        }
        row["registry_version"] = self.payload.get("evidence_as_of")
        row["provenance"] = self.source
        row["mode_resolution"] = ExecutionModeResolver().resolve("auto", row).to_dict()
        return row

    def summary(self, row: Mapping[str, Any]) -> dict[str, Any]:
        capabilities = row["capabilities"]
        resolution = ExecutionModeResolver().resolve("auto", row)
        return {
            "engine": row["slug"],
            "name": row["name"],
            "selected_context": _status(capabilities["selected_context"]),
            "typed_transport": _status(capabilities["typed_transport"]),
            "native_memory": _status(capabilities["native_memory"]),
            "native_serving": _status(capabilities["native_serving"]),
            "recommended": row["recommended_today"],
            "resolved_mode": resolution.resolved_mode.value,
            "resolution_reason": resolution.reason,
            "mode_status": [candidate.to_dict() for candidate in resolution.candidates],
            "evidence": row["evidence"],
            "provenance": self.source,
        }


def environment_report(registry: EngineProductRegistry | None = None) -> dict[str, Any]:
    """Return a product-oriented environment report without loading engines."""

    registry = registry or EngineProductRegistry.default()
    disk = shutil.disk_usage(Path.cwd())
    memory: int | None = None
    try:
        import psutil

        memory = int(psutil.virtual_memory().total)
    except ImportError:
        pass
    accelerator = "CPU"
    if torch.cuda.is_available():
        accelerator = f"CUDA: {torch.cuda.get_device_name(0)}"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        accelerator = "MPS (Apple Metal)"
    engines = []
    for row in registry.engines:
        slug = row["slug"]
        package = ENGINE_PACKAGES.get(slug)
        executable = ENGINE_EXECUTABLES.get(slug)
        installed = bool(shutil.which(executable)) if executable else (
            package is not None and importlib.util.find_spec(package) is not None
        )
        version = _version(package)
        engines.append({
            **registry.summary(row),
            "installed": installed,
            "version": version,
            "location": "external executable or remote endpoint" if package is None else "local Python",
            "connection": "local or remote" if package is None else "local",
            "reachable": None,
        })
    home = pra_home()
    local_models = sorted(
        str(path.parent) for path in home.glob("bundles/**/bundle.yaml")
    ) if home.exists() else []
    next_engine = next((row for row in engines if row["installed"]), None)
    return {
        "schema_version": SCHEMA_VERSION,
        "system": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "executable": sys.executable,
            "cpu": platform.processor() or platform.machine(),
            "accelerator": accelerator,
            "memory_bytes": memory,
            "disk_free_bytes": disk.free,
            "disk_total_bytes": disk.total,
        },
        "engines": engines,
        "models_and_adapters": {
            "known_local_bundles": local_models,
            "profile_registry": str(Path(__file__).with_name("model_profiles") / "pra_profile_benchmarks.json"),
        },
        "problems": [
            "No accelerator is available in this Python environment."
        ] if accelerator == "CPU" else [],
        "next_action": (
            f"pra evaluate MODEL --engine {next_engine['engine']} --dataset DATASET"
            if next_engine else "Install or configure an execution engine, then run pra doctor."
        ),
    }


def _empty_section(fields: Sequence[str]) -> dict[str, Any]:
    return {name: None for name in fields}


def empty_measurement(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "status": NOT_MEASURED,
        "quality": _empty_section(QUALITY_FIELDS),
        "context": _empty_section(CONTEXT_FIELDS),
        "performance": _empty_section(PERFORMANCE_FIELDS),
        "memory": _empty_section(MEMORY_FIELDS),
        "lifecycle": _empty_section(LIFECYCLE_FIELDS),
        "provenance": None,
    }


def _normalize_measurement(name: str, supplied: Mapping[str, Any]) -> dict[str, Any]:
    result = empty_measurement(MODE_LABELS[name])
    for section, fields in (
        ("quality", QUALITY_FIELDS), ("context", CONTEXT_FIELDS),
        ("performance", PERFORMANCE_FIELDS), ("memory", MEMORY_FIELDS),
        ("lifecycle", LIFECYCLE_FIELDS),
    ):
        values = supplied.get(section, {})
        if values is not None and not isinstance(values, Mapping):
            raise ValueError(f"Measurement section '{name}.{section}' must be an object.")
        for field in fields:
            if field in supplied:
                result[section][field] = supplied[field]
            elif field in (values or {}):
                result[section][field] = values[field]
    result["status"] = _status(supplied.get("status", "Measured"))
    if result["status"] not in PUBLIC_STATUSES:
        raise ValueError(f"Unknown product measurement status: {result['status']}")
    result["provenance"] = supplied.get("provenance")
    return result


def _ratio_gain(baseline: Any, candidate: Any) -> float | None:
    if baseline is None or candidate is None or float(candidate) == 0:
        return None
    return float(baseline) / float(candidate)


def _fraction_reduction(baseline: Any, candidate: Any) -> float | None:
    if baseline is None or candidate is None or float(baseline) == 0:
        return None
    return 1.0 - float(candidate) / float(baseline)


def derive_attribution(modes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Compute only adjacent gains so retrieval and transport remain distinct."""

    full = modes["full_context"]
    selected = modes["selected_context"]
    native = modes.get("native_memory_hot")
    serving = modes.get("native_serving")
    context_gain = {
        "comparison": "Full Context -> Selected Context",
        "visible_token_reduction": _fraction_reduction(
            full["context"]["visible_input_tokens"], selected["context"]["visible_input_tokens"]
        ),
        "ttft_speedup": _ratio_gain(
            full["performance"]["ttft_p95_ms"], selected["performance"]["ttft_p95_ms"]
        ),
    }
    native_gain = {
        "comparison": "Selected Context -> Native Memory",
        "ttft_speedup": None,
        "throughput_gain": None,
    }
    if native is not None:
        native_gain["ttft_speedup"] = _ratio_gain(
            selected["performance"]["ttft_p95_ms"], native["performance"]["ttft_p95_ms"]
        )
        native_gain["throughput_gain"] = _ratio_gain(
            native["performance"]["successful_requests_per_second"],
            selected["performance"]["successful_requests_per_second"],
        )
    serving_gain = {"comparison": "Native Memory -> Native Serving", "ttft_speedup": None, "throughput_gain": None}
    if native is not None and serving is not None:
        serving_gain["ttft_speedup"] = _ratio_gain(
            native["performance"]["ttft_p95_ms"], serving["performance"]["ttft_p95_ms"]
        )
        serving_gain["throughput_gain"] = _ratio_gain(
            serving["performance"]["successful_requests_per_second"],
            native["performance"]["successful_requests_per_second"],
        )
    return {"context_gain": context_gain, "native_gain": native_gain, "serving_gain": serving_gain}


def _quality_gate(modes: Mapping[str, Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    full = modes["full_context"]["quality"]
    selected = modes["selected_context"]["quality"]
    if selected["success"] is False:
        return {"passed": False, "threshold": threshold, "reason": "Selected Context reported failure."}
    if selected["success"] is True:
        return {"passed": True, "threshold": threshold, "reason": "Selected Context reported success."}
    for metric in ("f1", "em", "gold_log_probability"):
        baseline, candidate = full[metric], selected[metric]
        if baseline is not None and candidate is not None:
            passed = float(candidate) >= float(baseline) * threshold
            return {
                "passed": passed,
                "threshold": threshold,
                "reason": f"Selected Context {metric} retention is {float(candidate) / float(baseline):.3f}." if baseline else f"Compared {metric}.",
            }
    return {"passed": None, "threshold": threshold, "reason": "Quality is not measured."}


def _mode_quality_pass(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any], threshold: float,
) -> bool:
    quality = candidate["quality"]
    if quality["success"] is not None:
        return quality["success"] is True
    for metric in ("f1", "em", "gold_log_probability"):
        candidate_value = quality[metric]
        baseline_value = baseline["quality"][metric]
        if candidate_value is not None and baseline_value is not None:
            return float(candidate_value) >= float(baseline_value) * threshold
    return False


def recommend_run(document: Mapping[str, Any], *, allow_unqualified_native: bool = False) -> dict[str, Any]:
    """Apply conservative quality and incremental-economics gates."""

    modes = document["modes"]
    gate = document.get("quality_gate") or _quality_gate(modes, 0.95)
    engine = document["identity"]["engine"]
    capabilities = document["engine_capabilities"]
    fallback = "Full Context"
    if gate["passed"] is not True:
        return {
            "recommended_mode": None,
            "profile": None,
            "status": "Qualification pending" if gate["passed"] is None else "Blocked",
            "confidence": "None",
            "reason": gate["reason"],
            "limitations": ["No production mode can be recommended until the quality gate passes."],
            "fallback": fallback,
        }
    if not document["selection"]["frozen"]:
        return {
            "recommended_mode": None,
            "profile": None,
            "status": "Qualification pending",
            "confidence": "None",
            "reason": "The selector output was not frozen across execution modes.",
            "limitations": ["Record one selector digest before comparing execution modes."],
            "fallback": fallback,
        }

    recommendation = {
        "recommended_mode": "Selected Context",
        "profile": "recommended",
        "status": "Recommended",
        "confidence": document.get("evidence", {}).get("tier", "Candidate"),
        "reason": "The quality gate passed; Selected Context is the qualified portable baseline.",
        "limitations": [],
        "fallback": fallback,
    }
    native = modes.get("native_memory_hot")
    native_gain = document["attribution"]["native_gain"]
    native_economics = any(
        value is not None and float(value) > 1.0
        for value in (native_gain.get("ttft_speedup"), native_gain.get("throughput_gain"))
    )
    native_status = str(capabilities.get("native_memory", "Not measured")).lower()
    native_qualified = native_status in {"validated", "measured", "recommended"}
    warm = modes.get("native_memory_warm")
    threshold = float(gate.get("threshold", 0.95))
    native_quality = (
        native is not None
        and warm is not None
        and _mode_quality_pass(native, modes["selected_context"], threshold)
        and _mode_quality_pass(warm, modes["selected_context"], threshold)
    )
    native_complete = native is not None and warm is not None and all((
        native["performance"]["ttft_p95_ms"] is not None,
        warm["performance"]["ttft_p95_ms"] is not None,
        native["memory"]["native_memory_bytes"] is not None,
        warm["lifecycle"]["reuse"] is not None,
        native["lifecycle"]["reference_encoding_ms"] is not None,
        document["selection"]["frozen"],
    ))
    if engine == "airllm" and not allow_unqualified_native:
        recommendation["limitations"].append("AirLLM Native Memory has negative measured economics.")
    elif native_quality and native_complete and native_economics and (native_qualified or allow_unqualified_native):
        recommendation.update({
            "recommended_mode": "Native Memory",
            "reason": "Quality passed and Native Memory has a measured incremental economic gain over Selected Context.",
        })
    elif native is not None:
        recommendation["limitations"].append(
            "Native Memory is not promoted because incremental economics are unmeasured, non-positive, or unqualified."
        )
    serving = modes.get("native_serving")
    serving_gain = document["attribution"]["serving_gain"]
    serving_economics = any(
        value is not None and float(value) > 1.0
        for value in (serving_gain.get("ttft_speedup"), serving_gain.get("throughput_gain"))
    )
    serving_status = str(capabilities.get("native_serving", "Not measured")).lower()
    if (
        serving is not None
        and native is not None
        and _mode_quality_pass(serving, native, threshold)
        and serving_economics
        and (serving_status in {"validated", "measured", "recommended"} or allow_unqualified_native)
    ):
        recommendation.update({
            "recommended_mode": "Native Serving",
            "reason": "Quality passed and scheduler-owned serving adds a measured gain over Native Memory.",
        })
    return recommendation


def resolve_run_mode(
    document: Mapping[str, Any], *, allow_unqualified_native: bool = False
) -> dict[str, Any]:
    """Express measured qualification through the common four-axis resolver."""

    modes = document["modes"]
    capabilities = document["engine_capabilities"]
    gate = document.get("quality_gate", {})
    recommendation = recommend_run(
        document, allow_unqualified_native=allow_unqualified_native
    )
    recommended = str(recommendation.get("recommended_mode") or "").lower()

    def quality_status(name: str) -> ModeStatus:
        if name == "selected_context":
            passed = gate.get("passed")
            return (
                ModeStatus.VALIDATED if passed is True
                else ModeStatus.BLOCKED if passed is False
                else ModeStatus.NOT_MEASURED
            )
        row = modes.get(name)
        if row is None or row.get("status") == NOT_MEASURED:
            return ModeStatus.NOT_MEASURED
        baseline = modes["selected_context"] if name != "native_serving" else modes.get("native_memory_hot")
        if baseline is None:
            return ModeStatus.NOT_MEASURED
        return (
            ModeStatus.VALIDATED
            if _mode_quality_pass(row, baseline, float(gate.get("threshold", 0.95)))
            else ModeStatus.BLOCKED
        )

    def economic_status(gain_name: str) -> ModeStatus:
        values = [
            float(value)
            for key, value in document["attribution"][gain_name].items()
            if key != "comparison" and value is not None
        ]
        if not values:
            return ModeStatus.NOT_MEASURED
        if all(value > 1.0 for value in values):
            return ModeStatus.VALIDATED
        if all(value <= 1.0 for value in values):
            return ModeStatus.BLOCKED
        return ModeStatus.QUALIFICATION_PENDING

    native_quality = quality_status("native_memory_hot")
    warm_quality = quality_status("native_memory_warm")
    if ModeStatus.BLOCKED in {native_quality, warm_quality}:
        native_quality = ModeStatus.BLOCKED
    elif ModeStatus.NOT_MEASURED in {native_quality, warm_quality}:
        native_quality = ModeStatus.NOT_MEASURED
    candidates = (
        ModeEvidence(
            ExecutionMode.SELECTED_CONTEXT,
            normalize_status(capabilities.get("selected_context", "available")),
            quality_status("selected_context"),
            ModeStatus.VALIDATED,
            ModeStatus.RECOMMENDED if recommended == "selected context" else ModeStatus.CANDIDATE,
            "Measured Selected Context qualification.",
        ),
        ModeEvidence(
            ExecutionMode.NATIVE_MEMORY,
            normalize_status(capabilities.get("native_memory", "not measured")),
            native_quality,
            economic_status("native_gain"),
            ModeStatus.RECOMMENDED if recommended == "native memory" else ModeStatus.CANDIDATE,
            "Measured Native Memory qualification relative to Selected Context.",
        ),
        ModeEvidence(
            ExecutionMode.NATIVE_SERVING,
            normalize_status(capabilities.get("native_serving", "not measured")),
            quality_status("native_serving"),
            economic_status("serving_gain"),
            ModeStatus.RECOMMENDED if recommended == "native serving" else ModeStatus.CANDIDATE,
            "Measured Native Serving qualification relative to Native Memory.",
        ),
    )
    return ExecutionModeResolver().resolve_candidates(
        ExecutionMode.AUTO,
        candidates,
        allow_unqualified_native=allow_unqualified_native,
    ).to_dict()


class QualificationService:
    """Build, persist, recommend, and report one enterprise qualification run."""

    def __init__(self, registry: EngineProductRegistry | None = None) -> None:
        self.registry = registry or EngineProductRegistry.default()

    def inspect(self, model: str, engine: str, model_metadata: Mapping[str, Any]) -> dict[str, Any]:
        row = self.registry.resolve(engine)
        resolution = ExecutionModeResolver().resolve("auto", row)
        profiles = ["REFERENCE_CORRECTNESS", "BALANCED", "ECONOMY", "QUALITY_MAX_CANDIDATE"]
        return {
            "model": dict(model_metadata),
            "engine": self.registry.summary(row),
            "capabilities": {key: _status(value) for key, value in row["capabilities"].items()},
            "current_recommendation": row["recommended_today"],
            "mode_resolution": resolution.to_dict(),
            "available_profiles": profiles,
            "evidence_status": {"tier": row["evidence"], "as_of": self.registry.payload.get("evidence_as_of")},
            "next_command": f"pra evaluate {model} --engine {row['slug']} --dataset DATASET",
        }

    def evaluate(
        self,
        model: str,
        *,
        engine: str,
        dataset: str,
        output: str | Path,
        measurements: str | Path | None = None,
        include_native_memory: bool = False,
        include_native_serving: bool = False,
        quality_threshold: float = 0.95,
        revision: str | None = None,
        profile: str = "recommended",
    ) -> dict[str, Any]:
        engine_row = self.registry.resolve(engine)
        slug = engine_row["slug"]
        supplied = _read_json(measurements) if measurements else {}
        supplied_modes = supplied.get("modes", supplied.get("measurements", {}))
        if supplied_modes and not isinstance(supplied_modes, Mapping):
            raise ValueError("Measurement input must contain a 'modes' object.")
        unknown_modes = sorted(set(supplied_modes) - set(PUBLIC_MODES))
        if unknown_modes:
            raise ValueError(f"Unknown product measurement modes: {', '.join(unknown_modes)}")
        requested_modes = ["full_context", "selected_context"]
        blocked: list[str] = []
        native_status = str(engine_row["capabilities"]["native_memory"]).lower()
        if include_native_memory:
            if native_status in {"not applicable", "not qualified", "not measured"}:
                blocked.append(f"Native Memory: {engine_row['capabilities']['native_memory']}")
            else:
                requested_modes.extend(("native_memory_hot", "native_memory_warm"))
        serving_status = str(engine_row["capabilities"]["native_serving"]).lower()
        if include_native_serving:
            if serving_status in {"not applicable", "not qualified", "not measured"}:
                blocked.append(f"Native Serving: {engine_row['capabilities']['native_serving']}")
            else:
                if "native_memory_hot" not in requested_modes:
                    requested_modes.extend(("native_memory_hot", "native_memory_warm"))
                requested_modes.append("native_serving")
        modes = {
            name: _normalize_measurement(name, supplied_modes[name])
            if name in supplied_modes else empty_measurement(MODE_LABELS[name])
            for name in requested_modes
        }
        selection = supplied.get("selection", {})
        selector_digest = selection.get("digest") or supplied.get("selector_digest")
        document: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "identity": {
                "model": model,
                "revision": revision,
                "engine": slug,
                "engine_version": supplied.get("engine_version"),
                "hardware": supplied.get("hardware", platform.machine()),
                "workload": supplied.get("workload", Path(dataset).stem),
                "dataset": dataset,
                "profile": profile,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "commit": _git_commit(),
            },
            "engine_capabilities": {
                key: _status(value) for key, value in engine_row["capabilities"].items()
            },
            "selection": {
                "frozen": bool(selector_digest),
                "digest": selector_digest,
                "candidate_count": selection.get("candidate_count"),
                "selected_count": selection.get("selected_count"),
            },
            "modes": modes,
            "blocked_modes": blocked,
            "evidence": {
                "tier": supplied.get("evidence_tier", engine_row["evidence"]),
                "source": str(measurements) if measurements else None,
                "registry": self.registry.source,
            },
        }
        document["attribution"] = derive_attribution(modes)
        document["quality_gate"] = _quality_gate(modes, quality_threshold)
        document["missing_measurements"] = self.missing_measurements(document)
        document["recommendation"] = recommend_run(document)
        document["mode_resolution"] = resolve_run_mode(document)
        self.write_run(output, document)
        return document

    @staticmethod
    def missing_measurements(document: Mapping[str, Any]) -> list[str]:
        missing = []
        for mode, row in document["modes"].items():
            for section in ("quality", "context", "performance", "memory", "lifecycle"):
                for name, value in row[section].items():
                    if value is None:
                        missing.append(f"{mode}.{section}.{name}")
        if len(document["modes"]) > 2 and not document["selection"]["frozen"]:
            missing.append("selection.digest")
        return missing

    def write_run(self, output: str | Path, document: Mapping[str, Any]) -> None:
        root = Path(output)
        root.mkdir(parents=True, exist_ok=True)
        (root / "runs").mkdir(exist_ok=True)
        config = {
            "schema_version": SCHEMA_VERSION,
            "model": document["identity"]["model"],
            "revision": document["identity"]["revision"],
            "engine": document["identity"]["engine"],
            "dataset": document["identity"]["dataset"],
            "profile": document["identity"]["profile"],
        }
        (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        _write_json(root / "environment.json", environment_report(self.registry))
        _write_json(root / "quality.json", {"quality_gate": document["quality_gate"], "modes": {k: v["quality"] for k, v in document["modes"].items()}})
        _write_json(root / "metrics.json", document)
        _write_json(root / "recommendation.json", document["recommendation"])
        (root / "report.md").write_text(render_markdown_report(document), encoding="utf-8")


def load_run(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    source = root / "metrics.json" if root.is_dir() else root
    value = _read_json(source)
    if _is_canonical_evidence(value):
        record = _canonical_record(value)
        return record.serialize_for_control_plane()
    if value.get("schema_version") != SCHEMA_VERSION or "modes" not in value:
        raise ValueError(f"Not a PRA qualification run: {source}")
    return value


def _is_canonical_evidence(document: Mapping[str, Any]) -> bool:
    return all(name in document for name in ("key", "metric_definitions", "conditions", "provenance"))


def _canonical_record(document: Mapping[str, Any]) -> CanonicalEvidenceRecord:
    fields = CanonicalEvidenceRecord.model_fields
    return CanonicalEvidenceRecord.model_validate({name: document[name] for name in fields if name in document})


def _canonical_markdown_report(record: CanonicalEvidenceRecord) -> str:
    key = record.key
    lines = [
        "# PRA Canonical Evidence Report", "",
        f"- Task/dataset: `{key.task}`",
        f"- Hardware: `{key.hardware}`",
        f"- Engine: `{key.engine} {key.engine_version}`",
        f"- Model: `{key.model_id}`",
        f"- Model revision: `{key.model_revision}`",
        f"- Mode/profile: `{key.mode}` / `{key.profile}`",
        f"- Evidence tier: `{record.evidence_tier}`", "",
        "Deltas preserve the mathematical sign and use No PRA as the baseline. Metric direction states how to interpret that sign.", "",
        "## Matched conditions", "",
        render_canonical_markdown_table(record).rstrip(), "",
        "## Provenance", "",
        f"- Cohort: `{record.provenance.cohort}`",
        f"- Date: `{record.provenance.date}`",
        f"- Commit: `{record.provenance.commit or NOT_MEASURED}`",
        f"- Concurrency: `{record.provenance.concurrency or NOT_MEASURED}`",
        f"- Runs: `{len(record.provenance.run_ids)}`", "",
    ]
    return "\n".join(lines)


def _canonical_html_report(record: CanonicalEvidenceRecord) -> str:
    key = record.key
    rows = []
    for name, definition in record.metric_definitions.items():
        observations = [record.conditions[condition].metrics.get(name) for condition in EvidenceCondition]
        deltas = [
            record.delta(name, condition)
            for condition in (EvidenceCondition.PRA_NO_ADAPTOR, EvidenceCondition.PRA_ADAPTOR_BUNDLE)
        ]
        values = [
            name, definition.unit, definition.direction.value,
            *[
                MeasurementState.NOT_MEASURED.value
                if observation is None
                else observation.state.value
                if observation.state != MeasurementState.MEASURED
                else f"{observation.value:.6g}"
                for observation in observations
            ],
            *[
                delta.state.value
                if delta.state != MeasurementState.MEASURED or delta.delta is None
                else f"{delta.delta:+.6g}" + (
                    "" if delta.percent_delta is None else f" ({delta.percent_delta:+.2f}%)"
                )
                for delta in deltas
            ],
        ]
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in values) + "</tr>")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PRA Canonical Evidence Report</title>
<style>
body{{font:16px/1.55 system-ui;max-width:1440px;margin:40px auto;padding:0 24px;color:#182124}}
h1,h2{{line-height:1.2}} table{{width:100%;border-collapse:collapse;margin:20px 0}} th,td{{padding:9px;border:1px solid #d9dfdc;text-align:left}}
th{{background:#eef3f0}} code{{font-size:.9em}} .meta{{color:#52605b}} .blocked{{color:#8a331d}}
</style></head><body><main>
<h1>PRA Canonical Evidence Report</h1>
<p class="meta">Task <code>{html.escape(key.task)}</code> &middot; Engine <code>{html.escape(key.engine)} {html.escape(key.engine_version)}</code> &middot; Model <code>{html.escape(key.model_id)}</code> &middot; Profile <code>{html.escape(key.profile)}</code></p>
<p>Deltas are candidate minus No PRA; signs are not inverted. Missing measurements retain their explicit state.</p>
<h2>Matched conditions</h2>
<table><thead><tr><th>Metric</th><th>Unit</th><th>Direction</th><th>No PRA</th><th>PRA - No Adaptor</th><th>PRA - Adaptor Bundle</th><th>Delta No Adaptor</th><th>Delta Bundle</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Provenance</h2><p>Cohort <code>{html.escape(record.provenance.cohort)}</code><br>Date <code>{html.escape(record.provenance.date)}</code><br>Commit <code>{html.escape(record.provenance.commit or NOT_MEASURED)}</code><br>Runs <code>{len(record.provenance.run_ids)}</code></p>
</main></body></html>
"""


def render_markdown_report(document: Mapping[str, Any]) -> str:
    identity = document["identity"]
    lines = [
        "# PRA Optimization Assessment",
        "",
        f"- Model: `{identity['model']}`",
        f"- Engine: `{identity['engine']}`",
        f"- Dataset/workload: `{identity['dataset']}`",
        f"- Profile: `{identity['profile']}`",
        f"- Timestamp: `{identity['timestamp']}`",
        "",
        "## Quality gate",
        "",
        f"Status: **{document['quality_gate']['passed']}**. {document['quality_gate']['reason']}",
        "",
        "## Matched comparisons",
        "",
        "| Mode | Status | F1 | EM | Visible tokens | TTFT p95 (ms) | Successful req/s | Peak bytes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in document["modes"].items():
        values = [
            row["label"], row["status"], row["quality"]["f1"], row["quality"]["em"],
            row["context"]["visible_input_tokens"], row["performance"]["ttft_p95_ms"],
            row["performance"]["successful_requests_per_second"], row["memory"]["peak_bytes"],
        ]
        lines.append("| " + " | ".join(NOT_MEASURED if value is None else str(value) for value in values) + " |")
    lines.extend(["", "## Attribution", ""])
    for key in ("context_gain", "native_gain", "serving_gain"):
        gain = document["attribution"][key]
        measured = ", ".join(
            f"{name}={value:.3f}" for name, value in gain.items()
            if name != "comparison" and value is not None
        ) or NOT_MEASURED
        lines.append(f"- **{gain['comparison']}**: {measured}")
    recommendation = document["recommendation"]
    lines.extend([
        "", "## Recommendation", "",
        f"**{recommendation['recommended_mode'] or 'No production recommendation'}**",
        "", recommendation["reason"], "",
        f"Fallback: {recommendation['fallback']}",
        "", "## Provenance", "",
        f"- Commit: `{identity.get('commit') or NOT_MEASURED}`",
        f"- Engine registry: `{document['evidence']['registry']}`",
        f"- Measurement source: `{document['evidence'].get('source') or NOT_MEASURED}`",
        f"- Frozen selection digest: `{document['selection'].get('digest') or NOT_MEASURED}`",
        "", "## Missing measurements", "",
    ])
    lines.extend(f"- `{name}`" for name in document["missing_measurements"])
    return "\n".join(lines) + "\n"


def render_report(document: Mapping[str, Any], format_name: str) -> str:
    if _is_canonical_evidence(document):
        record = _canonical_record(document)
        if format_name == "json":
            return json.dumps(record.serialize_for_control_plane(), indent=2, sort_keys=True) + "\n"
        if format_name == "md":
            return _canonical_markdown_report(record)
        if format_name == "html":
            return _canonical_html_report(record)
        raise ValueError(f"Unknown report format: {format_name}")
    if format_name == "json":
        return json.dumps(document, indent=2, sort_keys=True, default=str) + "\n"
    markdown = render_markdown_report(document)
    if format_name == "md":
        return markdown
    if format_name == "html":
        identity = document["identity"]
        rows = []
        for row in document["modes"].values():
            values = (
                row["label"], row["status"], row["quality"]["f1"], row["quality"]["em"],
                row["context"]["visible_input_tokens"], row["performance"]["ttft_p95_ms"],
                row["performance"]["successful_requests_per_second"], row["memory"]["peak_bytes"],
            )
            cells = "".join(
                f"<td>{html.escape(NOT_MEASURED if value is None else str(value))}</td>" for value in values
            )
            rows.append(f"<tr>{cells}</tr>")
        recommendation = document["recommendation"]
        missing = "".join(f"<li><code>{html.escape(name)}</code></li>" for name in document["missing_measurements"])
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PRA Optimization Assessment</title>
<style>
body{{font:16px/1.55 system-ui;max-width:1180px;margin:40px auto;padding:0 24px;color:#182124}}
h1,h2{{line-height:1.2}} table{{width:100%;border-collapse:collapse;margin:20px 0}} th,td{{padding:10px;border:1px solid #d9dfdc;text-align:left}}
th{{background:#eef3f0}} .recommendation{{border-left:5px solid #17745b;background:#edf7f3;padding:18px}} code{{font-size:.9em}} .meta{{color:#52605b}}
</style></head><body><main>
<h1>PRA Optimization Assessment</h1>
<p class="meta">Model <code>{html.escape(str(identity['model']))}</code> · Engine <code>{html.escape(str(identity['engine']))}</code> · Dataset <code>{html.escape(str(identity['dataset']))}</code></p>
<h2>Quality gate</h2><p><strong>{html.escape(str(document['quality_gate']['passed']))}</strong> — {html.escape(str(document['quality_gate']['reason']))}</p>
<h2>Matched comparisons</h2><table><thead><tr><th>Mode</th><th>Status</th><th>F1</th><th>EM</th><th>Visible tokens</th><th>TTFT p95 ms</th><th>Successful req/s</th><th>Peak bytes</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Recommendation</h2><div class="recommendation"><strong>{html.escape(str(recommendation['recommended_mode'] or 'No production recommendation'))}</strong><p>{html.escape(str(recommendation['reason']))}</p><p>Fallback: {html.escape(str(recommendation['fallback']))}</p></div>
<h2>Provenance</h2><p>Commit <code>{html.escape(str(identity.get('commit') or NOT_MEASURED))}</code><br>Selector <code>{html.escape(str(document['selection'].get('digest') or NOT_MEASURED))}</code></p>
<h2>Missing measurements</h2><ul>{missing}</ul>
</main></body></html>
"""
    raise ValueError(f"Unknown report format: {format_name}")


def assessment_init(name: str, *, root: str | Path = ".pra/assessments") -> Path:
    destination = Path(root) / name
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "runs").mkdir(exist_ok=True)
    config = destination / "config.yaml"
    if not config.exists():
        config.write_text(
            yaml.safe_dump({
                "schema_version": SCHEMA_VERSION,
                "name": name,
                "model": "MODEL",
                "engine": "ENGINE",
                "dataset": "DATASET",
                "profile": "recommended",
                "include_native_memory": False,
                "include_native_serving": False,
            }, sort_keys=False),
            encoding="utf-8",
        )
    return destination
