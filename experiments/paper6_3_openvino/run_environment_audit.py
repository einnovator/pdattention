"""Audit installed OpenVINO GenAI APIs without converting availability into claims."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Iterable, Mapping


SCHEDULER_CONTROLS = {
    "cache_size",
    "num_kv_blocks",
    "max_num_seqs",
    "max_num_batched_tokens",
    "enable_prefix_caching",
    "dynamic_split_fuse",
    "use_cache_eviction",
}
NATIVE_ATTACHMENT_NAMES = {
    "attach_nonprefix_kv",
    "external_kv",
    "load_selected_kv",
    "selected_kv",
    "set_kv_cache",
}


def classify_api(
    scheduler_attributes: Iterable[str], pipeline_attributes: Iterable[str]
) -> Mapping[str, object]:
    """Classify scheduler controls and arbitrary non-prefix attachment separately."""

    scheduler = set(scheduler_attributes)
    pipeline = set(pipeline_attributes)
    controls = {name: name in scheduler for name in sorted(SCHEDULER_CONTROLS)}
    native_hooks = sorted(NATIVE_ATTACHMENT_NAMES & pipeline)
    return {
        "e0_pipeline": "READY",
        "continuous_batching_controls": controls,
        "continuous_batching_ready": all(controls.values()),
        "native_nonprefix_attachment_hooks": native_hooks,
        "e2_status": (
            "PUBLIC_NONPREFIX_ATTACHMENT_AVAILABLE"
            if native_hooks
            else "BLOCKED_NO_PUBLIC_NONPREFIX_ATTACHMENT_HOOK"
        ),
    }


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _devices(openvino_module) -> list[Mapping[str, object]]:
    core = openvino_module.Core()
    rows = []
    for device in core.available_devices:
        row: dict[str, object] = {"device": device}
        for key, output_name in (
            ("FULL_DEVICE_NAME", "full_name"),
            ("DEVICE_ARCHITECTURE", "architecture"),
            ("INFERENCE_PRECISION_HINT", "precision_hint"),
        ):
            try:
                row[output_name] = str(core.get_property(device, key))
            except Exception as error:
                row[output_name] = {"unavailable": type(error).__name__}
        rows.append(row)
    return rows


def audit() -> Mapping[str, object]:
    openvino = importlib.import_module("openvino")
    genai = importlib.import_module("openvino_genai")
    scheduler = genai.SchedulerConfig()
    pipeline_type = genai.LLMPipeline
    return {
        "schema_version": "1.0",
        "experiment": "paper6_3_openvino_environment_audit_v1",
        "evidence_tier": "LIVE_PACKAGE_AND_HARDWARE_AUDIT",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "packages": {
            "openvino": _version("openvino"),
            "openvino-genai": _version("openvino-genai"),
            "openvino-tokenizers": _version("openvino-tokenizers"),
            "ovms": _version("ovms"),
        },
        "devices": _devices(openvino),
        "api": {
            "scheduler_attributes": sorted(dir(scheduler)),
            "pipeline_attributes": sorted(dir(pipeline_type)),
            "scheduler_repr": str(scheduler),
        },
        "gates": classify_api(dir(scheduler), dir(pipeline_type)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "gates": payload["gates"]}, indent=2))


if __name__ == "__main__":
    main()
