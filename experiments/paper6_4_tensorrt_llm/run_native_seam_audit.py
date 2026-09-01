"""Audit whether installed TensorRT-LLM exposes a maintainable PRA E2 seam.

The audit is intentionally stricter than checking whether paged K/V or a KV
connector exists.  Native PRA requires request-scoped non-prefix block IDs,
separate local/memory geometry, and one attention normalization.  Without all
three, concatenating pages merely emulates a sequential prefix and can corrupt
positions or cache writes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
from pathlib import Path
from typing import Mapping, Sequence


SOURCE_FILES = {
    "metadata": "_torch/metadata.py",
    "attention_interface": "_torch/attention_backend/interface.py",
    "attention_backend": "_torch/attention_backend/trtllm.py",
    "model_engine": "_torch/pyexecutor/model_engine.py",
    "request_conversion": "_torch/pyexecutor/llm_request.py",
    "resource_manager": "_torch/pyexecutor/resource_manager.py",
    "kv_connector": "_torch/pyexecutor/kv_cache_connector.py",
}

NATIVE_FIELD_NAMES = {
    "pra_block_ids",
    "selected_kv_block_ids",
    "external_nonprefix_block_ids",
}


def _line_evidence(text: str, needles: Sequence[str]) -> list[Mapping[str, object]]:
    """Return the first exact source location for each relevant literal."""

    lines = text.splitlines()
    evidence: list[Mapping[str, object]] = []
    for needle in needles:
        for number, line in enumerate(lines, start=1):
            if needle in line:
                evidence.append(
                    {"needle": needle, "line": number, "source": line.strip()}
                )
                break
    return evidence


def classify_sources(
    request_attributes: Sequence[str], source_text: Mapping[str, str]
) -> Mapping[str, object]:
    """Classify the seam from independent request, metadata, and kernel gates."""

    joined = "\n".join(source_text.values())
    request_hook = bool(NATIVE_FIELD_NAMES.intersection(request_attributes))
    metadata_hook = any(
        name in source_text.get("attention_interface", "")
        for name in NATIVE_FIELD_NAMES
    )
    kernel_hook = any(
        marker in source_text.get("attention_backend", "")
        for marker in ("pra_memory_block_offsets", "external_nonprefix_block_offsets")
    )
    connector_present = all(
        marker in source_text.get("kv_connector", "")
        for marker in ("start_load_kv", "wait_for_layer_load")
    )
    request_owned_table = all(
        marker in joined
        for marker in ("copy_batch_block_offsets", "request_ids")
    )
    one_fused_attention_call = "thop.attention(" in source_text.get(
        "attention_backend", ""
    )
    maintainable = request_hook and metadata_hook and kernel_hook
    return {
        "public_request_nonprefix_field": request_hook,
        "attention_metadata_nonprefix_field": metadata_hook,
        "fused_kernel_nonprefix_input": kernel_hook,
        "official_kv_connector_present": connector_present,
        "block_table_is_request_owned": request_owned_table,
        "one_fused_attention_call": one_fused_attention_call,
        "maintainable_narrow_seam": maintainable,
        "decision": (
            "NARROW_NATIVE_SEAM_AVAILABLE"
            if maintainable
            else "STOP_NO_MAINTAINABLE_NARROW_SEAM"
        ),
    }


def _gpu() -> Mapping[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        line = subprocess.check_output(command, text=True, timeout=15).splitlines()[0]
        name, compute_capability, memory_mib, driver = map(str.strip, line.split(","))
        return {
            "available": True,
            "name": name,
            "compute_capability": compute_capability,
            "memory_mib": int(memory_mib),
            "driver_version": driver,
        }
    except (OSError, subprocess.SubprocessError, IndexError, ValueError) as error:
        return {"available": False, "error": str(error)}


def audit(module_root: Path | None = None) -> Mapping[str, object]:
    """Inspect an installed TensorRT-LLM package and emit a version-bound map."""

    if module_root is None:
        spec = importlib.util.find_spec("tensorrt_llm")
        if spec is None or spec.submodule_search_locations is None:
            raise RuntimeError("tensorrt_llm is not installed")
        module_root = Path(next(iter(spec.submodule_search_locations)))

    source_text: dict[str, str] = {}
    source_files: dict[str, Mapping[str, object]] = {}
    evidence_needles = {
        "metadata": ("block_ids_per_seq", "num_cached_tokens_per_seq"),
        "attention_interface": ("class AttentionMetadata", "kv_cache_params"),
        "attention_backend": (
            "copy_batch_block_offsets",
            "self.kv_cache_block_offsets",
            "thop.attention(",
        ),
        "model_engine": ("attn_metadata.prepare()", "num_cached_tokens_per_seq"),
        "request_conversion": ("executor_request_to_llm_request", "cache_salt_id"),
        "resource_manager": ("def pin_blocks", "def unpin_blocks_by_id"),
        "kv_connector": ("start_load_kv", "wait_for_layer_load"),
    }
    for name, relative in SOURCE_FILES.items():
        path = module_root / relative
        text = path.read_text(encoding="utf-8")
        source_text[name] = text
        source_files[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "evidence": _line_evidence(text, evidence_needles[name]),
        }

    executor = importlib.import_module("tensorrt_llm.bindings.executor")
    request_attributes = sorted(dir(executor.Request))
    gates = classify_sources(request_attributes, source_text)

    patch_surfaces = [
        {
            "surface": "request ABI and Python conversion",
            "why": "carry authorized selected block identities into each request",
            "required": not gates["public_request_nonprefix_field"],
        },
        {
            "surface": "scheduler and cache-manager lifecycle",
            "why": "retain, pin, authorize, cancel, and release non-prefix pages",
            "required": True,
        },
        {
            "surface": "attention metadata and CUDA-graph buffers",
            "why": "represent local and memory block tables and lengths independently",
            "required": not gates["attention_metadata_nonprefix_field"],
        },
        {
            "surface": "fused attention operator ABI/kernel",
            "why": "combine local and memory scores in one softmax without shifting writes",
            "required": not gates["fused_kernel_nonprefix_input"],
        },
        {
            "surface": "topology-aware connector persistence",
            "why": "load physical pages without treating connector prefix matches as PRA routing",
            "required": True,
        },
    ]
    rejected_shortcuts = [
        {
            "strategy": "prepend selected pages to the ordinary block table",
            "reason": "changes sequential K/V length, causal geometry, and cache-write offsets",
        },
        {
            "strategy": "report connector page loading as native PRA",
            "reason": "the connector restores request-token ranges but does not expose them as independent memory",
        },
        {
            "strategy": "reuse encoder-decoder cross attention",
            "reason": "decoder-only checkpoints lack the corresponding modules and trained projection geometry",
        },
    ]
    return {
        "schema_version": "1.0",
        "experiment": "paper6_4_tensorrt_llm_native_seam_audit_v1",
        "evidence_tier": "LIVE_INSTALLED_SOURCE_AUDIT",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("tensorrt-llm", "tensorrt", "torch")
        },
        "gpu": _gpu(),
        "request_attributes": request_attributes,
        "source_files": source_files,
        "gates": gates,
        "patch_surfaces": patch_surfaces,
        "rejected_shortcuts": rejected_shortcuts,
        "scope": (
            "The result is version-specific. It rejects a narrow maintained 1.2.1 "
            "extension, not the feasibility of an NVIDIA-supported future E2 API."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--module-root", type=Path)
    args = parser.parse_args()
    payload = audit(args.module_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "gates": payload["gates"]}, indent=2))


if __name__ == "__main__":
    main()
