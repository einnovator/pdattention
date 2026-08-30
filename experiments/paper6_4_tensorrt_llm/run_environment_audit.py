"""Audit installed TensorRT-LLM APIs without converting availability into claims."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Iterable, Mapping


def classify_api(
    request_attributes: Iterable[str],
    cache_manager_attributes: Iterable[str],
    connector_worker_abstract: Iterable[str],
    connector_scheduler_abstract: Iterable[str],
) -> Mapping[str, object]:
    """Classify E0, connector, and native attachment as independent gates."""

    request = set(request_attributes)
    manager = set(cache_manager_attributes)
    worker = set(connector_worker_abstract)
    scheduler = set(connector_scheduler_abstract)
    connector_ready = {
        "register_kv_caches",
        "start_load_kv",
        "wait_for_layer_load",
        "save_kv_layer",
    }.issubset(worker) and {
        "build_connector_meta",
        "get_num_new_matched_tokens",
        "update_state_after_alloc",
    }.issubset(scheduler)
    paged_ready = {
        "get_cache_block_ids",
        "pin_blocks",
        "unpin_blocks_by_id",
    }.issubset(manager)
    # The official 1.2 connector can load contiguous request-token cache pages.
    # E2 additionally needs a request-visible field or manager operation for
    # selected non-prefix blocks; do not infer it from generic page loading.
    native_names = {
        "pra_block_ids",
        "selected_kv_block_ids",
        "external_nonprefix_block_ids",
        "attach_nonprefix_blocks",
    }
    nonprefix_hook = bool(native_names & (request | manager))
    return {
        "e0_openai_facade": "READY",
        "cache_salt": "cache_salt_id" in request,
        "priority_retention": "kv_cache_retention_config" in request,
        "paged_cache_manager": paged_ready,
        "official_connector_interfaces": connector_ready,
        "native_nonprefix_attachment_hook": nonprefix_hook,
        "e2_status": (
            "PUBLIC_NONPREFIX_ATTACHMENT_AVAILABLE"
            if nonprefix_hook
            else "BLOCKED_NO_PUBLIC_NONPREFIX_ATTACHMENT_HOOK"
        ),
    }


def _nvidia_smi() -> Mapping[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        line = subprocess.check_output(command, text=True, timeout=15).splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError) as error:
        return {"available": False, "error": str(error)}
    name, compute_capability, memory_mib, driver = map(str.strip, line.split(","))
    return {
        "available": True,
        "name": name,
        "compute_capability": compute_capability,
        "memory_mib": int(memory_mib),
        "driver_version": driver,
    }


def audit() -> Mapping[str, object]:
    executor = importlib.import_module("tensorrt_llm.bindings.executor")
    batch_manager = importlib.import_module(
        "tensorrt_llm.bindings.internal.batch_manager"
    )
    connector = importlib.import_module(
        "tensorrt_llm._torch.pyexecutor.kv_cache_connector"
    )
    request_attributes = dir(executor.Request)
    manager_attributes = dir(batch_manager.KVCacheManager)
    worker_abstract = connector.KvCacheConnectorWorker.__abstractmethods__
    scheduler_abstract = connector.KvCacheConnectorScheduler.__abstractmethods__
    return {
        "schema_version": "1.0",
        "experiment": "paper6_4_tensorrt_llm_environment_audit_v1",
        "evidence_tier": "LIVE_PACKAGE_AND_HARDWARE_AUDIT",
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
        "gpu": _nvidia_smi(),
        "api": {
            "request_attributes": sorted(request_attributes),
            "kv_cache_manager_attributes": sorted(manager_attributes),
            "connector_worker_abstract": sorted(worker_abstract),
            "connector_scheduler_abstract": sorted(scheduler_abstract),
        },
        "gates": classify_api(
            request_attributes,
            manager_attributes,
            worker_abstract,
            scheduler_abstract,
        ),
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
