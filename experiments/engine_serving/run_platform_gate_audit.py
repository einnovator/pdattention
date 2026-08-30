"""Record engine-package and hardware gates for PRA serving experiments."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from importlib import metadata
from pathlib import Path


PACKAGES = {
    "mlx": "mlx",
    "mlx_lm": "mlx-lm",
    "sglang": "sglang",
    "vllm": "vllm",
    "lmcache": "lmcache",
    "mooncake": "mooncake",
    "nixl": "nixl",
    "hf3fs": "hf3fs",
    "aibrix": "aibrix",
}


def _package(module: str, distribution: str) -> dict[str, object]:
    available = importlib.util.find_spec(module) is not None
    version = None
    if available:
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            version = "source-checkout-or-unknown"
    return {"available": available, "version": version}


def audit() -> dict[str, object]:
    packages = {
        name: _package(module, distribution)
        for name, (module, distribution) in {
            name: (name, distribution) for name, distribution in PACKAGES.items()
        }.items()
    }
    torch_info: dict[str, object] = {"available": False}
    if importlib.util.find_spec("torch") is not None:
        import torch

        cuda = bool(torch.cuda.is_available())
        capability = None
        device = None
        if cuda:
            capability = list(torch.cuda.get_device_capability(0))
            device = torch.cuda.get_device_name(0)
        torch_info = {
            "available": True,
            "version": torch.__version__,
            "cuda_available": cuda,
            "cuda_version": torch.version.cuda,
            "device": device,
            "compute_capability": capability,
            "modern_cuda_gate": bool(capability and capability[0] >= 8),
            "mps_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
        }
    off_node = [
        name
        for name in ("mooncake", "nixl", "hf3fs", "aibrix")
        if packages[name]["available"]
    ]
    return {
        "schema_version": "1.0",
        "experiment": "pra_engine_platform_gate_audit_v1",
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "packages": packages,
        "torch": torch_info,
        "gates": {
            "mlx_native_available": packages["mlx"]["available"],
            "sglang_runtime_available": packages["sglang"]["available"],
            "vllm_runtime_available": packages["vllm"]["available"],
            "lmcache_connector_available": packages["lmcache"]["available"],
            "sglang_off_node_backend_available": bool(off_node),
            "sglang_off_node_backends": off_node,
            "cuda_cc80_or_newer": bool(torch_info.get("modern_cuda_gate", False)),
        },
        "interpretation": {
            "off_node": (
                "READY_FOR_BACKEND_EXPERIMENT"
                if off_node
                else "BLOCKED_NO_SUPPORTED_OFF_NODE_BACKEND_INSTALLED"
            ),
            "cuda_vllm": (
                "READY_FOR_CUDA_REPRODUCTION"
                if torch_info.get("modern_cuda_gate")
                else "BLOCKED_REQUIRES_CUDA_COMPUTE_CAPABILITY_8_OR_NEWER"
            ),
            "lmcache": (
                "READY_FOR_CONNECTOR_EXPERIMENT"
                if packages["lmcache"]["available"]
                else "BLOCKED_LMCACHE_NOT_INSTALLED"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
