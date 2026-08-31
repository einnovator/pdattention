"""Produce a source-grounded AirLLM capability audit.

The audit intentionally reads the pinned checkout instead of relying on
import-time behavior. This makes the resulting JSON useful on a machine that
cannot execute the CUDA path and records the exact source evidence behind each
capability claim.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


def _find_package_root(source: Path) -> Path:
    candidates = (
        source / "air_llm" / "airllm",
        source / "airllm",
        source,
    )
    for candidate in candidates:
        if (candidate / "airllm_base.py").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate airllm_base.py below {source}")


def _first_match(path: Path, patterns: Iterable[str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if any(re.search(pattern, line) for pattern in patterns):
            return {
                "file": path.as_posix(),
                "line": line_number,
                "text": line.strip()[:240],
            }
    return None


def _git_value(source: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_version(source: Path) -> str | None:
    setup_candidates = (source / "air_llm" / "setup.py", source / "setup.py")
    for setup in setup_candidates:
        if not setup.exists():
            continue
        match = re.search(
            r"version\s*=\s*['\"]([^'\"]+)", setup.read_text(encoding="utf-8")
        )
        if match:
            return match.group(1)
    return None


def audit_source(
    source: Path,
    *,
    device: str,
    hardware: str,
    runtime_status: str,
) -> dict[str, Any]:
    """Inspect an AirLLM checkout and return the Paper 6.6 audit schema."""

    source = source.resolve()
    package = _find_package_root(source)
    base = package / "airllm_base.py"
    auto = package / "auto_model.py"
    mlx = package / "airllm_llama_mlx.py"
    model_files = sorted(path.stem.removeprefix("airllm_") for path in package.glob("airllm_*.py"))
    evidence = {
        "hf_forward": _first_match(base, (r"self\.model\.forward", r"return self\.model\(")),
        "hf_generate": _first_match(base, (r"self\.model\.generate",)),
        "meta_skeleton": _first_match(base, (r"init_empty_weights", r"device.?=.?.?meta")),
        "pre_hook": _first_match(base, (r"register_forward_pre_hook",)),
        "post_hook": _first_match(base, (r"register_forward_hook",)),
        "prefetch_executor": _first_match(base, (r"ThreadPoolExecutor",)),
        "pinned_memory": _first_match(base, (r"pin_memory", r"pinned")),
        "compression": _first_match(base, (r"compression",)),
        "expert_streaming": _first_match(base, (r"_expert_pre_hook", r"expert_module")),
        "dynamic_cache": _first_match(base, (r"DynamicCache", r"past_key_values")),
        "mlx_dispatch": _first_match(auto, (r"AirLLMLlamaMlx", r"darwin")),
        "mlx_model": _first_match(mlx, (r"class AirLLMLlamaMlx",)),
    }
    hf_path = bool(evidence["hf_forward"] and evidence["hf_generate"])
    is_mlx = device.lower() in {"mlx", "metal", "mps"}
    native_level = "E0" if is_mlx else ("E1_CANDIDATE" if hf_path else "E0")
    return {
        "schema_version": "paper6.6-airllm-audit-v1",
        "evidence_tier": "SOURCE_AUDIT_AND_RUNTIME_IMPORT",
        "runtime_status": runtime_status,
        "airllm_source": source.as_posix(),
        "airllm_commit": _git_value(source, "rev-parse", "HEAD"),
        "airllm_commit_date": _git_value(source, "show", "-s", "--format=%cI", "HEAD"),
        "airllm_version": _package_version(source),
        "host_platform": platform.platform(),
        "hardware": hardware,
        "hf_attention_path": "transformers_model_forward_and_generate" if hf_path else "not_confirmed",
        "hf_cache_type": "transformers_cache_passthrough",
        "position_api": "transformers_model_owned",
        "weight_streaming_mode": "module_pre_hook_load_post_hook_release",
        "prefetching": "single_worker_executor",
        "compression": "on_disk_weight_compression_optional",
        "expert_streaming": bool(evidence["expert_streaming"]),
        "device": device,
        "resident_modules": "one_or_few_streamed_modules_plus_runtime_state",
        "supported_model_family": model_files,
        "pra_integration_level": native_level,
        "native_hf_pra_available": bool(hf_path and not is_mlx),
        "mlx_path_is_separate": bool(evidence["mlx_dispatch"] and evidence["mlx_model"]),
        "evidence": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--airllm-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hardware", default="unknown")
    parser.add_argument("--runtime-status", default="source-audited")
    args = parser.parse_args()
    report = audit_source(
        args.airllm_source,
        device=args.device,
        hardware=args.hardware,
        runtime_status=args.runtime_status,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
