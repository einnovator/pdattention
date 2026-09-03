"""Validate quantized bundle mappings against immutable Hub model configs.

This check deliberately stops short of claiming end-task or native-memory
qualification. It proves that the published structural adapter matches the
declared layer, head, and quantization geometry for the exact revision.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper4_5_runtime.build_hf_catalog_bundles import (
    QUANTIZED_RESULTS,
    SPECS,
    _quantization_manifest,
)


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("text_config")
    return value if isinstance(value, dict) else config


def _expected(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "architecture": spec["architecture"],
        "layers": spec["layers"],
        "hidden_size": spec["hidden_size"],
        "query_heads": spec["heads"]["query"],
        "kv_heads": spec["heads"]["kv"],
        "head_dim": spec["heads"]["head_dim"],
        "quantization": _quantization_manifest(spec),
    }


def _observed(config: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    text = _text_config(config)
    query_heads = int(text["num_attention_heads"])
    hidden_size = int(text["hidden_size"])
    return {
        "architecture": config.get("architectures", [None])[0],
        "layers": int(text["num_hidden_layers"]),
        "hidden_size": hidden_size,
        "query_heads": query_heads,
        "kv_heads": int(text.get("num_key_value_heads", query_heads)),
        "head_dim": int(text.get("head_dim") or hidden_size // query_heads),
        "checkpoint_quantization": config.get("quantization"),
        "runtime_quantization": _quantization_manifest(spec),
    }


def validate(slug: str) -> dict[str, Any]:
    spec = SPECS[slug]
    info = HfApi().model_info(spec["base_model"], revision=spec["revision"])
    config_path = hf_hub_download(
        spec["base_model"], "config.json", revision=spec["revision"]
    )
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    expected = _expected(spec)
    observed = _observed(config, spec)
    checks = {
        key: observed[key] == expected[key]
        for key in (
            "architecture",
            "layers",
            "hidden_size",
            "query_heads",
            "kv_heads",
            "head_dim",
        )
    }
    checkpoint_quantization = observed["checkpoint_quantization"]
    if spec.get("engine", "mlx") == "mlx":
        checks["quantization_bits"] = (
            isinstance(checkpoint_quantization, dict)
            and checkpoint_quantization.get("bits") == expected["quantization"].get("bits")
        )
        checks["quantization_group_size"] = (
            isinstance(checkpoint_quantization, dict)
            and checkpoint_quantization.get("group_size")
            == expected["quantization"].get("group_size")
        )
    else:
        # bitsandbytes quantizes the immutable full-precision checkpoint at load time.
        checks["runtime_quantization_declared"] = (
            expected["quantization"].get("bits") == 8
            and expected["quantization"].get("runtime") == "bitsandbytes/PyTorch"
        )
    result = {
        "schema_version": 1,
        "status": "STRUCTURAL_VALIDATED" if all(checks.values()) else "FAILED",
        "claim_scope": "model config and structural adapter geometry only",
        "model_id": spec["base_model"],
        "requested_revision": spec["revision"],
        "resolved_revision": info.sha,
        "engine": spec.get("engine", "mlx"),
        "expected": expected,
        "observed": observed,
        "checks": checks,
        "runtime": {"host": platform.node(), "platform": platform.platform()},
        "date": dt.date.today().isoformat(),
        "limitations": [
            "Model weights were not loaded by this structural check.",
            "End-task quality, native-memory parity, latency, and learned routing remain NOT_MEASURED.",
            "Evidence from another quantization is not transferred to this identity.",
        ],
    }
    if result["status"] != "STRUCTURAL_VALIDATED":
        failed = ", ".join(key for key, passed in checks.items() if not passed)
        raise ValueError(f"{slug} structural validation failed: {failed}")
    target = QUANTIZED_RESULTS / slug / "structural_validation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    quantized = tuple(
        slug
        for slug, spec in SPECS.items()
        if spec.get("routing_artifact") is False and spec.get("quantization") in {"6bit", "8bit", "bnb-8bit"}
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=(*quantized, "all"), default="all")
    args = parser.parse_args()
    selected = quantized if args.model == "all" else (args.model,)
    for slug in selected:
        result = validate(slug)
        print(f"{slug}: {result['status']} @ {result['resolved_revision']}")


if __name__ == "__main__":
    main()
