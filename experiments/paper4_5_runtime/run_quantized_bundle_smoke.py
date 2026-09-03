"""Load an exact quantized checkpoint and run a bounded generation smoke test."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper4_5_runtime.build_hf_catalog_bundles import QUANTIZED_RESULTS, SPECS


def _hardware() -> str:
    commands = (
        ["system_profiler", "SPHardwareDataType"],
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    )
    for command in commands:
        try:
            output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            continue
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if command[0] == "system_profiler":
            lines = [line for line in lines if line.startswith(("Model Name:", "Chip:", "Memory:"))]
        if lines:
            return "; ".join(lines)
    return platform.platform()


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _mlx_smoke(spec: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import generate, load

    started = time.perf_counter()
    model, tokenizer = load(spec["base_model"], revision=spec["revision"])
    load_seconds = time.perf_counter() - started
    layers = model.model.layers
    attention = layers[0].self_attn
    checks = {
        "layer_count": len(layers) == spec["layers"],
        "q_projection": hasattr(attention, "q_proj"),
        "k_projection": hasattr(attention, "k_proj"),
        "v_projection": hasattr(attention, "v_proj"),
        "o_projection": hasattr(attention, "o_proj"),
    }
    prompt = "Reply with exactly one word: ready"
    started = time.perf_counter()
    output = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    generation_seconds = time.perf_counter() - started
    memory = {}
    for name in ("get_active_memory", "get_cache_memory", "get_peak_memory"):
        operation = getattr(mx, name, None)
        if operation is not None:
            memory[name.removeprefix("get_") + "_bytes"] = int(operation())
    return {
        "checks": checks,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "generated_characters": len(output),
        "max_new_tokens": max_tokens,
        "memory": memory,
        "versions": {"mlx": _version("mlx"), "mlx_lm": _version("mlx-lm")},
    }


def _hf_smoke(spec: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(load_in_8bit=True)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(spec["base_model"], revision=spec["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        spec["base_model"],
        revision=spec["revision"],
        quantization_config=quantization,
        device_map="auto",
    )
    load_seconds = time.perf_counter() - started
    layers = model.model.layers
    attention = layers[0].self_attn
    checks = {
        "layer_count": len(layers) == spec["layers"],
        "q_projection": hasattr(attention, "q_proj"),
        "k_projection": hasattr(attention, "k_proj"),
        "v_projection": hasattr(attention, "v_proj"),
        "o_projection": hasattr(attention, "o_proj"),
        "weights_are_8bit": bool(getattr(model, "is_loaded_in_8bit", False)),
    }
    inputs = tokenizer("Reply with exactly one word: ready", return_tensors="pt").to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - started
    return {
        "checks": checks,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "generated_tokens": int(output.shape[-1] - inputs.input_ids.shape[-1]),
        "max_new_tokens": max_tokens,
        "memory": {
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else None
        },
        "versions": {
            "torch": torch.__version__,
            "transformers": _version("transformers"),
            "bitsandbytes": _version("bitsandbytes"),
        },
    }


def run(slug: str, max_tokens: int) -> dict[str, Any]:
    spec = SPECS[slug]
    engine = spec.get("engine", "mlx")
    measurement = (
        _hf_smoke(spec, max_tokens) if engine == "hf" else _mlx_smoke(spec, max_tokens)
    )
    checks = measurement.pop("checks")
    result = {
        "schema_version": 1,
        "status": "RUNTIME_SMOKE_VALIDATED" if all(checks.values()) else "FAILED",
        "claim_scope": "exact checkpoint load, adapter projection discovery, and bounded generation",
        "model_id": spec["base_model"],
        "model_revision": spec["revision"],
        "quantization": spec["quantization"],
        "engine": engine,
        "checks": checks,
        **measurement,
        "runtime": {
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "hardware": _hardware(),
        },
        "date": dt.date.today().isoformat(),
        "limitations": [
            "The fixed smoke prompt is not an end-task quality evaluation.",
            "Native PRA K/V parity, learned routing, TTFT, ITL, and throughput remain NOT_MEASURED.",
        ],
    }
    target = QUANTIZED_RESULTS / slug / "runtime_smoke.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if result["status"] != "RUNTIME_SMOKE_VALIDATED":
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"{slug} runtime smoke failed: {failed}")
    return result


def main() -> None:
    candidates = tuple(
        slug
        for slug, spec in SPECS.items()
        if spec.get("routing_artifact") is False
        and spec.get("quantization") in {"6bit", "8bit", "bnb-8bit"}
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=candidates, required=True)
    parser.add_argument("--max-tokens", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run(args.model, args.max_tokens), indent=2))


if __name__ == "__main__":
    main()
