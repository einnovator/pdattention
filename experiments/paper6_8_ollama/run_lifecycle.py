"""Measure Ollama load, keep-alive reuse, model switch, and unload behavior."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from pra_hf.deployment import PRAWireRequest, PRAWireResource
from pra_ollama import OllamaEngineAdapter


def request(model: str, request_id: str) -> PRAWireRequest:
    return PRAWireRequest(
        request_id=request_id,
        model=model,
        messages=({"role": "user", "content": "What is the access code? Answer briefly."},),
        resources=(
            PRAWireResource(
                resource_id="access-code",
                uri="memory://access-code",
                text="The access code is amber-17.",
                version="v1",
            ),
        ),
        max_new_tokens=12,
    )


def timed(adapter: OllamaEngineAdapter, model: str, label: str) -> dict[str, object]:
    started = time.perf_counter()
    result = adapter.generate(request(model, label))
    elapsed = time.perf_counter() - started
    return {
        "label": label,
        "model": model,
        "elapsed_ms": elapsed * 1000,
        "text": result.text,
        "load_ms": int(result.raw.get("load_duration", 0)) / 1e6,
        "prompt_eval_ms": int(result.raw.get("prompt_eval_duration", 0)) / 1e6,
        "decode_ms": int(result.raw.get("eval_duration", 0)) / 1e6,
        "prompt_tokens": int(result.raw.get("prompt_eval_count", 0)),
        "completion_tokens": int(result.raw.get("eval_count", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen2.5:0.5b")
    parser.add_argument("--switch-model", default="smollm2:135m")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/papers/shared/results/paper6_8_ollama/lifecycle.json"),
    )
    args = parser.parse_args()
    adapter = OllamaEngineAdapter(args.base_url, keep_alive="10m")
    adapter.unload(args.model)
    rows = [timed(adapter, args.model, "cold_after_unload")]
    rows.extend(timed(adapter, args.model, f"warm_{index}") for index in range(1, 6))
    rows.append(timed(adapter, args.switch_model, "model_switch"))
    adapter.unload(args.switch_model)
    rows.append(timed(adapter, args.model, "reload_after_switch"))
    endpoint = adapter.inspect_endpoint()
    payload = {
        "schema_version": "1.0",
        "benchmark": "paper6_8_ollama_lifecycle_v1",
        "evidence_tier": "LIVE_ENGINE",
        "ollama_version": endpoint.version,
        "platform": platform.platform(),
        "integration_level": "E0_SELECTED_TEXT",
        "models": [args.model, args.switch_model],
        "rows": rows,
        "native_backend_status": "NOT_NEGOTIATED_AUTO_FALLBACK_E0",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
