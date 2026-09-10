"""Qualify in-place llama.cpp updates of a pinned selected-context prefix."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any


def _post(base_url: str, path: str, payload: dict[str, Any]) -> Any:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def _erase(base_url: str, slot: int) -> None:
    _post(base_url, f"/slots/{slot}?action=erase", {})


def _prefill(
    base_url: str, *, slot: int, prompt: str, cache_prompt: bool,
) -> dict[str, Any]:
    return dict(_post(base_url, "/completion", {
        "prompt": prompt,
        "id_slot": slot,
        "n_predict": 0,
        "cache_prompt": cache_prompt,
        "temperature": 0,
        "pra_pin_resource": True,
    }))


def _generate(
    base_url: str, *, resource_slot: int, request_slot: int,
    query: str, max_tokens: int, seed: int,
) -> dict[str, Any]:
    return dict(_post(base_url, "/completion", {
        "prompt": query,
        "id_slot": request_slot,
        "pra_resource_slot": resource_slot,
        "n_predict": max_tokens,
        "cache_prompt": False,
        "temperature": 0,
        "seed": seed,
        "return_tokens": True,
    }))


def run(args: argparse.Namespace) -> dict[str, Any]:
    stable = " ".join(f"stable-{index}" for index in range(args.prefix_words))
    resource_a = stable + " discarded-tail"
    resource_b = stable + " selected-replacement-tail"

    for slot in (args.resource_slot, args.request_slot):
        _erase(args.base_url, slot)
    _prefill(
        args.base_url, slot=args.resource_slot,
        prompt=resource_a, cache_prompt=True,
    )
    delta_prefill = _prefill(
        args.base_url, slot=args.resource_slot,
        prompt=resource_b, cache_prompt=True,
    )
    delta_result = _generate(
        args.base_url, resource_slot=args.resource_slot,
        request_slot=args.request_slot, query=args.query,
        max_tokens=args.max_tokens, seed=args.seed,
    )

    for slot in (args.resource_slot, args.request_slot):
        _erase(args.base_url, slot)
    cold_prefill = _prefill(
        args.base_url, slot=args.resource_slot,
        prompt=resource_b, cache_prompt=False,
    )
    cold_result = _generate(
        args.base_url, resource_slot=args.resource_slot,
        request_slot=args.request_slot, query=args.query,
        max_tokens=args.max_tokens, seed=args.seed,
    )
    delta_tokens = list(delta_result.get("tokens", ()))
    cold_tokens = list(cold_result.get("tokens", ()))
    result = {
        "schema_version": 1,
        "probe": "llamacpp_pinned_resource_prefix_delta",
        "base_url": args.base_url,
        "prefix_words": args.prefix_words,
        "seed": args.seed,
        "delta_cache_n": (delta_prefill.get("timings") or {}).get("cache_n"),
        "delta_prompt_n": (delta_prefill.get("timings") or {}).get("prompt_n"),
        "cold_cache_n": (cold_prefill.get("timings") or {}).get("cache_n"),
        "cold_prompt_n": (cold_prefill.get("timings") or {}).get("prompt_n"),
        "delta_completion_tokens": len(delta_tokens),
        "cold_completion_tokens": len(cold_tokens),
        "completion_tokens_exact": delta_tokens == cold_tokens,
        "delta_physical_kv_copy": (delta_result.get("pra") or {}).get(
            "physical_kv_copy"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18082")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resource-slot", type=int, default=0)
    parser.add_argument("--request-slot", type=int, default=1)
    parser.add_argument("--prefix-words", type=int, default=512)
    parser.add_argument("--query", default=" Summarize the selected replacement briefly.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
    passed = (
        result["completion_tokens_exact"]
        and (result["delta_cache_n"] or 0) > 0
        and (result["cold_cache_n"] or 0) == 0
        and (result["delta_prompt_n"] or 0) < (result["cold_prompt_n"] or 0)
        and result["delta_physical_kv_copy"] is False
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
