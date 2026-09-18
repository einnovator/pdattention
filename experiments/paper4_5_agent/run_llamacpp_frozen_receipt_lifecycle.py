"""Run one receipt-aware frozen Paper 8.5 request on patched llama.cpp."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from transformers import AutoTokenizer

from .frozen_agent_plan import frozen_live_kv_geometry, load_frozen_agent_decisions


def _request(
    base_url: str,
    path: str,
    payload: Mapping[str, object] | None = None,
    *,
    method: str | None = None,
    timeout: float = 900.0,
) -> dict[str, Any] | list[dict[str, Any]]:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method=method or ("GET" if payload is None else "POST"),
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _erase(base_url: str, slot: int) -> None:
    _request(base_url, f"/pra/resources/{slot}", method="DELETE")


def _generate(
    base_url: str,
    *,
    source_slot: int,
    request_slot: int,
    geometry: object,
    max_tokens: int,
    seed: int,
) -> tuple[dict[str, Any], float]:
    ranges = [
        {
            "record_id": interval.record_id or f"range-{ordinal}",
            "parent_record_id": interval.record_id or f"range-{ordinal}",
            "causal_group_id": interval.causal_group_id or f"range-{ordinal}",
            "start": interval.start,
            "end": interval.end,
        }
        for ordinal, interval in enumerate(geometry.plan.intervals, 1)
    ]
    materialized = [
        {
            "record_id": span.record_id,
            "position_start": span.position_start,
            "token_ids": list(span.token_ids),
        }
        for span in geometry.materialized_history_spans
    ]
    started = time.perf_counter()
    result = _request(base_url, "/completion", {
        "prompt": list(geometry.wire_tail_ids),
        "id_slot": request_slot,
        "pra_source_slot": source_slot,
        "pra_source_prefix_tokens": len(geometry.source_ids),
        "pra_selected_ranges": ranges,
        "pra_materialized_history": materialized,
        "pra_commit_to_source": False,
        "n_predict": max_tokens,
        "cache_prompt": True,
        "temperature": 0,
        "seed": seed,
        "return_tokens": True,
    })
    return dict(result), time.perf_counter() - started


def run(args: argparse.Namespace) -> dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
        local_files_only=args.local_files_only,
    )
    decisions = load_frozen_agent_decisions(
        args.request_replay.expanduser().resolve(),
        args.selection_fixture.expanduser().resolve(),
    )
    if not 1 <= args.request_index <= len(decisions):
        raise ValueError("--request-index is outside the frozen replay")
    decision = decisions[args.request_index - 1]
    geometry = frozen_live_kv_geometry(tokenizer, decision)
    materialized_tokens = sum(
        len(span.token_ids) for span in geometry.materialized_history_spans
    )
    capabilities = _request(args.base_url, "/pra/capabilities")
    if not capabilities.get("positioned_materialized_history"):
        raise RuntimeError("llama.cpp lacks positioned materialized-history support")

    for slot in (args.source_slot, args.request_slot):
        _erase(args.base_url, slot)
    started = time.perf_counter()
    prime = dict(_request(args.base_url, "/completion", {
        "prompt": list(geometry.source_ids),
        "id_slot": args.source_slot,
        "n_predict": 0,
        "cache_prompt": True,
        "temperature": 0,
        "seed": args.seed,
        "pra_pin_resource": True,
    }))
    prime_seconds = time.perf_counter() - started

    first, first_seconds = _generate(
        args.base_url,
        source_slot=args.source_slot,
        request_slot=args.request_slot,
        geometry=geometry,
        max_tokens=args.continuation_tokens,
        seed=args.seed,
    )
    _erase(args.base_url, args.request_slot)
    second, second_seconds = _generate(
        args.base_url,
        source_slot=args.source_slot,
        request_slot=args.request_slot,
        geometry=geometry,
        max_tokens=args.continuation_tokens,
        seed=args.seed,
    )
    first_pra = dict(first.get("pra") or {})
    second_pra = dict(second.get("pra") or {})
    first_tokens = [int(token) for token in first.get("tokens", ())]
    second_tokens = [int(token) for token in second.get("tokens", ())]
    _erase(args.base_url, args.request_slot)
    _erase(args.base_url, args.source_slot)
    slots = _request(args.base_url, "/slots")
    relevant_slots = [
        row for row in slots
        if int(row.get("id", -1)) in {args.source_slot, args.request_slot}
    ]
    lifecycle_empty = all(
        int(row.get("n_past", row.get("n_prompt_tokens", 0)) or 0) == 0
        and not bool(row.get("pra_resource_pinned", False))
        for row in relevant_slots
    )

    source_tokens = len(geometry.source_ids)
    selected_tokens = geometry.plan.selected_tokens
    visible_tokens = selected_tokens + materialized_tokens + len(geometry.wire_tail_ids)
    full_visible_tokens = len(geometry.prompt_ids)
    checks = {
        "positioned_receipts_supported": bool(
            capabilities.get("positioned_materialized_history")
        ),
        "selected_history_not_reencoded": all(
            int(row.get("selected_text_reencoded_tokens", -1)) == 0
            for row in (first_pra, second_pra)
        ),
        "selected_original_kv_not_copied": all(
            row.get("physical_kv_copy") is False
            for row in (first_pra, second_pra)
        ),
        "selected_token_accounting": all(
            int(row.get("selected_kv_tokens", -1)) == selected_tokens
            for row in (first_pra, second_pra)
        ),
        "receipt_token_accounting": all(
            int(row.get("materialized_history_encoded_tokens", -1))
            == materialized_tokens
            for row in (first_pra, second_pra)
        ),
        "repeat_token_exact": first_tokens == second_tokens,
        "source_positions_preserved": all(
            row.get("source_positions_preserved") is True
            for row in (first_pra, second_pra)
        ),
        "lifecycle_empty": lifecycle_empty,
    }
    payload: dict[str, object] = {
        "schema_version": "paper4.5.llamacpp-frozen-receipt-lifecycle.v1",
        "qualified": all(checks.values()),
        "qualification_blockers": [name for name, passed in checks.items() if not passed],
        "checks": checks,
        "experiment_revision": args.experiment_revision,
        "engine": "llama.cpp-metal",
        "engine_revision": capabilities.get("build_commit"),
        "python_version": platform.python_version(),
        "model": args.model_label,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "request_index": decision.request_index,
        "request_input_sha256": decision.request_input_sha256,
        "source_policy": decision.source_policy,
        "logical_source_tokens": source_tokens,
        "selected_logical_kv_tokens": selected_tokens,
        "materialized_history_tokens": materialized_tokens,
        "wire_suffix_tokens": len(geometry.wire_tail_ids),
        "total_visible_tokens": visible_tokens,
        "full_visible_tokens": full_visible_tokens,
        "history_saving_after_receipts_fraction": (
            source_tokens - selected_tokens - materialized_tokens
        ) / source_tokens,
        "total_visible_saving_fraction": (
            full_visible_tokens - visible_tokens
        ) / full_visible_tokens,
        "resident_original_kv_omission_fraction": (
            source_tokens - selected_tokens
        ) / source_tokens,
        "selected_history_reencoded_tokens": 0,
        "selected_history_kv_copy_bytes": 0,
        "receipt_encoded_tokens_per_request": materialized_tokens,
        "final_token_ids_a": first_tokens,
        "final_token_ids_b": second_tokens,
        "prime_seconds": prime_seconds,
        "final_seconds_a": first_seconds,
        "final_seconds_b": second_seconds,
        "prime_timings": prime.get("timings"),
        "final_timings_a": first.get("timings"),
        "final_timings_b": second.get("timings"),
        "capabilities": capabilities,
        "final_slots": relevant_slots,
        "fixture_hashes": {
            "request_replay_sha256": hashlib.sha256(
                args.request_replay.expanduser().resolve().read_bytes()
            ).hexdigest(),
            "selection_fixture_sha256": hashlib.sha256(
                args.selection_fixture.expanduser().resolve().read_bytes()
            ).hexdigest(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18082")
    parser.add_argument("--request-replay", type=Path, required=True)
    parser.add_argument("--selection-fixture", type=Path, required=True)
    parser.add_argument("--request-index", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--model-label", default="Qwen3-14B-Q4_K_M")
    parser.add_argument("--experiment-revision", default="uncommitted")
    parser.add_argument("--source-slot", type=int, default=0)
    parser.add_argument("--request-slot", type=int, default=1)
    parser.add_argument("--continuation-tokens", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "qualified": result["qualified"],
        "qualification_blockers": result["qualification_blockers"],
        "total_visible_saving_fraction": result["total_visible_saving_fraction"],
        "final_token_ids": result["final_token_ids_a"],
        "output": str(args.output),
    }, indent=2))
    raise SystemExit(0 if result["qualified"] else 1)


if __name__ == "__main__":
    main()
