"""Qualify llama.cpp live-prefix commit/reattach against the same K/V state."""

from __future__ import annotations

import argparse
import json
import urllib.request
from typing import Any


def post(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def tokenize(base_url: str, text: str) -> list[int]:
    return list(
        map(
            int,
            post(base_url, "/tokenize", {"content": text, "add_special": True})[
                "tokens"
            ],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18083")
    parser.add_argument("--source-slot", type=int, default=0)
    parser.add_argument("--live-slot", type=int, default=1)
    parser.add_argument("--fresh-control-slot", type=int, default=2)
    parser.add_argument("--reattach-slot", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8)
    args = parser.parse_args()

    source = tokenize(
        args.base_url, "System facts:\nalpha=17\nbeta=23\nEnd facts.\n"
    )
    query1 = tokenize(
        args.base_url, "What is alpha plus beta? Answer with only the integer:"
    )
    query2 = tokenize(args.base_url, "Repeat that integer with no explanation:")
    for values in (query1, query2):
        if values and values[0] == source[0]:
            del values[0]

    def completion(prompt: list[int], slot: int, **extra: Any) -> dict[str, Any]:
        return post(
            args.base_url,
            "/completion",
            {
                "prompt": prompt,
                "id_slot": slot,
                "n_predict": args.max_tokens,
                "cache_prompt": True,
                "temperature": 0,
                "seed": 123,
                "return_tokens": True,
                **extra,
            },
        )

    post(
        args.base_url,
        "/completion",
        {
            "prompt": source,
            "id_slot": args.source_slot,
            "n_predict": 0,
            "cache_prompt": True,
            "pra_pin_resource": True,
            "temperature": 0,
        },
    )
    source_range = [
        {
            "record_id": "task",
            "parent_record_id": "task",
            "causal_group_id": "task",
            "start": 0,
            "end": len(source),
        }
    ]

    # The fresh arm is useful only to expose numerical reconstruction effects.
    fresh1 = completion(source + query1, args.fresh_control_slot, cache_prompt=False)
    live1 = completion(
        query1,
        args.live_slot,
        pra_source_slot=args.source_slot,
        pra_selected_ranges=source_range,
        pra_commit_to_source=True,
    )
    generated1 = list(map(int, live1["tokens"]))
    source_after_turn1 = len(source) + len(query1) + max(0, len(generated1) - 1)
    suffix2 = generated1[-1:] + query2
    full2 = source + query1 + generated1 + query2

    # The live control continues the exact state that was committed. The
    # treatment reattaches that committed state to a clean request sequence.
    live_control2 = completion(full2, args.live_slot, cache_prompt=True)
    reattached2 = completion(
        suffix2,
        args.reattach_slot,
        pra_source_slot=args.source_slot,
        pra_selected_ranges=[
            {
                "record_id": "history-through-turn-1",
                "parent_record_id": "history",
                "causal_group_id": "turn:1",
                "start": 0,
                "end": source_after_turn1,
            }
        ],
        pra_commit_to_source=True,
    )
    fresh2 = completion(full2, args.fresh_control_slot, cache_prompt=True)

    result = {
        "schema_version": "1.0",
        "purpose": "correctness_smoke_not_performance",
        "turn1": {
            "fresh_tokens": fresh1["tokens"],
            "live_tokens": live1["tokens"],
            "fresh_matches_live": fresh1["tokens"] == live1["tokens"],
            "pra": live1.get("pra"),
        },
        "turn2": {
            "live_control_tokens": live_control2["tokens"],
            "reattached_tokens": reattached2["tokens"],
            "same_state_exact": live_control2["tokens"] == reattached2["tokens"],
            "fresh_tokens": fresh2["tokens"],
            "fresh_matches_live": fresh2["tokens"] == live_control2["tokens"],
            "pra": reattached2.get("pra"),
        },
    }
    print(json.dumps(result, indent=2))
    if not result["turn2"]["same_state_exact"]:
        raise SystemExit("same-state live-prefix qualification failed")


if __name__ == "__main__":
    main()
