"""Long same-state equivalence gate for llama.cpp live-prefix attachment."""

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
    return list(map(int, post(
        base_url, "/tokenize", {"content": text, "add_special": True},
    )["tokens"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18083")
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=4)
    args = parser.parse_args()

    source_slot = 0
    request_slot = 1
    for slot in (source_slot, request_slot):
        request = urllib.request.Request(
            f"{args.base_url.rstrip('/')}/pra/resources/{slot}", method="DELETE",
        )
        with urllib.request.urlopen(request, timeout=120):
            pass

    source = tokenize(args.base_url, "System: retain state exactly. Key=ALPHA.\n")
    post(args.base_url, "/completion", {
        "prompt": source,
        "id_slot": source_slot,
        "n_predict": 0,
        "cache_prompt": True,
        "pra_pin_resource": True,
        "temperature": 0,
    })
    source_tokens = len(source)
    bridge: list[int] = []
    rows: list[dict[str, Any]] = []

    for turn in range(1, args.turns + 1):
        query = tokenize(
            args.base_url,
            f"Turn {turn}: return ALPHA and the turn number, briefly.\n",
        )
        if query and query[0] == source[0]:
            del query[0]
        suffix = bridge + query
        common = {
            "prompt": suffix,
            "id_slot": request_slot,
            "pra_source_slot": source_slot,
            "pra_source_prefix_tokens": source_tokens,
            "pra_selected_ranges": [{
                "record_id": f"full-before-turn-{turn}",
                "parent_record_id": f"history-before-turn-{turn}",
                "causal_group_id": f"turn:{turn}",
                "start": 0,
                "end": source_tokens,
            }],
            "n_predict": args.max_tokens,
            "cache_prompt": True,
            "temperature": 0,
            "seed": 123,
            "return_tokens": True,
        }
        control = post(args.base_url, "/completion", {
            **common, "pra_commit_to_source": False,
        })
        treatment = post(args.base_url, "/completion", {
            **common, "pra_commit_to_source": True,
        })
        exact = control["tokens"] == treatment["tokens"]
        pra = dict(treatment.get("pra") or {})
        rows.append({
            "turn": turn,
            "exact": exact,
            "control_tokens": control["tokens"],
            "treatment_tokens": treatment["tokens"],
            "source_tokens": pra.get("source_tokens"),
            "selected_kv_tokens": pra.get("selected_kv_tokens"),
            "selected_text_reencoded_tokens": pra.get(
                "selected_text_reencoded_tokens"
            ),
            "committed_tokens": pra.get("committed_tokens"),
            "commit_succeeded": pra.get("commit_succeeded"),
        })
        if not exact:
            break
        generated = list(map(int, treatment["tokens"]))
        source_tokens += len(suffix) + max(0, len(generated) - 1)
        bridge = generated[-1:]

    result = {
        "schema_version": "1.0",
        "purpose": "long same-state correctness gate, not a speed result",
        "turns_requested": args.turns,
        "turns_completed": len(rows),
        "exact_turns": sum(int(row["exact"]) for row in rows),
        "all_exact": len(rows) == args.turns and all(row["exact"] for row in rows),
        "zero_selected_text_reencoding": all(
            row["selected_text_reencoded_tokens"] == 0 for row in rows
        ),
        "rows": rows,
    }
    print(json.dumps(result, indent=2))
    if not result["all_exact"] or not result["zero_selected_text_reencoding"]:
        raise SystemExit("long live-prefix equivalence gate failed")


if __name__ == "__main__":
    main()
