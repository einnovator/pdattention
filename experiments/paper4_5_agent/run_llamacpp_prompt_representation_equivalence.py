"""Compare llama.cpp string prompts with pre-tokenized live-prefix prompts.

The Easy-50 plain path submits the rendered chat prompt as text, while the
100-percent PRA path submits the exact token IDs so it can preserve the live
slot.  This probe replays one frozen agent trajectory through both forms on two
clean slots of the same server and fails the qualification gate on any sampled
token mismatch.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any


def _post(base_url: str, path: str, payload: dict[str, Any]) -> Any:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def run(args: argparse.Namespace) -> dict[str, Any]:
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    messages = trajectory["messages"]
    assistant_indexes = [
        index for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ][: args.turns]
    for slot in (args.string_slot, args.token_slot):
        _post(args.base_url, f"/slots/{slot}?action=erase", {})

    rows = []
    for turn, assistant_index in enumerate(assistant_indexes, start=1):
        prefix = messages[:assistant_index]
        rendered = _post(args.base_url, "/apply-template", {
            "messages": prefix,
            "add_generation_prompt": True,
        })["prompt"]
        prompt_tokens = _post(args.base_url, "/tokenize", {
            "content": rendered,
            "add_special": True,
        })["tokens"]
        common = {
            "n_predict": args.max_tokens,
            "cache_prompt": True,
            "temperature": 0,
            "seed": args.seed,
            "return_tokens": True,
        }
        text_result = _post(args.base_url, "/completion", {
            **common, "id_slot": args.string_slot, "prompt": rendered,
        })
        token_result = _post(args.base_url, "/completion", {
            **common, "id_slot": args.token_slot, "prompt": prompt_tokens,
        })
        text_output = text_result.get("tokens", [])
        token_output = token_result.get("tokens", [])
        mismatch = next(
            (
                index for index, pair in enumerate(zip(text_output, token_output))
                if pair[0] != pair[1]
            ),
            None,
        )
        exact = text_output == token_output
        rows.append({
            "turn": turn,
            "trajectory_message_index": assistant_index,
            "prompt_tokens": len(prompt_tokens),
            "text_completion_tokens": len(text_output),
            "token_completion_tokens": len(token_output),
            "exact": exact,
            "first_mismatch": mismatch,
            "text_cache_tokens": (text_result.get("timings") or {}).get("cache_n"),
            "token_cache_tokens": (token_result.get("timings") or {}).get("cache_n"),
        })
        # Persist every completed pair: long 30B probes can be interrupted by
        # unrelated workstation pressure, and completed exact pairs remain
        # useful diagnostic evidence.
        checkpoint = {
            "schema_version": 1,
            "probe": "llamacpp_string_vs_pretokenized_live_prefix",
            "trajectory": str(args.trajectory),
            "base_url": args.base_url,
            "seed": args.seed,
            "requested_turns": args.turns,
            "completed_turns": len(rows),
            "exact_turns": sum(row["exact"] for row in rows),
            "all_exact": bool(rows) and all(row["exact"] for row in rows),
            "complete": len(rows) == len(assistant_indexes),
            "rows": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8",
        )
        if not exact and args.fail_fast:
            break

    result = {
        "schema_version": 1,
        "probe": "llamacpp_string_vs_pretokenized_live_prefix",
        "trajectory": str(args.trajectory),
        "base_url": args.base_url,
        "seed": args.seed,
        "requested_turns": args.turns,
        "completed_turns": len(rows),
        "exact_turns": sum(row["exact"] for row in rows),
        "all_exact": bool(rows) and all(row["exact"] for row in rows),
        "complete": len(rows) == len(assistant_indexes),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18082")
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--string-slot", type=int, default=0)
    parser.add_argument("--token-slot", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["all_exact"] else 1)


if __name__ == "__main__":
    main()
