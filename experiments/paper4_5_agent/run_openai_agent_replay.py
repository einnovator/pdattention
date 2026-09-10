"""Replay frozen agent turns through an OpenAI-compatible chat endpoint.

The runner is intentionally engine-neutral.  Run it once against a server with
prefix caching enabled and once against the same model/server configuration
with caching disabled, then pass the first artifact through ``--reference``.
This separates an engine's cache semantics from autonomous-agent randomness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any, Mapping


def _post(url: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(dict(payload)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1200) as response:
        return json.load(response)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _assistant_prefixes(trajectory: Mapping[str, Any], turns: int) -> list[list[dict[str, str]]]:
    messages = trajectory["messages"]
    indexes = [
        index for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ][:turns]
    return [
        [
            {"role": str(message["role"]), "content": str(message.get("content", ""))}
            for message in messages[:index]
        ]
        for index in indexes
    ]


def _choice_row(turn: int, raw: Mapping[str, Any]) -> dict[str, Any]:
    choice = raw["choices"][0]
    content = str(choice.get("message", {}).get("content") or "")
    logprobs = choice.get("logprobs")
    token_rows = logprobs.get("content", []) if isinstance(logprobs, Mapping) else []
    tokens = [str(row.get("token", "")) for row in token_rows]
    token_ids = [row.get("token_id") for row in token_rows]
    return {
        "turn": turn,
        "content": content,
        "content_sha256": _digest(content),
        "tokens": tokens,
        "token_ids": token_ids,
        "finish_reason": choice.get("finish_reason"),
        "usage": raw.get("usage"),
        "logprobs": token_rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    prefixes = _assistant_prefixes(trajectory, args.turns)
    rows = []
    endpoint = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    for turn, messages in enumerate(prefixes, start=1):
        raw = _post(endpoint, {
            "model": args.model,
            "messages": messages,
            "temperature": 0,
            # Do not inherit a model repository's generation_config.  The
            # cache comparison must vary only cache state, not sampling
            # truncation or repetition penalties.
            "top_p": 1,
            "top_k": -1,
            "min_p": 0,
            "repetition_penalty": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "logprobs": True,
            "top_logprobs": args.top_logprobs,
        })
        rows.append(_choice_row(turn, raw))

    result: dict[str, Any] = {
        "schema_version": 1,
        "probe": "openai_compatible_frozen_agent_replay",
        "engine": args.engine,
        "cache_mode": args.cache_mode,
        "base_url": args.base_url,
        "model": args.model,
        "trajectory": str(args.trajectory),
        "seed": args.seed,
        "temperature": 0,
        "requested_turns": args.turns,
        "completed_turns": len(rows),
        "rows": rows,
    }
    if args.reference is not None:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        reference_by_turn = {int(row["turn"]): row for row in reference["rows"]}
        comparisons = []
        for row in rows:
            other = reference_by_turn.get(int(row["turn"]))
            if other is None:
                continue
            left_ids = other.get("token_ids", [])
            right_ids = row.get("token_ids", [])
            ids_available = bool(left_ids) and all(value is not None for value in left_ids + right_ids)
            exact = left_ids == right_ids if ids_available else other["content"] == row["content"]
            comparisons.append({
                "turn": row["turn"],
                "exact": exact,
                "comparison_basis": "token_ids" if ids_available else "content",
                "reference_sha256": other["content_sha256"],
                "candidate_sha256": row["content_sha256"],
            })
        result["reference"] = str(args.reference)
        result["comparisons"] = comparisons
        result["exact_turns"] = sum(int(row["exact"]) for row in comparisons)
        result["all_exact"] = bool(comparisons) and all(row["exact"] for row in comparisons)
        result["first_divergent_turn"] = next(
            (row["turn"] for row in comparisons if not row["exact"]), None
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--cache-mode", choices=("on", "off"), required=True)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-logprobs", type=int, default=5)
    args = parser.parse_args()
    result = run(args)
    summary = {key: result[key] for key in (
        "engine", "cache_mode", "completed_turns", "exact_turns", "all_exact",
        "first_divergent_turn",
    ) if key in result}
    print(json.dumps(summary, indent=2))
    if args.reference is not None and not result.get("all_exact", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
