"""Replay one frozen agent-history prefix against an OpenAI-compatible endpoint.

This is a diagnostic, not an autonomous task score.  It answers a narrower
question: given the exact messages visible before a previously recorded
assistant action, does the current backend reproduce that action?
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_prefix(path: Path, after_message_index: int) -> tuple[list[dict[str, str]], str]:
    trajectory = json.loads(path.read_text(encoding="utf-8"))
    messages = trajectory["messages"]
    if after_message_index < 0 or after_message_index + 1 >= len(messages):
        raise ValueError("after-message-index must precede a recorded assistant message")
    expected = messages[after_message_index + 1]
    if expected.get("role") != "assistant":
        raise ValueError("the message after the replay prefix is not an assistant message")
    prefix = [
        {"role": message["role"], "content": message["content"]}
        for message in messages[: after_message_index + 1]
    ]
    return prefix, expected["content"]


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--after-message-index", type=int, default=3)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    prefix, expected = _load_prefix(args.trajectory, args.after_message_index)
    payload = {
        "model": args.model,
        "messages": prefix,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    response = _post_json(args.endpoint.rstrip("/") + "/chat/completions", payload, args.timeout)
    actual = response["choices"][0]["message"]["content"]
    receipt = {
        "schema_version": 1,
        "study": "paper8_5_frozen_trajectory_prefix_replay",
        "trajectory": str(args.trajectory),
        "after_message_index_zero_based": args.after_message_index,
        "endpoint": args.endpoint,
        "model": args.model,
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "max_tokens": args.max_tokens,
        },
        "prefix_roles": [message["role"] for message in prefix],
        "prefix_content_sha256": [_sha256(message["content"]) for message in prefix],
        "expected_assistant_sha256": _sha256(expected),
        "actual_assistant_sha256": _sha256(actual),
        "assistant_exact_match": actual == expected,
        "expected_assistant_content": expected,
        "actual_assistant_content": actual,
        "usage": response.get("usage"),
        "observed_model": response.get("model"),
        "finish_reason": response["choices"][0].get("finish_reason"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in (
        "assistant_exact_match",
        "expected_assistant_sha256",
        "actual_assistant_sha256",
        "finish_reason",
    )}, indent=2))


if __name__ == "__main__":
    main()
