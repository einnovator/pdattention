"""Reconstruct an audited Pi OpenAI request from its native session log.

Prospective campaigns capture the first genuinely selective logical request
directly.  This utility exists for older completed campaigns that captured only
their first continuation request.  It replays no model or selector: it projects
Pi's append-only native message events to the OpenAI wire schema and requires
the reconstructed message digest to equal the frozen proxy trace identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _text(parts: Any) -> str:
    if isinstance(parts, str):
        return parts
    if not isinstance(parts, list):
        return ""
    return "".join(
        str(part.get("text") or "")
        for part in parts
        if isinstance(part, Mapping) and part.get("type") == "text"
    )


def pi_message_to_openai(message: Mapping[str, Any]) -> dict[str, Any]:
    """Project one Pi session message without changing semantic content."""

    role = str(message.get("role") or "")
    parts = message.get("content")
    if role == "user":
        if not isinstance(parts, (str, list)):
            raise ValueError("Pi user message has unsupported content")
        return {"role": "user", "content": parts}
    if role == "toolResult":
        tool_call_id = str(message.get("toolCallId") or "")
        if not tool_call_id:
            raise ValueError("Pi tool result has no tool-call identity")
        return {
            "role": "tool",
            "content": _text(parts),
            "tool_call_id": tool_call_id,
        }
    if role != "assistant":
        raise ValueError(f"unsupported Pi native message role: {role!r}")

    tool_calls: list[dict[str, Any]] = []
    if isinstance(parts, list):
        for part in parts:
            if not isinstance(part, Mapping) or part.get("type") != "toolCall":
                continue
            tool_calls.append({
                "id": str(part.get("id") or ""),
                "type": "function",
                "function": {
                    "name": str(part.get("name") or ""),
                    # Pi/OpenAI preserve argument insertion order and emit
                    # Unicode directly. Both details are part of the frozen
                    # request identity even though engines later parse this
                    # string into a semantic mapping.
                    "arguments": json.dumps(
                        part.get("arguments") or {},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            })
    content = _text(parts)
    result: dict[str, Any] = {
        "role": "assistant",
        "content": content if content else None,
    }
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def reconstruct_request(
    template: Mapping[str, Any],
    native_events: Sequence[Mapping[str, Any]],
    *,
    native_request_index: int,
    expected_request_input_sha256: str,
) -> dict[str, Any]:
    if native_request_index <= 0:
        raise ValueError("native request index must be positive")
    template_messages = template.get("messages")
    if not isinstance(template_messages, list):
        raise ValueError("request template has no messages")
    preamble: list[dict[str, Any]] = []
    for message in template_messages:
        if not isinstance(message, Mapping):
            raise ValueError("request template message is not an object")
        if message.get("role") not in {"system", "developer"}:
            break
        preamble.append(dict(message))
    if not preamble:
        raise ValueError("request template has no provider preamble")

    native_messages = [
        event["message"]
        for event in native_events
        if event.get("type") == "message"
        and isinstance(event.get("message"), Mapping)
    ]
    seen = 0
    stop: int | None = None
    for index, message in enumerate(native_messages):
        if message.get("role") == "assistant":
            seen += 1
            if seen == native_request_index:
                stop = index
                break
    if stop is None:
        raise ValueError(
            "Pi native session has fewer assistant decisions than the requested "
            f"index: {seen} < {native_request_index}"
        )
    messages = [
        *preamble,
        *[pi_message_to_openai(message) for message in native_messages[:stop]],
    ]
    observed = _digest(messages)
    if observed != expected_request_input_sha256:
        raise ValueError(
            "reconstructed Pi request disagrees with the frozen proxy identity: "
            f"{observed} != {expected_request_input_sha256}"
        )
    result = dict(template)
    result["messages"] = messages
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-template", type=Path, required=True)
    parser.add_argument("--native-session", type=Path, required=True)
    parser.add_argument("--native-request-index", type=int, required=True)
    parser.add_argument("--expected-request-input-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    template = json.loads(args.request_template.read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in args.native_session.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    request = reconstruct_request(
        template,
        events,
        native_request_index=args.native_request_index,
        expected_request_input_sha256=args.expected_request_input_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(request, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "contract": "paper8.5-pi-native-request-reconstruction-v1",
        "native_request_index": args.native_request_index,
        "request_input_sha256": args.expected_request_input_sha256,
        "request_template_sha256": hashlib.sha256(
            args.request_template.read_bytes()
        ).hexdigest(),
        "native_session_sha256": hashlib.sha256(
            args.native_session.read_bytes()
        ).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
