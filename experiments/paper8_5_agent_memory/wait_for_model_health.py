"""Wait for a pinned model endpoint and emit a fail-closed health receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping
import urllib.request

from .model_identity import fetch_ollama_model_identity


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_chat_completion(
    payload: Mapping[str, Any], *, expected_text: str | None = None
) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("health completion must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ValueError("health completion choice must be an object")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("health completion is missing an assistant message")
    content = str(message.get("content") or "").strip()
    if not content:
        raise ValueError("health completion returned empty assistant content")
    if expected_text is not None and content != expected_text:
        raise ValueError(
            f"health completion mismatch: expected {expected_text!r}, got {content!r}"
        )
    return content


def fetch_completion(
    url: str,
    *,
    model: str,
    expected_text: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    body = json.dumps({
        "model": model,
        "messages": [{
            "role": "user",
            "content": f"Reply with exactly: {expected_text}",
        }],
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "max_tokens": 16,
        "stream": False,
    }).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = validate_chat_completion(payload, expected_text=expected_text)
    return {
        "observed_at": _now(),
        "latency_seconds": time.monotonic() - started,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "finish_reason": payload["choices"][0].get("finish_reason"),
    }


def wait_for_health(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    consecutive: list[dict[str, Any]] = []
    failure_count = 0
    last_failure: dict[str, Any] | None = None
    while True:
        if args.max_wait_seconds and time.monotonic() - started > args.max_wait_seconds:
            raise TimeoutError("model endpoint did not pass the health gate in time")
        try:
            probe_started = time.monotonic()
            identity = fetch_ollama_model_identity(
                args.tags_url,
                expected_model=args.model,
                expected_revision=args.revision,
                timeout_seconds=args.request_timeout_seconds,
            )
            consecutive.append({
                "observed_at": _now(),
                "latency_seconds": time.monotonic() - probe_started,
                "model": identity["model"],
                "revision": identity["revision"],
            })
            if len(consecutive) < args.consecutive_probes:
                time.sleep(args.retry_seconds)
                continue
            completion = fetch_completion(
                args.completion_url,
                model=args.model,
                expected_text=args.expected_text,
                timeout_seconds=args.completion_timeout_seconds,
            )
            return {
                "schema_version": 1,
                "status": "qualified",
                "qualified_at": _now(),
                "model": args.model,
                "revision": args.revision,
                "consecutive_probe_count": len(consecutive),
                "probes": consecutive,
                "deterministic_completion": completion,
                "failure_count_before_qualification": failure_count,
                "last_failure": last_failure,
            }
        except Exception as error:  # endpoint transport and identity failures
            failure_count += 1
            last_failure = {
                "observed_at": _now(),
                "error_type": type(error).__name__,
                "error_detail": str(error),
            }
            consecutive = []
            time.sleep(args.retry_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags-url", required=True)
    parser.add_argument("--completion-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--consecutive-probes", type=int, default=3)
    parser.add_argument("--retry-seconds", type=float, default=30)
    parser.add_argument("--request-timeout-seconds", type=float, default=10)
    parser.add_argument("--completion-timeout-seconds", type=float, default=120)
    parser.add_argument("--max-wait-seconds", type=float, default=0)
    parser.add_argument("--expected-text", default="PRA_HEALTH_OK")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evidence = wait_for_health(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
