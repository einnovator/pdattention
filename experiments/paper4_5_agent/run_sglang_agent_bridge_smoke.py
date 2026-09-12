"""Run a two-turn resident-history smoke against the direct SGLang endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any


def _manifest(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "message_index": index,
            "role": message["role"],
            "content_sha256": hashlib.sha256(
                message["content"].encode("utf-8")
            ).hexdigest(),
        }
        for index, message in enumerate(messages)
    ]


def _read_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def _trace(response: dict[str, Any]) -> dict[str, Any]:
    rows = response.get("pra_trace") or ()
    matches = [row for row in rows if row.get("stage") == "native_attach"]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one native_attach trace, found {len(matches)}.")
    return dict(matches[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18121")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--session-id", default="sglang-live-kv-two-turn-smoke"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.base_url.rstrip("/").removesuffix("/v1")
    health = _read_json(root + "/health")
    digest = str(health.get("chat_template_digest") or "")
    if not digest:
        raise RuntimeError("Direct endpoint did not advertise a template digest.")
    system = {"role": "system", "content": "You are a concise coding agent."}
    task = {
        "role": "user",
        "content": (
            "This is a deterministic transport smoke. Reply with exactly OK and "
            "no other text."
        ),
    }
    first_logical = [system, task]
    common_pra = {
        "tenant_id": "paper4-5-smoke",
        "session_id": args.session_id,
        "resources": [],
        "allow_text_fallback": False,
        "metadata": {
            "history_projection": "live-agent-kv-v1",
            "target_retention_fraction": 1.0,
            "chat_template_digest": digest,
        },
    }
    first_pra = json.loads(json.dumps(common_pra))
    first_pra["metadata"].update({
        "logical_message_manifest": _manifest(first_logical),
        "mandatory_message_indices": [0, 1],
    })
    first = _post(root + "/v1/chat/completions", {
        "model": args.model,
        "messages": first_logical,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 64,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "pra": first_pra,
    })
    assistant = {
        "role": "assistant",
        "content": str(first["choices"][0]["message"]["content"]),
    }
    observation = {
        "role": "user",
        "content": (
            "Chunk ID: smoke\nProcess exited with code 0\n"
            "Final output:\nOK\n"
            "The command completed successfully without changing any files. "
            "This deterministic padding makes the completed observation cross "
            "the frozen wire-tail boundary. Reply with exactly DONE."
        ),
    }
    second_logical = [system, task, assistant, observation]
    second_pra = json.loads(json.dumps(common_pra))
    second_pra["metadata"].update({
        "logical_message_manifest": _manifest(second_logical),
        # The prior assistant is resident and intentionally absent from wire
        # messages; this proves ledger reconstruction rather than text replay.
        "mandatory_message_indices": [0, 1, 3],
    })
    second = _post(root + "/v1/chat/completions", {
        "model": args.model,
        "messages": [system, task, observation],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 64,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "pra": second_pra,
    })
    first_trace, second_trace = _trace(first), _trace(second)
    assertions = {
        "template_digest_stable": (
            first_trace.get("chat_template_digest")
            == second_trace.get("chat_template_digest") == digest
        ),
        "second_turn_zero_history_reencoding": (
            second_trace.get("selected_history_reencoded_tokens") == 0
        ),
        "selected_history_zero_copy": all(
            row.get("selected_history_kv_copy_bytes") == 0
            and row.get("native_attach_bytes", 0) == 0
            and row.get("physical_kv_copy_bytes") == 0
            for row in (first.get("pra") or {}, second.get("pra") or {})
        ),
        "host_to_device_zero": all(
            row.get("host_to_device_bytes") == 0
            for row in (first_trace, second_trace)
        ),
        "full_retention_exact": all(
            row.get("selected_kv_tokens") == row.get("source_tokens")
            for row in (first_trace, second_trace)
        ),
        "second_turn_appends_only": (
            0 < int(second_trace.get("new_history_encoded_tokens") or 0)
            < int(second_trace.get("source_tokens") or 0)
        ),
    }
    result = {
        "schema_version": "paper4.5.sglang-agent-bridge-smoke.v1",
        "health": health,
        "first": {
            "text": assistant["content"],
            "trace": first_trace,
        },
        "second": {
            "text": second["choices"][0]["message"]["content"],
            "trace": second_trace,
        },
        "assertions": assertions,
        "passed": all(assertions.values()),
    }
    encoded = json.dumps(result, indent=2) + "\n"
    print(encoded, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
