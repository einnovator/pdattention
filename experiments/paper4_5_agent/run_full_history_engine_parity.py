"""Gate direct agent engines on plain versus PRA-100 full-history parity.

This probe intentionally contains no selection-policy experiment.  It sends the
same three deterministic logical turns through ordinary full prefill and through
resident full-history K/V, then checks the physical accounting and session
tombstone contract required before any PRA-90 result can be interpreted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence


TURN_INSTRUCTIONS = (
    "Reply with exactly ALPHA and no other text.",
    "Reply with exactly BETA and no other text.",
    "Reply with exactly GAMMA and no other text.",
)


def _read_json(url: str, *, timeout: float = 30) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _request_json(
    url: str,
    *,
    method: str = "POST",
    payload: Mapping[str, Any] | None = None,
    timeout: float = 600,
) -> dict[str, Any]:
    encoded = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _manifest(messages: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
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


def _text(response: Mapping[str, Any]) -> str:
    return str(response["choices"][0]["message"]["content"])


def _native_trace(response: Mapping[str, Any]) -> dict[str, Any]:
    matches = [
        row
        for row in response.get("pra_trace", ())
        if row.get("stage") == "native_attach"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one native_attach trace, found {len(matches)}.")
    return dict(matches[0])


def _plain_payload(
    model: str, messages: Sequence[Mapping[str, str]], max_tokens: int
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": list(messages),
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _pra100_payload(
    model: str,
    messages: Sequence[Mapping[str, str]],
    max_tokens: int,
    *,
    session_id: str,
    template_digest: str,
) -> dict[str, Any]:
    payload = _plain_payload(model, messages, max_tokens)
    payload["pra"] = {
        "tenant_id": "paper4-5-full-history-parity",
        "session_id": session_id,
        "resources": [],
        "allow_text_fallback": False,
        "metadata": {
            "history_projection": "live-agent-kv-v1",
            "target_retention_fraction": 1.0,
            "chat_template_digest": template_digest,
            "logical_message_manifest": _manifest(messages),
            "mandatory_message_indices": list(range(len(messages))),
        },
    }
    return payload


def _turn_gates(rows: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    traces = [row["trace"] for row in rows]
    return {
        "byte_exact_outputs": bool(rows) and all(row["exact"] for row in rows),
        "full_history_coverage": bool(traces) and all(
            row.get("full_retention") is True
            and row.get("selected_kv_tokens") == row.get("source_tokens")
            and row.get("source_position_base") == row.get("source_tokens")
            for row in traces
        ),
        "zero_selected_history_reencoding": bool(traces) and all(
            row.get("selected_history_reencoded_tokens") == 0 for row in traces
        ),
        "zero_selected_history_kv_copy": bool(traces) and all(
            row.get("physical_kv_copy") is False
            and row.get("physical_kv_copy_bytes") == 0
            and row.get("selected_history_kv_copy_bytes") == 0
            for row in traces
        ),
        "zero_native_attach_payload": bool(rows) and all(
            row["response_pra"].get("native_attach_bytes") == 0 for row in rows
        ),
        "append_only_history_encoding": all(
            0 < int(row.get("new_history_encoded_tokens") or 0)
            < int(row.get("source_tokens") or 0)
            for row in traces[1:]
        ),
        "canonical_copy_separately_accounted": bool(traces) and all(
            row.get("canonical_suffix_graft_d2d_bytes") is not None
            and row.get("total_kv_copy_bytes") is not None
            and int(row["total_kv_copy_bytes"])
            >= int(row["canonical_suffix_graft_d2d_bytes"])
            for row in traces
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.base_url.rstrip("/").removesuffix("/v1")
    completion_url = root + "/v1/chat/completions"
    health = _read_json(root + "/health")
    template_digest = str(health.get("chat_template_digest") or "")
    if not template_digest:
        raise RuntimeError("Direct engine did not advertise a chat-template digest.")

    suffix = uuid.uuid4().hex[:12]
    session_id = f"{args.session_prefix}-{suffix}"
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a deterministic engine-parity test assistant. "
                "Follow exact reply constraints."
            ),
        },
        {"role": "user", "content": TURN_INSTRUCTIONS[0]},
    ]
    rows: list[dict[str, Any]] = []
    for turn, instruction in enumerate(TURN_INSTRUCTIONS, start=1):
        if turn > 1:
            messages.extend((
                {"role": "assistant", "content": rows[-1]["plain_text"]},
                {"role": "user", "content": instruction},
            ))
        plain = _request_json(
            completion_url,
            payload=_plain_payload(args.model, messages, args.max_tokens),
        )
        pra100 = _request_json(
            completion_url,
            payload=_pra100_payload(
                args.model,
                messages,
                args.max_tokens,
                session_id=session_id,
                template_digest=template_digest,
            ),
        )
        plain_text, pra100_text = _text(plain), _text(pra100)
        rows.append({
            "turn": turn,
            "plain_text": plain_text,
            "pra100_text": pra100_text,
            "exact": plain_text == pra100_text,
            "trace": _native_trace(pra100),
            "response_pra": dict(pra100.get("pra") or {}),
        })
        if plain_text != pra100_text:
            break

    closed = _request_json(
        root + "/v1/pra/sessions/" + urllib.parse.quote(session_id, safe=""),
        method="DELETE",
        payload=None,
    )
    tombstone: dict[str, Any]
    try:
        _request_json(
            completion_url,
            payload=_pra100_payload(
                args.model,
                messages[:2],
                args.max_tokens,
                session_id=session_id,
                template_digest=template_digest,
            ),
        )
        tombstone = {"status": 200, "error": "terminated_session_was_reused"}
    except urllib.error.HTTPError as error:
        body = json.loads(error.read().decode("utf-8"))
        tombstone = {"status": error.code, "body": body}

    fresh_session = f"{args.session_prefix}-fresh-{suffix}"
    fresh = _request_json(
        completion_url,
        payload=_pra100_payload(
            args.model,
            messages[:2],
            args.max_tokens,
            session_id=fresh_session,
            template_digest=template_digest,
        ),
    )
    fresh_trace = _native_trace(fresh)
    fresh_closed = _request_json(
        root + "/v1/pra/sessions/" + urllib.parse.quote(fresh_session, safe=""),
        method="DELETE",
        payload=None,
    )

    gates = _turn_gates(rows)
    gates["terminated_session_tombstoned"] = bool(
        closed.get("closed") is True
        and tombstone.get("status") == 409
        and (tombstone.get("body") or {}).get("error") == "session_terminated"
    )
    gates["fresh_session_rebuilds_source"] = bool(
        fresh_closed.get("closed") is True
        and fresh_trace.get("new_history_encoded_tokens")
        == fresh_trace.get("source_tokens")
    )
    result = {
        "schema_version": "paper4.5.full-history-engine-parity.v1",
        "engine": args.engine,
        "model": args.model_source or args.model,
        "served_model": args.model,
        "model_revision": args.revision,
        "python_version": platform.python_version(),
        "hardware": args.hardware_label,
        "temperature": 0,
        "health": health,
        "turns_completed": len(rows),
        "gates": gates,
        "passed": len(rows) == len(TURN_INSTRUCTIONS) and all(gates.values()),
        "terminated_session_response": tombstone,
        "fresh_session_trace": fresh_trace,
        "rows": rows,
        "claim_boundary": (
            "This is an engine-only ordinary-text full-history mechanism and "
            "lifecycle gate. It is not an autonomous task solve or a selection result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", required=True, help="Served OpenAI model ID.")
    parser.add_argument("--model-source")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--session-prefix", default="paper45-full-history-parity")
    parser.add_argument("--hardware-label", default="unspecified")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "engine": result["engine"],
        "model": result["model"],
        "turns_completed": result["turns_completed"],
        "gates": result["gates"],
        "passed": result["passed"],
        "output": str(args.output),
    }, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
