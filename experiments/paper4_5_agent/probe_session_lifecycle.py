"""Exercise live-agent idempotency, stale commits, and drained termination."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from experiments.paper4_5_agent.context_treatment import (
    ContextTreatment,
    transform_chat_payload,
)


def _post(url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=7200) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode())


def _delete(url: str, session_id: str) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/pra/sessions/{urllib.parse.quote(session_id)}",
        method="DELETE",
    )
    with urllib.request.urlopen(request, timeout=7200) as response:
        return response.status, json.loads(response.read().decode())


def _status(url: str, session_id: str) -> dict[str, Any]:
    with urllib.request.urlopen(
        f"{url.rstrip('/')}/v1/pra/sessions/{urllib.parse.quote(session_id)}",
        timeout=30,
    ) as response:
        return json.loads(response.read().decode())


def _content(response: dict[str, Any]) -> str:
    return str(response["choices"][0]["message"]["content"])


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _payload(
    model: str,
    messages: list[dict[str, str]],
    *,
    session_id: str,
    request_index: int,
    seed: int,
    max_tokens: int = 8,
) -> dict[str, Any]:
    payload, _ = transform_chat_payload(
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "seed": seed,
            "max_tokens": max_tokens,
        },
        mode=ContextTreatment.DIRECT_NATIVE_PRA,
        budget_fraction=1.0,
        request_index=request_index,
    )
    payload["pra"]["session_id"] = session_id
    return payload


def run(output: Path, *, wrapper_url: str, model: str) -> dict[str, Any]:
    session_id = "lifecycle-gate"
    initial_messages = [
        {"role": "system", "content": "Reply with one short word."},
        {"role": "user", "content": "Say ready."},
    ]
    initial = _payload(
        model, initial_messages, session_id=session_id, request_index=1, seed=0,
    )
    initial_status, initial_response = _post(wrapper_url, initial)
    retry_status, retry_response = _post(wrapper_url, initial)

    assistant = _content(initial_response)
    next_messages = [
        *initial_messages,
        {"role": "assistant", "content": assistant},
        {"role": "user", "content": "Say done."},
    ]
    forks = [
        _payload(
            model,
            next_messages,
            session_id=session_id,
            request_index=2,
            seed=seed,
        )
        for seed in (11, 22)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_post, wrapper_url, payload) for payload in forks]
        fork_results = [future.result() for future in futures]

    delete_started = time.perf_counter()
    delete_status, delete_response = _delete(wrapper_url, session_id)
    delete_elapsed_s = time.perf_counter() - delete_started
    late_status, late_response = _post(wrapper_url, initial)

    cancel_session = "active-cancellation-gate"
    cancel_payload = _payload(
        model,
        initial_messages,
        session_id=cancel_session,
        request_index=1,
        seed=0,
        max_tokens=512,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        active_future = executor.submit(_post, wrapper_url, cancel_payload)
        active_seen = False
        for _ in range(200):
            state = _status(wrapper_url, cancel_session)
            if state.get("upstream_stream_open"):
                active_seen = True
                break
            time.sleep(0.01)
        cancel_started = time.perf_counter()
        cancel_delete_status, _ = _delete(wrapper_url, cancel_session)
        cancel_elapsed_s = time.perf_counter() - cancel_started
        cancelled_status, cancelled_response = active_future.result()

    successful_forks = [row for row in fork_results if row[0] == 200]
    conflicting_forks = [row for row in fork_results if row[0] == 409]
    retry_trace = retry_response.get("pra_trace", [])
    payload = {
        "model": model,
        "session_id": session_id,
        "initial_status": initial_status,
        "initial_response_sha256": _sha256(_content(initial_response)),
        "retry_status": retry_status,
        "retry_response_sha256": _sha256(_content(retry_response)),
        "retry_idempotent": (
            initial_status == retry_status == 200
            and _content(initial_response) == _content(retry_response)
            and any(
                row.get("stage") == "llama_cpp_idempotent_generation_replay"
                for row in retry_trace
            )
        ),
        "fork_statuses": sorted(row[0] for row in fork_results),
        "single_commit_winner": (
            len(successful_forks) == 1 and len(conflicting_forks) == 1
        ),
        "conflict_error": (
            conflicting_forks[0][1].get("error") if conflicting_forks else None
        ),
        "delete_status": delete_status,
        "delete_response": delete_response,
        "delete_elapsed_s": delete_elapsed_s,
        "late_status": late_status,
        "late_error": late_response.get("error"),
        "termination_tombstone_enforced": (
            delete_status == 200
            and late_status == 410
            and late_response.get("error") == "session_closed"
        ),
        "active_stream_seen": active_seen,
        "active_cancel_delete_status": cancel_delete_status,
        "active_cancel_request_status": cancelled_status,
        "active_cancel_request_error": cancelled_response.get("error"),
        "active_cancel_elapsed_s": cancel_elapsed_s,
        "active_cancellation_qualified": (
            active_seen
            and cancel_delete_status == 200
            and cancelled_status == 499
            and cancelled_response.get("error") == "generation_cancelled"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wrapper-url", required=True)
    parser.add_argument("--model", required=True)
    options = parser.parse_args()
    result = run(options.output, wrapper_url=options.wrapper_url, model=options.model)
    print(options.output)
    print(json.dumps({
        key: result[key] for key in (
            "retry_idempotent", "fork_statuses", "single_commit_winner",
            "termination_tombstone_enforced", "active_cancellation_qualified",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
