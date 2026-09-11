"""Qualify live-agent K/V offload, slot reuse, and exact restore."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

from experiments.paper4_5_agent.probe_session_lifecycle import (
    _content,
    _delete,
    _payload,
    _post,
    _sha256,
    _status,
)


def _offload(url: str, session_id: str) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/pra/sessions/{session_id}/offload",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=7200) as response:
        return response.status, json.loads(response.read().decode())


def _first_payload(model: str, session_id: str) -> dict[str, Any]:
    return _payload(
        model,
        [
            {"role": "system", "content": "Reply with one short word."},
            {"role": "user", "content": "Say ready."},
        ],
        session_id=session_id,
        request_index=1,
        seed=0,
        max_tokens=8,
    )


def _next_payload(
    model: str, session_id: str, assistant: str,
) -> dict[str, Any]:
    return _payload(
        model,
        [
            {"role": "system", "content": "Reply with one short word."},
            {"role": "user", "content": "Say ready."},
            {"role": "assistant", "content": assistant},
            {"role": "user", "content": "Say done."},
        ],
        session_id=session_id,
        request_index=2,
        seed=0,
        max_tokens=8,
    )


def run(output: Path, *, wrapper_url: str, model: str) -> dict[str, Any]:
    control_id = "offload-control"
    offload_id = "offload-treatment"
    pressure_id = "offload-pressure"

    control_first_status, control_first = _post(
        wrapper_url, _first_payload(model, control_id),
    )
    control_assistant = _content(control_first)
    control_next_status, control_next = _post(
        wrapper_url,
        _next_payload(model, control_id, control_assistant),
    )
    _delete(wrapper_url, control_id)

    treatment_first_status, treatment_first = _post(
        wrapper_url, _first_payload(model, offload_id),
    )
    treatment_assistant = _content(treatment_first)
    offload_status, offload_receipt = _offload(wrapper_url, offload_id)
    offloaded_state = _status(wrapper_url, offload_id)

    pressure_status, pressure_response = _post(
        wrapper_url, _first_payload(model, pressure_id),
    )
    pressure_state = _status(wrapper_url, pressure_id)
    _delete(wrapper_url, pressure_id)

    restored_status, restored_response = _post(
        wrapper_url,
        _next_payload(model, offload_id, treatment_assistant),
    )
    restored_state = _status(wrapper_url, offload_id)
    _delete(wrapper_url, offload_id)

    payload = {
        "model": model,
        "control_statuses": [control_first_status, control_next_status],
        "treatment_statuses": [treatment_first_status, restored_status],
        "initial_response_exact": control_assistant == treatment_assistant,
        "restored_response_exact": _content(control_next) == _content(restored_response),
        "control_response_sha256": _sha256(_content(control_next)),
        "restored_response_sha256": _sha256(_content(restored_response)),
        "offload_status": offload_status,
        "offload_receipt": offload_receipt,
        "offloaded_state": offloaded_state,
        "pressure_status": pressure_status,
        "pressure_response_sha256": _sha256(_content(pressure_response)),
        "pressure_source_slot": pressure_state.get("source_slot"),
        "restored_state": restored_state,
        "slot_reclaimed_while_offloaded": (
            offload_receipt.get("source_slot") == pressure_state.get("source_slot")
        ),
        "restored_pinned": bool(restored_state.get("source_slot") is not None),
        "restored_without_text_prefill": bool(restored_state.get("last_restore")),
    }
    payload["qualified"] = all((
        payload["control_statuses"] == [200, 200],
        payload["treatment_statuses"] == [200, 200],
        payload["initial_response_exact"],
        payload["restored_response_exact"],
        offload_status == 200,
        offloaded_state.get("offloaded") is True,
        pressure_status == 200,
        payload["slot_reclaimed_while_offloaded"],
        payload["restored_pinned"],
        payload["restored_without_text_prefill"],
    ))
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
            "qualified", "initial_response_exact", "restored_response_exact",
            "slot_reclaimed_while_offloaded", "restored_without_text_prefill",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
