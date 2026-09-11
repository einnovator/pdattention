"""Compare isolated live-agent sessions under concurrent and sequential use."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from experiments.paper4_5_agent.context_treatment import (
    ContextTreatment,
    transform_chat_payload,
)
from experiments.paper4_5_agent.probe_live_vs_fresh_multiseed import (
    _chat,
    _close_session,
    _content,
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _physical(
    model: str,
    messages: list[dict[str, str]],
    *,
    session_id: str,
    request_index: int,
    budget_fraction: float,
) -> dict[str, Any]:
    payload, _ = transform_chat_payload(
        {
            "model": model,
            "messages": messages,
            "max_tokens": 32,
            "temperature": 0,
            "seed": 0,
        },
        mode=ContextTreatment.DIRECT_NATIVE_PRA,
        budget_fraction=budget_fraction,
        request_index=request_index,
        recent_completed_turns=1,
        recent_mutation_turns=0,
        recent_verification_turns=0,
    )
    payload["pra"]["session_id"] = session_id
    return payload


def _observation(label: str, turn: int) -> dict[str, str]:
    detail = " ".join(
        f"{label}-line-{turn}-{index}" for index in range(80)
    )
    return {
        "role": "user",
        "content": f"Command output for {label} turn {turn}: {detail}",
    }


def _build_source(
    wrapper_url: str, model: str, label: str, session_id: str,
) -> tuple[list[dict[str, str]], list[str], list[int]]:
    messages = [
        {
            "role": "system",
            "content": "You are a deterministic coding agent. Reply briefly.",
        },
        {"role": "user", "content": f"Task {label}: inspect the repository state."},
    ]
    hashes = []
    slots = []
    for request_index in range(1, 4):
        response = _chat(wrapper_url, _physical(
            model,
            messages,
            session_id=session_id,
            request_index=request_index,
            budget_fraction=1.0,
        ))
        assistant = _content(response)
        hashes.append(_sha256(assistant))
        plain_trace = next(
            (row for row in response.get("pra_trace", ())
             if row.get("stage") == "llama_cpp_plain"),
            None,
        )
        if plain_trace is not None and plain_trace.get("request_slot") is not None:
            slots.append(int(plain_trace["request_slot"]))
        messages.append({"role": "assistant", "content": assistant})
        messages.append(_observation(label, request_index))
    return messages, hashes, slots


def _target(
    wrapper_url: str, model: str, messages: list[dict[str, str]], session_id: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = _chat(wrapper_url, _physical(
        model,
        messages,
        session_id=session_id,
        request_index=4,
        budget_fraction=0.75,
    ))
    elapsed = time.perf_counter() - started
    trace = next(
        row for row in reversed(response.get("pra_trace", ()))
        if row.get("stage") in {
            "llama_cpp_live_prefix_subset",
            "llama_cpp_live_prefix_full_continue",
        }
    )
    return {
        "response_sha256": _sha256(_content(response)),
        "response": _content(response),
        "elapsed_s": elapsed,
        "trace": trace,
    }


def _scenario(
    wrapper_url: str, model: str, prefix: str, *, run_concurrently: bool,
) -> dict[str, Any]:
    sessions = {label: f"{prefix}-{label.lower()}" for label in ("A", "B")}
    built = {
        label: _build_source(wrapper_url, model, label, sessions[label])
        for label in ("A", "B")
    }
    started = time.perf_counter()
    if run_concurrently:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                label: executor.submit(
                    _target, wrapper_url, model, built[label][0], sessions[label],
                )
                for label in ("A", "B")
            }
            targets = {label: futures[label].result() for label in ("A", "B")}
    else:
        targets = {
            label: _target(wrapper_url, model, built[label][0], sessions[label])
            for label in ("A", "B")
        }
    wall = time.perf_counter() - started
    for session_id in sessions.values():
        _close_session(wrapper_url, session_id)
    return {
        "concurrent": run_concurrently,
        "source_response_sha256": {
            label: built[label][1] for label in ("A", "B")
        },
        "source_slots": {label: built[label][2] for label in ("A", "B")},
        "targets": targets,
        "target_wall_s": wall,
    }


def run(output_path: Path, *, wrapper_url: str, model: str) -> dict[str, Any]:
    concurrent = _scenario(
        wrapper_url, model, "concurrent", run_concurrently=True,
    )
    sequential = _scenario(
        wrapper_url, model, "sequential", run_concurrently=False,
    )
    labels = ("A", "B")
    payload = {
        "model": model,
        "concurrent": concurrent,
        "sequential": sequential,
        "source_replay_exact": all(
            concurrent["source_response_sha256"][label]
            == sequential["source_response_sha256"][label]
            for label in labels
        ),
        "target_replay_exact": all(
            concurrent["targets"][label]["response_sha256"]
            == sequential["targets"][label]["response_sha256"]
            for label in labels
        ),
        "distinct_source_slots": len({
            concurrent["source_slots"][label][0] for label in labels
        }) == len(labels),
        "both_targets_sparse": all(
            concurrent["targets"][label]["trace"]["stage"]
            == "llama_cpp_live_prefix_subset"
            for label in labels
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
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
            "source_replay_exact", "target_replay_exact",
            "distinct_source_slots", "both_targets_sparse",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
