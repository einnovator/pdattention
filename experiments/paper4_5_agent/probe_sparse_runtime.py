"""Measure full-prefix versus sparse live-K/V request cost in llama.cpp."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from experiments.paper4_5_agent.context_treatment import (
    ContextTreatment,
    transform_chat_payload,
)
from experiments.paper4_5_agent.probe_session_lifecycle import (
    _content,
    _delete,
    _post,
)


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
            "temperature": 0,
            "seed": 0,
            "max_tokens": 1,
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


def _build_source(
    wrapper_url: str, model: str, session_id: str,
) -> tuple[list[dict[str, str]], list[str]]:
    messages = [
        {"role": "system", "content": "Reply with one token."},
        {"role": "user", "content": "Inspect the synthetic repository."},
    ]
    responses = []
    for request_index in range(1, 6):
        status, response = _post(wrapper_url, _physical(
            model,
            messages,
            session_id=session_id,
            request_index=request_index,
            budget_fraction=1.0,
        ))
        if status != 200:
            raise RuntimeError(response)
        assistant = _content(response)
        responses.append(assistant)
        messages.append({"role": "assistant", "content": assistant})
        detail = " ".join(
            f"file{request_index}/line{index}=unchanged" for index in range(40)
        )
        messages.append({
            "role": "user",
            "content": f"Synthetic command output {request_index}: {detail}",
        })
    return messages, responses


def _target(
    wrapper_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    session_id: str,
    budget_fraction: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    status, response = _post(wrapper_url, _physical(
        model,
        messages,
        session_id=session_id,
        request_index=6,
        budget_fraction=budget_fraction,
    ))
    wall_ms = (time.perf_counter() - started) * 1000
    if status != 200:
        raise RuntimeError(response)
    trace = next(
        row for row in reversed(response.get("pra_trace", ()))
        if row.get("stage") in {
            "llama_cpp_live_prefix_subset",
            "llama_cpp_live_prefix_full_continue",
        }
    )
    timings = response.get("engine_timings", {})
    return {
        "wall_ms": wall_ms,
        "prompt_ms": float(timings.get("prompt_ms", 0)),
        "predicted_ms": float(timings.get("predicted_ms", 0)),
        "response": _content(response),
        "trace": trace,
    }


def _median(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def run(
    output: Path, *, wrapper_url: str, model: str, repeats: int,
) -> dict[str, Any]:
    full_rows = []
    sparse_rows = []
    source_exact = []
    for repeat in range(repeats):
        sessions = {
            "full": f"runtime-full-{repeat}",
            "sparse": f"runtime-sparse-{repeat}",
        }
        full_messages, full_source = _build_source(
            wrapper_url, model, sessions["full"],
        )
        sparse_messages, sparse_source = _build_source(
            wrapper_url, model, sessions["sparse"],
        )
        source_exact.append(full_source == sparse_source)
        arms = (
            (("full", 1.0), ("sparse", 0.5))
            if repeat % 2 == 0
            else (("sparse", 0.5), ("full", 1.0))
        )
        for name, fraction in arms:
            row = _target(
                wrapper_url,
                model,
                full_messages if name == "full" else sparse_messages,
                session_id=sessions[name],
                budget_fraction=fraction,
            )
            (full_rows if name == "full" else sparse_rows).append(row)
        for session in sessions.values():
            _delete(wrapper_url, session)

    full_prompt = _median(full_rows, "prompt_ms")
    sparse_prompt = _median(sparse_rows, "prompt_ms")
    full_wall = _median(full_rows, "wall_ms")
    sparse_wall = _median(sparse_rows, "wall_ms")
    sparse_trace = sparse_rows[0]["trace"]
    full_trace = full_rows[0]["trace"]
    payload = {
        "model": model,
        "repeats": repeats,
        "order_alternated": True,
        "source_trajectories_exact": all(source_exact),
        "full": {
            "median_prompt_ms": full_prompt,
            "median_wall_ms": full_wall,
            "trace": full_trace,
        },
        "sparse": {
            "median_prompt_ms": sparse_prompt,
            "median_wall_ms": sparse_wall,
            "trace": sparse_trace,
        },
        "prompt_speedup": full_prompt / sparse_prompt if sparse_prompt else None,
        "wall_speedup": full_wall / sparse_wall if sparse_wall else None,
        "selected_fraction": (
            float(sparse_trace["selected_kv_tokens"])
            / float(sparse_trace["source_tokens"])
        ),
        "selected_text_reencoded_tokens": sparse_trace.get(
            "selected_text_reencoded_tokens"
        ),
        "physical_kv_copy": sparse_trace.get("physical_kv_copy"),
        "raw": {"full": full_rows, "sparse": sparse_rows},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wrapper-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    options = parser.parse_args()
    result = run(
        options.output,
        wrapper_url=options.wrapper_url,
        model=options.model,
        repeats=options.repeats,
    )
    print(options.output)
    print(json.dumps({
        key: result[key] for key in (
            "repeats", "source_trajectories_exact", "selected_fraction",
            "prompt_speedup", "wall_speedup",
            "selected_text_reencoded_tokens", "physical_kv_copy",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
