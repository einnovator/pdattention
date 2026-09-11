"""Replicate live-versus-fresh agent K/V equivalence on another model."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    _sample,
)
from experiments.paper4_5_agent.probe_position_preserving_prefill import (
    _ranges,
    _request_json,
    _tokens,
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _frozen_scaffold(
    interaction_path: Path, request_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [
        json.loads(line)
        for line in interaction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    requests = [row for row in rows if row.get("event") == "request"]
    messages = [
        dict(row)
        for row in requests[request_index - 1]["logical_payload"]["messages"]
    ]
    first_user = next(
        index for index, message in enumerate(messages)
        if message.get("role") == "user"
    )
    initial = [dict(row) for row in messages[:first_user + 1]]
    observations = [
        dict(row) for row in messages[first_user + 1:]
        if row.get("role") == "user"
    ]
    if len(observations) < request_index - 1:
        raise ValueError("frozen scaffold lacks enough causal observations")
    return initial, observations[:request_index - 1]


def _build_live_source(
    *,
    wrapper_url: str,
    model: str,
    initial: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    build_max_tokens: int,
) -> tuple[list[dict[str, Any]], list[str], str]:
    messages = [dict(row) for row in initial]
    response_hashes = []
    session_id = ""
    for request_index, observation in enumerate(observations, 1):
        logical = {
            "model": model,
            "messages": messages,
            "max_tokens": build_max_tokens,
            "temperature": 0,
            "seed": 0,
        }
        physical, _ = transform_chat_payload(
            logical,
            mode=ContextTreatment.DIRECT_NATIVE_PRA,
            budget_fraction=1.0,
            request_index=request_index,
        )
        session_id = str(physical["pra"]["session_id"])
        response = _chat(wrapper_url, physical)
        assistant = _content(response)
        response_hashes.append(_sha256(assistant))
        messages.append({"role": "assistant", "content": assistant})
        messages.append(dict(observation))
    return messages, response_hashes, session_id


def _sparse_target(
    *,
    wrapper_url: str,
    model: str,
    messages: list[dict[str, Any]],
    request_index: int,
) -> tuple[dict[str, Any], int, str]:
    logical = {
        "model": model,
        "messages": messages,
        "max_tokens": 1,
        "temperature": 0,
        "seed": 0,
    }
    physical, treatment = transform_chat_payload(
        logical,
        mode=ContextTreatment.DIRECT_NATIVE_PRA,
        budget_fraction=0.9,
        request_index=request_index,
    )
    if physical["pra"]["metadata"]["selection_complete"]:
        raise RuntimeError("cross-model target did not enter sparse selection")
    response = _chat(wrapper_url, physical)
    trace = next(
        row for row in reversed(response["pra_trace"])
        if row.get("stage") == "llama_cpp_live_prefix_subset"
    )
    return physical, int(trace["source_tokens"]), str(
        physical["pra"]["session_id"]
    )


def run(
    interaction_path: Path,
    output_path: Path,
    *,
    wrapper_url: str,
    llama_url: str,
    model: str,
    model_fingerprint: str,
    request_index: int,
    seeds: list[int],
    temperature: float,
    build_max_tokens: int,
    sample_max_tokens: int,
    source_slot: int,
    request_slot: int,
) -> dict[str, Any]:
    initial, observations = _frozen_scaffold(interaction_path, request_index)
    _request_json(llama_url, f"/pra/resources/{request_slot}", method="DELETE")
    _request_json(llama_url, f"/pra/resources/{source_slot}", method="DELETE")

    first_messages, first_hashes, first_session = _build_live_source(
        wrapper_url=wrapper_url,
        model=model,
        initial=initial,
        observations=observations,
        build_max_tokens=build_max_tokens,
    )
    _, source_tokens, target_session = _sparse_target(
        wrapper_url=wrapper_url,
        model=model,
        messages=first_messages,
        request_index=request_index,
    )
    _close_session(wrapper_url, target_session or first_session)

    messages, replay_hashes, session_id = _build_live_source(
        wrapper_url=wrapper_url,
        model=model,
        initial=initial,
        observations=observations,
        build_max_tokens=build_max_tokens,
    )
    if replay_hashes != first_hashes:
        _close_session(wrapper_url, session_id)
        raise RuntimeError("cross-model source-building replay is not deterministic")
    logical = {
        "model": model,
        "messages": messages,
        "max_tokens": sample_max_tokens,
        "temperature": temperature,
    }
    physical, treatment = transform_chat_payload(
        logical,
        mode=ContextTreatment.DIRECT_NATIVE_PRA,
        budget_fraction=0.9,
        request_index=request_index,
    )
    full_tokens = _tokens(llama_url, messages, generate=True)
    ranges = _ranges(llama_url, messages, physical, source_tokens)
    suffix = list(full_tokens[source_tokens:])

    live = {}
    for seed in seeds:
        live[seed] = _sample(
            llama_url,
            source_slot=source_slot,
            request_slot=request_slot,
            source_tokens=source_tokens,
            ranges=ranges,
            suffix=suffix,
            seed=seed,
            temperature=temperature,
            max_tokens=sample_max_tokens,
        )
    _close_session(wrapper_url, session_id)

    _request_json(llama_url, f"/pra/resources/{request_slot}", method="DELETE")
    _request_json(llama_url, f"/pra/resources/{source_slot}", method="DELETE")
    _request_json(llama_url, "/completion", {
        "prompt": full_tokens[:source_tokens],
        "id_slot": source_slot,
        "n_predict": 0,
        "cache_prompt": False,
        "temperature": 0,
        "seed": 0,
        "pra_pin_resource": True,
    })
    comparisons = []
    for seed in seeds:
        fresh = _sample(
            llama_url,
            source_slot=source_slot,
            request_slot=request_slot,
            source_tokens=source_tokens,
            ranges=ranges,
            suffix=suffix,
            seed=seed,
            temperature=temperature,
            max_tokens=sample_max_tokens,
        )
        live_text = str(live[seed].get("content", ""))
        fresh_text = str(fresh.get("content", ""))
        comparisons.append({
            "seed": seed,
            "exact_response": live_text == fresh_text,
            "live_sha256": _sha256(live_text),
            "fresh_sha256": _sha256(fresh_text),
            "live_response": live_text,
            "fresh_response": fresh_text,
        })

    _request_json(llama_url, f"/pra/resources/{request_slot}", method="DELETE")
    _request_json(llama_url, f"/pra/resources/{source_slot}", method="DELETE")
    payload = {
        "model": model,
        "model_fingerprint": model_fingerprint,
        "request_index": request_index,
        "source_build_responses": len(first_hashes),
        "source_build_response_sha256": first_hashes,
        "source_replay_exact": replay_hashes == first_hashes,
        "source_tokens": source_tokens,
        "selected_kv_tokens": sum(row["end"] - row["start"] for row in ranges),
        "wire_tokens": len(suffix),
        "logical_token_saving_fraction_estimate": treatment.token_saving_fraction_estimate,
        "temperature": temperature,
        "sample_max_tokens": sample_max_tokens,
        "seeds": seeds,
        "comparisons": comparisons,
        "all_exact": all(row["exact_response"] for row in comparisons),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interaction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wrapper-url", required=True)
    parser.add_argument("--llama-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--request-index", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--build-max-tokens", type=int, default=96)
    parser.add_argument("--sample-max-tokens", type=int, default=128)
    parser.add_argument("--source-slot", type=int, default=1)
    parser.add_argument("--request-slot", type=int, default=0)
    options = parser.parse_args()
    result = run(
        options.interaction,
        options.output,
        wrapper_url=options.wrapper_url,
        llama_url=options.llama_url,
        model=options.model,
        model_fingerprint=options.model_fingerprint,
        request_index=options.request_index,
        seeds=options.seeds,
        temperature=options.temperature,
        build_max_tokens=options.build_max_tokens,
        sample_max_tokens=options.sample_max_tokens,
        source_slot=options.source_slot,
        request_slot=options.request_slot,
    )
    print(options.output)
    print(json.dumps({
        "model": result["model"],
        "source_replay_exact": result["source_replay_exact"],
        "source_tokens": result["source_tokens"],
        "selected_kv_tokens": result["selected_kv_tokens"],
        "all_exact": result["all_exact"],
        "comparisons": [
            {key: row[key] for key in (
                "seed", "exact_response", "live_sha256", "fresh_sha256",
            )}
            for row in result["comparisons"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
