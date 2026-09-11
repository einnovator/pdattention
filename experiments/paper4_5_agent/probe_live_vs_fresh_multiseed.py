"""Compare live-built and fresh-built selected K/V at matched positions and seeds."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from experiments.paper4_5_agent.probe_position_preserving_prefill import (
    _ranges,
    _request_json,
    _tokens,
)


def _chat(wrapper_url: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        wrapper_url.rstrip("/") + "/v1/chat/completions",
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=7200) as response:
        return json.loads(response.read().decode("utf-8"))


def _close_session(wrapper_url: str, session_id: str) -> dict[str, Any]:
    request = urllib.request.Request(
        wrapper_url.rstrip("/") + "/v1/pra/sessions/"
        + urllib.parse.quote(session_id, safe=""),
        method="DELETE",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _content(payload: Mapping[str, Any]) -> str:
    return str(payload["choices"][0]["message"]["content"])


def _sample(
    llama_url: str,
    *,
    source_slot: int,
    request_slot: int,
    source_tokens: int,
    ranges: list[dict[str, Any]],
    suffix: list[int],
    seed: int,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    _request_json(llama_url, f"/pra/resources/{request_slot}", method="DELETE")
    return _request_json(llama_url, "/completion", {
        "prompt": suffix,
        "id_slot": request_slot,
        "pra_source_slot": source_slot,
        "pra_source_prefix_tokens": source_tokens,
        "pra_selected_ranges": ranges,
        "pra_commit_to_source": False,
        "n_predict": max_tokens,
        "cache_prompt": True,
        "temperature": temperature,
        "seed": seed,
        "return_tokens": True,
    })


def run(
    interaction_path: Path,
    output_path: Path,
    *,
    wrapper_url: str,
    llama_url: str,
    request_index: int,
    seeds: list[int],
    temperature: float,
    max_tokens: int,
    source_slot: int,
    request_slot: int,
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in interaction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    requests = [row for row in rows if row.get("event") == "request"]
    responses = [row for row in rows if row.get("event") == "response"]
    if request_index < 2 or request_index > len(requests):
        raise ValueError("request index must have at least one prior frozen turn")

    session_id = f"kv-multiseed-r{request_index}"
    _request_json(llama_url, f"/pra/resources/{request_slot}", method="DELETE")
    _request_json(llama_url, f"/pra/resources/{source_slot}", method="DELETE")
    replay = []
    for offset in range(request_index - 1):
        payload = copy.deepcopy(requests[offset]["physical_payload"])
        payload["pra"]["session_id"] = session_id
        actual = _chat(wrapper_url, payload)
        expected = _content(responses[offset]["payload"])
        actual_text = _content(actual)
        exact = actual_text == expected
        replay.append({
            "request_index": offset + 1,
            "exact_response": exact,
            "expected_sha256": hashlib.sha256(expected.encode()).hexdigest(),
            "actual_sha256": hashlib.sha256(actual_text.encode()).hexdigest(),
        })
        if not exact:
            _close_session(wrapper_url, session_id)
            raise RuntimeError(
                f"live source replay diverged at request {offset + 1}"
            )

    target_request = requests[request_index - 1]
    target_response = responses[request_index - 1]
    trace = target_response["payload"]["pra_trace"][-1]
    source_tokens = int(trace["source_tokens"])
    messages = list(target_request["logical_payload"]["messages"])
    full_tokens = _tokens(llama_url, messages, generate=True)
    ranges = _ranges(
        llama_url, messages, target_request["physical_payload"], source_tokens,
    )
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
            max_tokens=max_tokens,
        )

    closed = _close_session(wrapper_url, session_id)
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
            max_tokens=max_tokens,
        )
        live_text = str(live[seed].get("content", ""))
        fresh_text = str(fresh.get("content", ""))
        comparisons.append({
            "seed": seed,
            "exact_response": live_text == fresh_text,
            "live_sha256": hashlib.sha256(live_text.encode()).hexdigest(),
            "fresh_sha256": hashlib.sha256(fresh_text.encode()).hexdigest(),
            "live_tokens_predicted": live[seed].get("tokens_predicted"),
            "fresh_tokens_predicted": fresh.get("tokens_predicted"),
            "live_response": live_text,
            "fresh_response": fresh_text,
        })

    _request_json(llama_url, f"/pra/resources/{request_slot}", method="DELETE")
    _request_json(llama_url, f"/pra/resources/{source_slot}", method="DELETE")
    payload = {
        "request_index": request_index,
        "source_tokens": source_tokens,
        "selected_kv_tokens": sum(row["end"] - row["start"] for row in ranges),
        "selected_ranges": ranges,
        "wire_tokens": len(suffix),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seeds": seeds,
        "source_replay": replay,
        "session_close": closed,
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
    parser.add_argument("--request-index", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303, 404, 505])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--source-slot", type=int, default=1)
    parser.add_argument("--request-slot", type=int, default=0)
    options = parser.parse_args()
    result = run(
        options.interaction,
        options.output,
        wrapper_url=options.wrapper_url,
        llama_url=options.llama_url,
        request_index=options.request_index,
        seeds=options.seeds,
        temperature=options.temperature,
        max_tokens=options.max_tokens,
        source_slot=options.source_slot,
        request_slot=options.request_slot,
    )
    print(options.output)
    print(json.dumps({
        "all_exact": result["all_exact"],
        "source_replay": result["source_replay"],
        "comparisons": [
            {key: row[key] for key in (
                "seed", "exact_response", "live_sha256", "fresh_sha256",
            )}
            for row in result["comparisons"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
