"""Compare live agent K/V with freshly encoded K/V at identical positions."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence


def _request_json(
    base_url: str, path: str, body: Mapping[str, Any] | None = None,
    *, method: str = "POST",
) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=7200) as response:
        return json.loads(response.read().decode("utf-8"))


def _tokens(base_url: str, messages: Sequence[Mapping[str, Any]], *, generate: bool) -> list[int]:
    rendered = _request_json(base_url, "/apply-template", {
        "messages": list(messages),
        "add_generation_prompt": generate,
    })
    tokenized = _request_json(base_url, "/tokenize", {
        "content": rendered["prompt"],
        "add_special": True,
    })
    return [int(token) for token in tokenized["tokens"]]


def _ranges(
    base_url: str,
    messages: list[dict[str, Any]],
    physical_payload: Mapping[str, Any],
    source_tokens: int,
) -> list[dict[str, Any]]:
    full = _tokens(base_url, messages, generate=True)
    boundaries = []
    for end in range(1, len(messages) + 1):
        prefix = _tokens(base_url, messages[:end], generate=False)
        if full[:len(prefix)] != prefix:
            raise RuntimeError(f"chat template is not prefix-separable at message {end - 1}")
        boundaries.append(len(prefix))

    pra = physical_payload["pra"]
    metadata = pra["metadata"]
    selected = sorted({
        int(resource["metadata"]["message_index"])
        for resource in pra.get("resources", ())
    })
    mandatory = [int(index) for index in metadata["mandatory_message_indices"]]
    systems = [
        index for index, message in enumerate(messages)
        if message.get("role") == "system"
    ]
    active_non_system = [
        index for index in mandatory if messages[index].get("role") != "system"
    ]
    active_start = min(active_non_system) if active_non_system else len(messages)

    resources_by_index = {}
    for resource in pra.get("resources", ()):
        resources_by_index.setdefault(
            int(resource["metadata"]["message_index"]), resource["metadata"]
        )
    ranges = []
    for index in [*systems, *selected]:
        start = 0 if index == 0 else boundaries[index - 1]
        stop = min(boundaries[index], source_tokens)
        if stop <= start:
            continue
        row = resources_by_index.get(index, {})
        ranges.append({
            "record_id": "system-prefix" if index in systems else f"m{index}",
            "parent_record_id": str(row.get("parent_record_id", f"m{index}")),
            "causal_group_id": str(row.get("causal_group_id", f"record:m{index}")),
            "start": start,
            "end": stop,
        })
    active_token_start = 0 if active_start == 0 else boundaries[active_start - 1]
    if active_token_start < source_tokens:
        ranges.append({
            "record_id": f"active-tail:m{active_start}",
            "parent_record_id": f"active-tail:m{active_start}",
            "causal_group_id": f"active-tail:m{active_start}",
            "start": active_token_start,
            "end": source_tokens,
        })
    ranges.sort(key=lambda row: (row["start"], row["end"]))
    return ranges


def run(
    interaction_path: Path,
    output_path: Path,
    *,
    llama_url: str,
    request_index: int,
    source_slot: int = 0,
    request_slot: int = 1,
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in interaction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    requests = [row for row in rows if row.get("event") == "request"]
    responses = [row for row in rows if row.get("event") == "response"]
    request_row = requests[request_index - 1]
    response_row = responses[request_index - 1]
    trace = response_row["payload"]["pra_trace"][-1]
    if trace.get("stage") != "llama_cpp_live_prefix_subset":
        raise ValueError(f"request {request_index} is not a sparse live-prefix request")
    source_tokens = int(trace["source_tokens"])
    messages = list(request_row["logical_payload"]["messages"])
    full_tokens = _tokens(llama_url, messages, generate=True)
    ranges = _ranges(
        llama_url, messages, request_row["physical_payload"], source_tokens,
    )

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
    result = _request_json(llama_url, "/completion", {
        "prompt": full_tokens[source_tokens:],
        "id_slot": request_slot,
        "pra_source_slot": source_slot,
        "pra_source_prefix_tokens": source_tokens,
        "pra_selected_ranges": ranges,
        "pra_commit_to_source": False,
        "n_predict": int(request_row["logical_payload"].get("max_tokens", 4096)),
        "cache_prompt": True,
        "temperature": float(request_row["logical_payload"].get("temperature", 0)),
        "seed": 0,
        "return_tokens": True,
    })
    expected = str(response_row["payload"]["choices"][0]["message"]["content"])
    actual = str(result.get("content", ""))
    payload = {
        "request_index": request_index,
        "source_tokens": source_tokens,
        "selected_kv_tokens": sum(row["end"] - row["start"] for row in ranges),
        "selected_ranges": ranges,
        "wire_tokens": len(full_tokens) - source_tokens,
        "expected_response_sha256": hashlib.sha256(expected.encode()).hexdigest(),
        "actual_response_sha256": hashlib.sha256(actual.encode()).hexdigest(),
        "exact_response": actual == expected,
        "expected_response": expected,
        "actual_response": actual,
        "raw": result,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interaction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llama-url", required=True)
    parser.add_argument("--request-index", type=int, required=True)
    parser.add_argument("--source-slot", type=int, default=0)
    parser.add_argument("--request-slot", type=int, default=1)
    options = parser.parse_args()
    result = run(
        options.interaction,
        options.output,
        llama_url=options.llama_url,
        request_index=options.request_index,
        source_slot=options.source_slot,
        request_slot=options.request_slot,
    )
    print(options.output)
    print(json.dumps({key: result[key] for key in (
        "exact_response", "source_tokens", "selected_kv_tokens", "wire_tokens",
        "expected_response_sha256", "actual_response_sha256",
    )}, indent=2))


if __name__ == "__main__":
    main()
