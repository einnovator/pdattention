"""Replay frozen agent requests directly against one cache-mode endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run(
    interaction_path: Path,
    output_path: Path,
    *,
    base_url: str,
    request_count: int,
    prefix_caching: bool,
    payload_field: str = "logical_payload",
    require_exact: bool = False,
) -> list[dict[str, Any]]:
    events = [
        json.loads(line)
        for line in interaction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    requests = [row for row in events if row.get("event") == "request"]
    expected_responses = [
        row for row in events if row.get("event") == "response"
    ]
    if len(requests) < request_count:
        raise ValueError(
            f"fixture has {len(requests)} requests, needs {request_count}"
        )
    if len(expected_responses) < request_count:
        raise ValueError(
            f"fixture has {len(expected_responses)} responses, needs {request_count}"
        )

    results = []
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    for ordinal, row in enumerate(requests[:request_count], 1):
        payload = dict(row[payload_field])
        payload["prefix_caching"] = prefix_caching
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=7200) as response:
            body = json.loads(response.read().decode("utf-8"))
        text = str(body["choices"][0]["message"]["content"])
        expected_text = str(
            expected_responses[ordinal - 1]["payload"]["choices"][0]
            ["message"]["content"]
        )
        exact = text == expected_text
        results.append({
            "request_index": ordinal,
            "request_input_sha256": row.get("request_input_sha256"),
            "response_text_sha256": _sha256(text),
            "expected_response_text_sha256": _sha256(expected_text),
            "exact_response": exact,
            "response_text": text,
            "usage": body.get("usage"),
            "pra": body.get("pra"),
            "pra_trace": body.get("pra_trace"),
        })
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
        if require_exact and not exact:
            raise RuntimeError(
                f"frozen response diverged at request {ordinal}: "
                f"expected {_sha256(expected_text)}, observed {_sha256(text)}"
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interaction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--request-count", type=int, default=2)
    parser.add_argument(
        "--payload-field",
        choices=("logical_payload", "physical_payload"),
        default="logical_payload",
    )
    parser.add_argument(
        "--prefix-caching", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        "--require-exact", action=argparse.BooleanOptionalAction, default=False,
    )
    options = parser.parse_args()
    rows = run(
        options.interaction,
        options.output,
        base_url=options.base_url,
        request_count=options.request_count,
        prefix_caching=options.prefix_caching,
        payload_field=options.payload_field,
        require_exact=options.require_exact,
    )
    print(options.output)
    print("\n".join(row["response_text_sha256"] for row in rows))


if __name__ == "__main__":
    main()
