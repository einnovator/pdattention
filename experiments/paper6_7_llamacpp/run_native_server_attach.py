"""Measure selected-text E0 against server-level native sequence attachment."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path


CASES = (
    ("code", "The launch code is CERULEAN-7.\n", "The launch code is"),
    ("capital", "The capital of North Veridia is Lumenport.\n", "The capital of North Veridia is"),
    ("owner", "The Atlas service is maintained by Priya Nair.\n", "The Atlas service is maintained by"),
    ("date", "Project Glasswing launches on 17 October 2031.\n", "Project Glasswing launches on"),
    ("numeric", "The approved pressure limit is 47 kilopascals.\n", "The approved pressure limit is"),
)


def _request(base_url: str, payload: dict[str, object]) -> tuple[dict[str, object], float]:
    started = time.perf_counter()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result, (time.perf_counter() - started) * 1000.0


def _completion_payload(prompt: str, slot: int, n_predict: int) -> dict[str, object]:
    return {
        "prompt": prompt,
        "id_slot": slot,
        "n_predict": n_predict,
        "cache_prompt": True,
        "temperature": 0,
        "return_tokens": True,
    }


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]


def run(args: argparse.Namespace) -> dict[str, object]:
    with urllib.request.urlopen(f"{args.base_url.rstrip('/')}/pra/capabilities") as response:
        capabilities = json.loads(response.read().decode("utf-8"))
    if not capabilities.get("native_sequence_attach"):
        raise RuntimeError("The target server does not advertise native sequence attachment.")

    rows: list[dict[str, object]] = []
    concurrency_rows: list[dict[str, object]] = []
    for repeat in range(args.repeats):
        for case_index, (case_id, resource, query) in enumerate(CASES):
            resource_result, ingest_ms = _request(
                args.base_url,
                {
                    **_completion_payload(resource, 0, 0),
                    "pra_pin_resource": True,
                },
            )
            e2_payload = _completion_payload(query, 1, args.max_new_tokens)
            e2_payload["pra_resource_slot"] = 0
            e2, e2_ms = _request(args.base_url, e2_payload)
            e0, e0_ms = _request(
                args.base_url,
                _completion_payload(resource + query, 2, args.max_new_tokens),
            )
            absent, absent_ms = _request(
                args.base_url,
                _completion_payload(query, 3, args.max_new_tokens),
            )
            warm, warm_ms = _request(args.base_url, e2_payload)
            rows.append(
                {
                    "repeat": repeat,
                    "case_id": case_id,
                    "resource_sha256": hashlib.sha256(resource.encode()).hexdigest(),
                    "resource_ingest_ms": ingest_ms,
                    "resource_tokens": resource_result.get("tokens_evaluated"),
                    "e0_ms": e0_ms,
                    "e2_ms": e2_ms,
                    "e2_warm_ms": warm_ms,
                    "absent_ms": absent_ms,
                    "e0_tokens": e0.get("tokens", []),
                    "e2_tokens": e2.get("tokens", []),
                    "e2_warm_tokens": warm.get("tokens", []),
                    "absent_tokens": absent.get("tokens", []),
                    "e0_text": e0.get("content", ""),
                    "e2_text": e2.get("content", ""),
                    "e2_warm_text": warm.get("content", ""),
                    "absent_text": absent.get("content", ""),
                    "e0_prompt_tokens": e0.get("tokens_evaluated"),
                    "e2_wire_tokens": e2.get("pra", {}).get("wire_tokens"),
                    "e2_native_tokens": e2.get("pra", {}).get("native_tokens"),
                    "e2_cached_tokens": e2.get("timings", {}).get("cache_n"),
                    "physical_kv_copy": e2.get("pra", {}).get("physical_kv_copy"),
                }
            )

            if case_index == 0:
                slots = list(range(1, args.concurrency + 1))
                started = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    futures = []
                    for slot in slots:
                        payload = _completion_payload(query, slot, args.max_new_tokens)
                        payload["pra_resource_slot"] = 0
                        futures.append(pool.submit(_request, args.base_url, payload))
                    results = [future.result() for future in futures]
                wall_ms = (time.perf_counter() - started) * 1000.0
                concurrency_rows.append(
                    {
                        "repeat": repeat,
                        "concurrency": args.concurrency,
                        "wall_ms": wall_ms,
                        "requests_per_second": args.concurrency / (wall_ms / 1000.0),
                        "exact_to_serial_e2": sum(
                            result.get("tokens", []) == e2.get("tokens", [])
                            for result, _ in results
                        ),
                        "request_ms": [elapsed for _, elapsed in results],
                        "physical_kv_copy": any(
                            result.get("pra", {}).get("physical_kv_copy", True)
                            for result, _ in results
                        ),
                    }
                )

    e0_ms = [float(row["e0_ms"]) for row in rows]
    e2_ms = [float(row["e2_ms"]) for row in rows]
    warm_ms = [float(row["e2_warm_ms"]) for row in rows]
    payload = {
        "schema_version": "paper6.7-llamacpp-native-server-v1",
        "experiment": "matched_e0_e2_server_sequence_attach",
        "evidence_tier": "LIVE_ENGINE_SERVER_COHORT",
        "capabilities": capabilities,
        "configuration": {
            "base_url": args.base_url,
            "repeats": args.repeats,
            "cases": len(CASES),
            "max_new_tokens": args.max_new_tokens,
            "concurrency": args.concurrency,
        },
        "rows": rows,
        "concurrency_rows": concurrency_rows,
        "summary": {
            "runs": len(rows),
            "e0_e2_exact": sum(row["e0_tokens"] == row["e2_tokens"] for row in rows),
            "e2_warm_exact": sum(row["e2_tokens"] == row["e2_warm_tokens"] for row in rows),
            "absent_differs": sum(row["e2_tokens"] != row["absent_tokens"] for row in rows),
            "physical_kv_copy": any(bool(row["physical_kv_copy"]) for row in rows),
            "mean_e0_ms": statistics.mean(e0_ms),
            "mean_e2_ms": statistics.mean(e2_ms),
            "mean_e2_warm_ms": statistics.mean(warm_ms),
            "e2_over_e0": statistics.mean(e2_ms) / statistics.mean(e0_ms),
            "e2_warm_over_e0": statistics.mean(warm_ms) / statistics.mean(e0_ms),
            "e0_p95_ms": _percentile(e0_ms, 0.95),
            "e2_p95_ms": _percentile(e2_ms, 0.95),
            "e2_warm_p95_ms": _percentile(warm_ms, 0.95),
            "concurrent_exact": sum(
                row["exact_to_serial_e2"] for row in concurrency_rows
            ),
            "concurrent_requests": len(concurrency_rows) * args.concurrency,
            "mean_concurrent_requests_per_second": statistics.mean(
                row["requests_per_second"] for row in concurrency_rows
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18087")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args())["summary"], indent=2))
