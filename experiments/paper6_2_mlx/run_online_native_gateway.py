"""Exercise native MLX PRA through the public HTTP gateway request path."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from experiments.engine_serving.matched_qa import load_matched_examples
from experiments.paper6_2_mlx.run_live_storage_concurrency import _percentile


def _post(base: str, payload: dict[str, object], *, stream: bool) -> dict[str, object]:
    body = json.dumps({**payload, "stream": stream}).encode("utf-8")
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        response_context = urllib.request.urlopen(request, timeout=300)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"PRA gateway returned HTTP {error.code}: {body}"
        ) from error
    with response_context as response:
        headers_ms = (time.perf_counter() - started) * 1000.0
        if not stream:
            result = json.loads(response.read())
            wall_ms = (time.perf_counter() - started) * 1000.0
            return {
                "status": response.status,
                "headers_ms": headers_ms,
                "ttft_ms": None,
                "itl_ms": None,
                "wall_ms": wall_ms,
                "output": result["choices"][0]["message"]["content"],
                "trace": result.get("pra_trace", ()),
            }

        pieces = []
        arrivals = []
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            row = json.loads(line[6:])
            if row.get("text"):
                pieces.append(str(row["text"]))
                arrivals.append((time.perf_counter() - started) * 1000.0)
        wall_ms = (time.perf_counter() - started) * 1000.0
    intervals = [right - left for left, right in zip(arrivals, arrivals[1:])]
    return {
        "status": 200,
        "headers_ms": headers_ms,
        "ttft_ms": arrivals[0] if arrivals else None,
        "itl_ms": sum(intervals) / len(intervals) if intervals else None,
        "wall_ms": wall_ms,
        "output": "".join(pieces),
        "stream_chunks": len(pieces),
    }


def _cancel_after_first_text(base: str, payload: dict[str, object]) -> dict[str, object]:
    body = json.dumps({**payload, "stream": True}).encode("utf-8")
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    response = urllib.request.urlopen(request, timeout=300)
    first_text_ms = None
    try:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            row = json.loads(line[6:])
            if row.get("text"):
                first_text_ms = (time.perf_counter() - started) * 1000.0
                break
    finally:
        response.close()
    return {"first_text_ms": first_text_ms, "client_closed": True}


def _payload(
    *,
    model: str,
    session_id: str,
    request_id: str,
    question: str,
    resource_id: str,
    source: str,
    version: str,
    max_new_tokens: int,
) -> dict[str, object]:
    return {
        "id": request_id,
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_new_tokens,
        "temperature": 0,
        "pra": {
            "tenant_id": "benchmark",
            "session_id": session_id,
            "resources": [
                {
                    "resource_id": resource_id,
                    "uri": f"pra://benchmark/{resource_id}",
                    "record_type": "qa_evidence",
                    "text": source,
                    "metadata": {
                        "tenant_id": "benchmark",
                        "shareable": True,
                        "version": version,
                    },
                }
            ],
            "pra_policy": {"selected_resource_ids": [resource_id]},
            "budget": {"max_resources": 1, "max_selected_tokens": 2048},
            "max_new_tokens": max_new_tokens,
            "required_capabilities": ["native_kv", "logical_refs"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="qasper")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/papers/shared/results/matched_e0_e2_qa_manifest.json"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "pra")
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--concurrency", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from mlx_lm import load
    from pra_hf.engine_memory import LogicalPRABlockStore
    from pra_hf.gateway import PRAGateway, create_gateway_server
    from pra_hf.storage_lifecycle import PRAStoragePolicy, PRAStorageTierConfig
    from pra_mlx import MLXEngineAdapter
    from pra_mlx.native import MLXInProcessNativeExecutor

    _manifest, examples = load_matched_examples(
        args.manifest, args.dataset, args.cache_dir
    )
    example = examples[0]
    model, tokenizer = load(args.model, revision=args.revision)
    block_store = LogicalPRABlockStore()
    with tempfile.TemporaryDirectory(prefix="pra-mlx-online-") as directory:
        root = Path(directory)
        policy = PRAStoragePolicy(
            profile="mlx-online-native",
            hot=PRAStorageTierConfig(max_bytes=2 * 1024**3),
            warm=PRAStorageTierConfig(
                path=str(root / "warm"), max_bytes=4 * 1024**3, representation="mmap"
            ),
            cold=PRAStorageTierConfig(enabled=False),
        )
        executor = MLXInProcessNativeExecutor(
            model,
            tokenizer,
            model_id=args.model,
            model_revision=args.revision,
            block_store=block_store,
            storage_policy=policy,
            max_resident_bytes=2 * 1024**3,
        )
        adapter = MLXEngineAdapter(
            "http://in-process", block_store=block_store, native_executor=executor
        )
        gateway = PRAGateway(adapter, mode="G11")
        server = create_gateway_server(gateway, host="127.0.0.1", port=0)
        thread = threading.Thread(
            target=server.serve_forever, name="pra-mlx-gateway", daemon=True
        )
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        resource_id = f"online-{example.example_id}"
        version = hashlib.sha256(example.selected_source.encode()).hexdigest()
        try:
            health_started = time.perf_counter()
            with urllib.request.urlopen(f"{base}/health", timeout=10) as response:
                health = json.loads(response.read())
            health_ms = (time.perf_counter() - health_started) * 1000.0

            prime_payload = _payload(
                model=args.model,
                session_id="prime",
                request_id="prime-request",
                question=example.question,
                resource_id=resource_id,
                source=example.selected_source,
                version=version,
                max_new_tokens=args.max_new_tokens,
            )
            prime = _post(base, prime_payload, stream=False)
            streamed = _post(
                base,
                _payload(
                    model=args.model,
                    session_id="stream",
                    request_id="stream-request",
                    question=example.question,
                    resource_id=resource_id,
                    source=example.selected_source,
                    version=version,
                    max_new_tokens=args.max_new_tokens,
                ),
                stream=True,
            )

            concurrency_rows = []
            sessions = []
            for concurrency in args.concurrency:
                payloads = []
                for index in range(concurrency):
                    session_id = f"concurrency-{concurrency}-{index}"
                    sessions.append(session_id)
                    payloads.append(
                        _payload(
                            model=args.model,
                            session_id=session_id,
                            request_id=f"request-{concurrency}-{index}",
                            question=example.question,
                            resource_id=resource_id,
                            source=example.selected_source,
                            version=version,
                            max_new_tokens=args.max_new_tokens,
                        )
                    )
                wave_started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    results = list(
                        pool.map(lambda payload: _post(base, payload, stream=True), payloads)
                    )
                wave_ms = (time.perf_counter() - wave_started) * 1000.0
                walls = [float(row["wall_ms"]) for row in results]
                ttfts = [float(row["ttft_ms"]) for row in results if row["ttft_ms"]]
                concurrency_rows.append(
                    {
                        "concurrency": concurrency,
                        "wave_ms": wave_ms,
                        "requests_per_second": concurrency
                        / max(wave_ms / 1000.0, 1e-9),
                        "request_p50_ms": _percentile(walls, 0.50),
                        "request_p95_ms": _percentile(walls, 0.95),
                        "request_p99_ms": _percentile(walls, 0.99),
                        "ttft_p50_ms": _percentile(ttfts, 0.50),
                        "ttft_p95_ms": _percentile(ttfts, 0.95),
                        "exact_output_rate": sum(
                            row["output"] == prime["output"] for row in results
                        )
                        / len(results),
                        "stream_chunk_min": min(int(row["stream_chunks"]) for row in results),
                    }
                )

            cancellation_session = "cancelled-session"
            cancelled = _cancel_after_first_text(
                base,
                _payload(
                    model=args.model,
                    session_id=cancellation_session,
                    request_id="cancelled-request",
                    question=example.question,
                    resource_id=resource_id,
                    source=example.selected_source,
                    version=version,
                    max_new_tokens=max(64, args.max_new_tokens),
                ),
            )
            time.sleep(0.1)
            inspect_url = (
                f"{base}/v1/pra/sessions/prime?tenant_id=benchmark&model="
                + urllib.parse.quote(args.model, safe="")
            )
            with urllib.request.urlopen(inspect_url, timeout=10) as response:
                session_state = json.loads(response.read())
            cleanup = []
            for session_id in set(sessions + ["prime", "stream", cancellation_session]):
                url = (
                    f"{base}/v1/pra/sessions/{session_id}?tenant_id=benchmark&model="
                    + urllib.parse.quote(args.model, safe="")
                )
                request = urllib.request.Request(url, method="DELETE")
                with urllib.request.urlopen(request, timeout=10) as response:
                    cleanup.append(json.loads(response.read()))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=10)
            executor.close()

    result = {
        "schema_version": "1.0",
        "experiment": "paper6_2_mlx_online_native_gateway_v1",
        "engine": "mlx-lm",
        "model_id": args.model,
        "model_revision": args.revision,
        "dataset": args.dataset,
        "health_ms": health_ms,
        "capabilities": health,
        "prime": prime,
        "streamed": streamed,
        "concurrency_rows": concurrency_rows,
        "cancellation": cancelled,
        "session_affinity": {
            "engine_type": session_state.get("engine_type"),
            "known_resources": session_state.get("known_resources"),
        },
        "cleanup": cleanup,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
