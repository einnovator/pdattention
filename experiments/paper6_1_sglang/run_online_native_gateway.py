"""Exercise SGLang selected K/V through the public PRA HTTP gateway."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import urllib.parse
import urllib.request

from experiments.engine_serving.matched_qa import load_matched_examples
from experiments.paper6_1_sglang.run_builtin_hicache_backend import _storage_config
from experiments.paper6_1_sglang.run_live_storage_lifecycle import _resolve_revision
from experiments.paper6_2_mlx.run_live_storage_concurrency import _percentile
from experiments.paper6_2_mlx.run_online_native_gateway import (
    _cancel_after_first_text,
    _payload,
    _post,
)


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
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--concurrency", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = _resolve_revision(args.model, args.revision)

    import sglang
    from pra_hf.engine_memory import LogicalPRABlockStore
    from pra_hf.gateway import PRAGateway, create_gateway_server
    from pra_hf.storage_lifecycle import (
        PRAStorageManager,
        PRAStoragePolicy,
        PRAStorageTierConfig,
    )
    from pra_sglang import (
        SGLangEngineAdapter,
        SGLangHiCacheByteBackend,
        SGLangHiCacheHotBridge,
        SGLangInProcessNativeExecutor,
        SGLangPRAHiCache,
    )
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
    from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory
    from transformers import AutoTokenizer

    _manifest, examples = load_matched_examples(
        args.manifest, args.dataset, args.cache_dir
    )
    example = examples[0]
    block_store = LogicalPRABlockStore()
    runner = MlxModelRunner(
        args.model,
        revision=revision,
        disable_radix_cache=False,
        pool_size=4096,
        mem_fraction_static=0.3,
        enable_sampling=False,
    )
    runner.init_cache_pools(req_to_token_pool=None)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)

    with tempfile.TemporaryDirectory(prefix="pra-sglang-online-") as directory:
        root = Path(directory)
        os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = str(
            root / "builtin-hicache"
        )
        builtin = StorageBackendFactory.create_backend(
            "file", _storage_config(args.model), None
        )
        hicache = SGLangPRAHiCache(
            root / "l1-l2", max_l1_bytes=2 * 1024**3, max_l2_bytes=2 * 1024**3
        )
        storage = PRAStorageManager(
            PRAStoragePolicy(
                profile="sglang-online-native",
                hot=PRAStorageTierConfig(max_bytes=2 * 1024**3),
                warm=PRAStorageTierConfig(
                    path=str(root / "warm-state"), max_bytes=4 * 1024**3
                ),
                cold=PRAStorageTierConfig(enabled=False),
            ),
            hot=SGLangHiCacheHotBridge(hicache),
            warm=SGLangHiCacheByteBackend(
                builtin, namespace="paper6-1-online-native"
            ),
        )
        executor = SGLangInProcessNativeExecutor(
            runner,
            tokenizer,
            model_id=args.model,
            model_revision=revision,
            block_store=block_store,
            storage_manager=storage,
        )
        adapter = SGLangEngineAdapter(
            "http://in-process", block_store=block_store, native_executor=executor
        )
        gateway = PRAGateway(adapter, mode="G11")
        server = create_gateway_server(gateway, host="127.0.0.1", port=0)
        thread = threading.Thread(
            target=server.serve_forever, name="pra-sglang-gateway", daemon=True
        )
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        resource_id = f"online-{example.example_id}"
        version = hashlib.sha256(example.selected_source.encode()).hexdigest()
        sessions = []
        try:
            health_started = time.perf_counter()
            with urllib.request.urlopen(f"{base}/health", timeout=10) as response:
                health = json.loads(response.read())
            health_ms = (time.perf_counter() - health_started) * 1000.0

            prime = _post(
                base,
                _payload(
                    model=args.model,
                    session_id="prime",
                    request_id="prime-request",
                    question=example.question,
                    resource_id=resource_id,
                    source=example.selected_source,
                    version=version,
                    max_new_tokens=args.max_new_tokens,
                ),
                stream=False,
            )
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
        "experiment": "paper6_1_sglang_online_native_gateway_v1",
        "engine": "sglang-mlx",
        "engine_version": getattr(sglang, "__version__", "unknown"),
        "model_id": args.model,
        "model_revision": revision,
        "dataset": args.dataset,
        "health_ms": health_ms,
        "capabilities": health,
        "prime": prime,
        "streamed": streamed,
        "concurrency_rows": concurrency_rows,
        "concurrency_execution": "threaded HTTP arrivals, serialized in-process runner",
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
