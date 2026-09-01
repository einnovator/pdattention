"""Measure off-node SGLang WARM prefetch and worker-affinity economics."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.paper6_1_sglang.run_hicache import SEEDS, _source


def _integers(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("Expected comma-separated non-negative integers.")
    return result


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def _stable_worker(key: str, workers: int) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % workers


def _max_delta(expected, restored) -> float:
    import mlx.core as mx

    values = []
    for left, right in zip(expected.layers, restored.layers):
        values.extend(
            (
                float(mx.max(mx.abs(left.keys - right.keys)).item()),
                float(mx.max(mx.abs(left.values - right.values)).item()),
            )
        )
    return max(values, default=0.0)


class PromotionCoordinator:
    """Coalesce one worker/resource transfer on the caller's event loop."""

    def __init__(self, caches, executor: ThreadPoolExecutor) -> None:
        self.caches = caches
        self.executor = executor
        self.tasks: dict[tuple[int, str], asyncio.Future] = {}

    def prefetch(self, worker: int, key: str) -> asyncio.Future:
        identity = (worker, key)
        existing = self.tasks.get(identity)
        if existing is not None:
            return existing
        loop = asyncio.get_running_loop()
        task = loop.run_in_executor(self.executor, self.caches[worker].get, key)
        self.tasks[identity] = task
        return task


async def _wave(
    *,
    caches,
    keys: list[str],
    policy: str,
    workers: int,
    lead_ms: int,
    randomizer: random.Random,
    executor: ThreadPoolExecutor,
) -> dict[str, object]:
    coordinator = PromotionCoordinator(caches, executor)
    assignments = [
        _stable_worker(key, workers)
        if policy == "affinity"
        else randomizer.randrange(workers)
        for key in keys
    ]
    hot_before = [
        caches[worker].placement(key) is not None
        and caches[worker].placement(key).value == "l1"
        for worker, key in zip(assignments, keys)
    ]
    prefetch_started = time.perf_counter()
    futures = [
        coordinator.prefetch(worker, key)
        for worker, key in zip(assignments, keys)
    ]
    await asyncio.sleep(lead_ms / 1000.0)
    demand_started = time.perf_counter()

    async def resolve(future) -> tuple[object, float, bool]:
        started = time.perf_counter()
        ready = future.done()
        value = await future
        return value, (time.perf_counter() - started) * 1000.0, ready

    resolved = await asyncio.gather(*(resolve(future) for future in futures))
    finished = time.perf_counter()
    stalls = [row[1] for row in resolved]
    return {
        "lead_ms": lead_ms,
        "requests": len(keys),
        "unique_worker_resources": len(set(zip(assignments, keys))),
        "hot_at_schedule": sum(hot_before),
        "ready_at_demand": sum(row[2] for row in resolved),
        "stall_ms": stalls,
        "demand_elapsed_ms": (finished - demand_started) * 1000.0,
        "prefetch_elapsed_ms": (finished - prefetch_started) * 1000.0,
        "assignments": assignments,
        "resolved": [row[0] for row in resolved],
    }


def _new_caches(root: Path, workers: int, remote, capacity: int):
    from pra_sglang.hicache import SGLangPRAHiCache
    from pra_sglang.hicache_backend import SGLangHiCacheStorageBackend

    return [
        SGLangPRAHiCache(
            root / f"worker-{worker}",
            max_l1_bytes=capacity,
            max_l2_bytes=capacity,
            l3_backend=SGLangHiCacheStorageBackend(
                remote, namespace="paper6-1-offnode"
            ),
        )
        for worker in range(workers)
    ]


async def _run(args, remote, prepared) -> tuple[list[dict], list[dict]]:
    expected = {item["key"]: item["memory"] for item in prepared}
    object_bytes = {item["key"]: item["memory"].nbytes for item in prepared}
    total_capacity = sum(object_bytes.values()) + max(object_bytes.values())
    lead_rows: list[dict] = []
    placement_rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="pra-sglang-offnode-") as directory:
        root = Path(directory)
        with ThreadPoolExecutor(max_workers=max(16, args.workers * 4)) as executor:
            for lead_ms in args.lead_ms:
                caches = _new_caches(
                    root / f"lead-{lead_ms}", args.workers, remote, total_capacity
                )
                remote.reset_metrics()
                keys = [prepared[0]["key"]] * args.lead_concurrency
                wave = await _wave(
                    caches=caches,
                    keys=keys,
                    policy="affinity",
                    workers=args.workers,
                    lead_ms=lead_ms,
                    randomizer=random.Random(args.seed),
                    executor=executor,
                )
                restored = {id(value): value for value in wave.pop("resolved")}.values()
                delta = max(
                    (_max_delta(expected[keys[0]], value) for value in restored),
                    default=0.0,
                )
                metrics = remote.metrics().to_dict()
                stalls = wave.pop("stall_ms")
                lead_rows.append(
                    {
                        **wave,
                        "stall_p50_ms": statistics.median(stalls),
                        "stall_p95_ms": _percentile(stalls, 0.95),
                        "remote_reads": metrics["reads"],
                        "remote_read_bytes": metrics["read_bytes"],
                        "remote_read_ms": metrics["read_ns"] / 1e6,
                        "requests_per_second": args.lead_concurrency
                        / max(float(wave["demand_elapsed_ms"]) / 1000.0, 1e-9),
                        "duplicate_kv_bytes_avoided": (
                            args.lead_concurrency - int(wave["unique_worker_resources"])
                        )
                        * object_bytes[keys[0]],
                        "max_tensor_delta": delta,
                    }
                )

            for concurrency in args.concurrency:
                for workload in ("shared", "mixed"):
                    for policy in ("affinity", "random"):
                        caches = _new_caches(
                            root / f"placement-{concurrency}-{workload}-{policy}",
                            args.workers,
                            remote,
                            total_capacity,
                        )
                        remote.reset_metrics()
                        randomizer = random.Random(
                            f"{args.seed}:{concurrency}:{workload}:{policy}"
                        )
                        all_stalls: list[float] = []
                        ready = hot = requests = elapsed_ms = 0.0
                        unique_resolved: dict[tuple[str, int], object] = {}
                        for round_index in range(args.rounds):
                            if workload == "shared":
                                keys = [prepared[0]["key"]] * concurrency
                            else:
                                keys = [
                                    prepared[(round_index * concurrency + index) % len(prepared)][
                                        "key"
                                    ]
                                    for index in range(concurrency)
                                ]
                            wave = await _wave(
                                caches=caches,
                                keys=keys,
                                policy=policy,
                                workers=args.workers,
                                lead_ms=args.affinity_lead_ms,
                                randomizer=randomizer,
                                executor=executor,
                            )
                            values = wave.pop("resolved")
                            for key, value in zip(keys, values):
                                unique_resolved[(key, id(value))] = value
                            all_stalls.extend(wave.pop("stall_ms"))
                            ready += int(wave["ready_at_demand"])
                            hot += int(wave["hot_at_schedule"])
                            requests += int(wave["requests"])
                            elapsed_ms += float(wave["prefetch_elapsed_ms"])
                        delta = max(
                            (
                                _max_delta(expected[key], value)
                                for (key, _identity), value in unique_resolved.items()
                            ),
                            default=0.0,
                        )
                        metrics = remote.metrics().to_dict()
                        placement_rows.append(
                            {
                                "concurrency": concurrency,
                                "rounds": args.rounds,
                                "workload": workload,
                                "policy": policy,
                                "lead_ms": args.affinity_lead_ms,
                                "requests": int(requests),
                                "hot_at_schedule": int(hot),
                                "ready_at_demand": int(ready),
                                "stall_p50_ms": statistics.median(all_stalls),
                                "stall_p95_ms": _percentile(all_stalls, 0.95),
                                "requests_per_second": requests
                                / max(elapsed_ms / 1000.0, 1e-9),
                                "remote_reads": metrics["reads"],
                                "remote_read_bytes": metrics["read_bytes"],
                                "remote_read_ms": metrics["read_ns"] / 1e6,
                                "max_tensor_delta": delta,
                            }
                        )
    return lead_rows, placement_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-url", required=True)
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument(
        "--revision", default="73e3e38d981303bc594367cd910ea6eb48349da8"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lead-concurrency", type=int, default=8)
    parser.add_argument("--lead-ms", type=_integers, default=(0, 10, 50, 100, 250))
    parser.add_argument("--concurrency", type=_integers, default=(1, 2, 4, 8, 16))
    parser.add_argument("--affinity-lead-ms", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import mlx.core as mx
    import sglang
    from mlx_lm import load
    from pra_mlx.native import encode_native_memory
    from pra_sglang.hicache_backend import SGLangHiCacheStorageBackend
    from pra_sglang.remote_warm import HTTPHiCacheStorageClient

    remote = HTTPHiCacheStorageClient(args.remote_url)
    health = remote.health()
    model, tokenizer = load(args.model, revision=args.revision)
    writer = SGLangHiCacheStorageBackend(remote, namespace="paper6-1-offnode")
    prepared = []
    for seed in SEEDS:
        tokens = tokenizer.encode(_source(seed), add_special_tokens=False)
        memory = encode_native_memory(model, tokens)
        mx.eval(*(array for layer in memory.layers for array in (layer.keys, layer.values)))
        key = f"paper6-1-offnode-seed-{seed}"
        writer.put(key, memory)
        prepared.append(
            {"key": key, "seed": seed, "source_tokens": len(tokens), "memory": memory}
        )

    lead_rows, placement_rows = asyncio.run(_run(args, remote, prepared))
    payload = {
        "schema_version": "paper6.1-offnode-warm-v1",
        "experiment": "offnode_warm_prefetch_affinity",
        "evidence_tier": "TWO_HOST_CONTROLLED_NATIVE_KV",
        "engine": "sglang-mlx",
        "engine_version": getattr(sglang, "__version__", "unknown"),
        "model_id": args.model,
        "model_revision": args.revision,
        "remote_url": args.remote_url,
        "remote_health": health,
        "workers": args.workers,
        "objects": [
            {
                "key": item["key"],
                "seed": item["seed"],
                "source_tokens": item["source_tokens"],
                "native_bytes": item["memory"].nbytes,
            }
            for item in prepared
        ],
        "lead_curve": lead_rows,
        "placement_curve": placement_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"lead_rows": len(lead_rows), "placement_rows": len(placement_rows)}, indent=2))


if __name__ == "__main__":
    main()
