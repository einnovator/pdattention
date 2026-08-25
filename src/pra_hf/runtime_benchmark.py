"""Structured microbenchmarks for portable PRA K/V runtime primitives."""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .runtime import (
    CompilationMode,
    KVInterval,
    KVMaterializer,
    MaterializationPlan,
    NativeKV,
    PackedNativeKVStore,
    RuntimeKVCache,
    SelectedKVGather,
    runtime_capabilities,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _time_call(operation, device: torch.device) -> tuple[float, Any]:
    _sync(device)
    started = time.perf_counter()
    result = operation()
    _sync(device)
    return time.perf_counter() - started, result


def _memory(
    *,
    batch: int,
    kv_heads: int,
    tokens: int,
    head_dim: int,
    device: torch.device,
    seed: int,
) -> NativeKV:
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (batch, kv_heads, tokens, head_dim)
    key = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    value = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    return NativeKV(key, value)


def _indices(tokens: int, selected_tokens: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(0, tokens - 1, selected_tokens, device=device).round().long().unique()


def _summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = ("study", "mode", "device", "batch", "candidate_tokens", "selected_tokens", "status")
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    result = []
    for group, values in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        seconds = [float(row["seconds"]) for row in values if row.get("seconds") is not None]
        base = dict(zip(keys, group))
        base.update(
            {
                "samples": len(seconds),
                "median_seconds": statistics.median(seconds) if seconds else None,
                "p95_seconds": _percentile(seconds, 0.95) if seconds else None,
                "p99_seconds": _percentile(seconds, 0.99) if seconds else None,
                "output_bytes": values[0].get("output_bytes", 0),
                "parity": all(bool(row.get("parity", True)) for row in values),
                "error": next((row.get("error") for row in values if row.get("error")), None),
            }
        )
        if seconds and values[0].get("selected_tokens"):
            base["selected_tokens_per_second"] = (
                int(values[0]["selected_tokens"]) * int(values[0].get("batch", 1))
                / max(base["median_seconds"], 1e-12)
            )
        result.append(base)
    return result


def run_runtime_microbenchmark(
    *,
    device: str = "auto",
    candidate_tokens: int = 4096,
    selected_tokens: int = 256,
    batches: Sequence[int] = (1, 4),
    kv_heads: int = 4,
    head_dim: int = 64,
    warmups: int = 3,
    repeats: int = 10,
    seed: int = 20260825,
    include_compile: bool = True,
) -> dict[str, Any]:
    """Measure identical eager/compiled gathers and cache reuse.

    The benchmark is a mechanism gate, not an end-to-end TTFT claim.  It emits
    unsupported compiler paths as rows and never substitutes eager timings.
    """

    if candidate_tokens <= 0 or not 0 < selected_tokens <= candidate_tokens:
        raise ValueError("Expected 0 < selected_tokens <= candidate_tokens.")
    if warmups < 0 or repeats <= 0:
        raise ValueError("warmups must be non-negative and repeats positive.")
    if device == "auto":
        target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target = torch.device(device)
    rows: list[dict[str, Any]] = []
    for batch in batches:
        memory = _memory(
            batch=batch,
            kv_heads=kv_heads,
            tokens=candidate_tokens,
            head_dim=head_dim,
            device=target,
            seed=seed + batch,
        )
        indices = _indices(candidate_tokens, selected_tokens, target)
        selected_count = int(indices.numel())
        eager = SelectedKVGather(CompilationMode.EAGER)
        eager_result = eager(memory, indices)
        for _ in range(warmups):
            eager(memory, indices)
        for repeat in range(repeats):
            seconds, result = _time_call(lambda: eager(memory, indices), target)
            rows.append(
                {
                    "study": "indexed_gather",
                    "mode": "eager_warm",
                    "device": str(target),
                    "batch": batch,
                    "candidate_tokens": candidate_tokens,
                    "selected_tokens": selected_count,
                    "repeat": repeat,
                    "seconds": seconds,
                    "output_bytes": result.nbytes,
                    "parity": True,
                    "status": "measured",
                }
            )

        compiled = SelectedKVGather(CompilationMode.TORCH_COMPILE) if include_compile else None
        if compiled is not None and compiled.compiled:
            try:
                cold_seconds, cold_result = _time_call(lambda: compiled(memory, indices), target)
                parity = torch.equal(cold_result.key, eager_result.key) and torch.equal(
                    cold_result.value, eager_result.value
                )
                rows.append(
                    {
                        "study": "indexed_gather",
                        "mode": "torch_compile_cold",
                        "device": str(target),
                        "batch": batch,
                        "candidate_tokens": candidate_tokens,
                        "selected_tokens": selected_count,
                        "repeat": 0,
                        "seconds": cold_seconds,
                        "output_bytes": cold_result.nbytes,
                        "parity": parity,
                        "status": "measured",
                    }
                )
                for _ in range(warmups):
                    compiled(memory, indices)
                for repeat in range(repeats):
                    seconds, result = _time_call(lambda: compiled(memory, indices), target)
                    rows.append(
                        {
                            "study": "indexed_gather",
                            "mode": "torch_compile_warm",
                            "device": str(target),
                            "batch": batch,
                            "candidate_tokens": candidate_tokens,
                            "selected_tokens": selected_count,
                            "repeat": repeat,
                            "seconds": seconds,
                            "output_bytes": result.nbytes,
                            "parity": torch.equal(result.key, eager_result.key)
                            and torch.equal(result.value, eager_result.value),
                            "status": "measured",
                        }
                    )
            except Exception as error:  # pragma: no cover - platform/toolchain dependent
                rows.append(
                    {
                        "study": "indexed_gather",
                        "mode": "torch_compile",
                        "device": str(target),
                        "batch": batch,
                        "candidate_tokens": candidate_tokens,
                        "selected_tokens": selected_count,
                        "repeat": 0,
                        "seconds": None,
                        "output_bytes": 0,
                        "parity": False,
                        "status": "unsupported",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        else:
            rows.append(
                {
                    "study": "indexed_gather",
                    "mode": "torch_compile",
                    "device": str(target),
                    "batch": batch,
                    "candidate_tokens": candidate_tokens,
                    "selected_tokens": selected_count,
                    "repeat": 0,
                    "seconds": None,
                    "output_bytes": 0,
                    "parity": False,
                    "status": "unsupported" if include_compile else "disabled",
                    "error": compiled.compile_error if compiled is not None else "disabled by protocol",
                }
            )

        interval_width = max(1, selected_count // 8)
        starts = torch.linspace(
            0,
            candidate_tokens - interval_width,
            8,
        ).round().long().tolist()
        plan = MaterializationPlan.build(
            tuple(KVInterval("memory://benchmark", 0, int(start), int(start) + interval_width) for start in starts),
            max_tokens=selected_count,
        )
        materializer = KVMaterializer()
        sources = {("memory://benchmark", 0): memory}
        for _ in range(warmups):
            materializer.materialize(sources, plan)
        for repeat in range(repeats):
            seconds, result = _time_call(lambda: materializer.materialize(sources, plan), target)
            rows.append(
                {
                    "study": "interval_pack",
                    "mode": "eager_warm",
                    "device": str(target),
                    "batch": batch,
                    "candidate_tokens": candidate_tokens,
                    "selected_tokens": plan.unique_tokens,
                    "repeat": repeat,
                    "seconds": seconds,
                    "output_bytes": result.physical_bytes,
                    "parity": True,
                    "status": "measured",
                }
            )

        if target.type == "cuda":
            cpu_memory = _memory(
                batch=batch,
                kv_heads=kv_heads,
                tokens=candidate_tokens,
                head_dim=head_dim,
                device=torch.device("cpu"),
                seed=seed + 100 + batch,
            )
            hierarchy_sources = [("ordinary_cpu_to_cuda", cpu_memory, False)]
            try:
                pinned = NativeKV(cpu_memory.key.pin_memory(), cpu_memory.value.pin_memory())
                hierarchy_sources.append(("pinned_cpu_to_cuda", pinned, True))
            except RuntimeError:
                pass
            hierarchy_sources.append(("hbm_resident", memory, False))
            for mode, hierarchy_memory, non_blocking in hierarchy_sources:
                hierarchy = {("memory://benchmark", 0): hierarchy_memory}
                for _ in range(warmups):
                    materializer.materialize(
                        hierarchy,
                        plan,
                        device=target,
                        non_blocking=non_blocking,
                    )
                for repeat in range(repeats):
                    seconds, result = _time_call(
                        lambda: materializer.materialize(
                            hierarchy,
                            plan,
                            device=target,
                            non_blocking=non_blocking,
                        ),
                        target,
                    )
                    rows.append(
                        {
                            "study": "memory_hierarchy",
                            "mode": mode,
                            "device": str(target),
                            "batch": batch,
                            "candidate_tokens": candidate_tokens,
                            "selected_tokens": plan.unique_tokens,
                            "repeat": repeat,
                            "seconds": seconds,
                            "output_bytes": result.physical_bytes,
                            "transfer_bytes": result.transfer_bytes,
                            "parity": True,
                            "status": "measured",
                        }
                    )

        layout_source_tokens = max(1, candidate_tokens // 8)
        layout_sources = {
            (f"memory://layout/{reference}", layer): _memory(
                batch=batch,
                kv_heads=kv_heads,
                tokens=layout_source_tokens,
                head_dim=head_dim,
                device=target,
                seed=seed + 1000 + 10 * reference + layer + batch,
            )
            for reference in range(4)
            for layer in range(2)
        }
        interval_tokens = max(1, selected_count // len(layout_sources))
        layout_plan = MaterializationPlan.build(
            tuple(
                KVInterval(uri, layer, 1, min(layout_source_tokens, 1 + interval_tokens))
                for uri, layer in layout_sources
            ),
            max_tokens=selected_count,
        )
        layout_reference = materializer.materialize(layout_sources, layout_plan)
        for layout in ("layer_major", "reference_major", "chunk_major", "block_major"):
            build_seconds, store = _time_call(
                lambda layout=layout: PackedNativeKVStore(
                    layout_sources,
                    layout=layout,
                    page_tokens=32,
                ),
                target,
            )
            rows.append(
                {
                    "study": "layout_build",
                    "mode": layout,
                    "device": str(target),
                    "batch": batch,
                    "candidate_tokens": candidate_tokens,
                    "selected_tokens": layout_plan.unique_tokens,
                    "repeat": 0,
                    "seconds": build_seconds,
                    "output_bytes": store.nbytes,
                    "index_bytes": store.index_bytes,
                    "parity": True,
                    "status": "measured",
                }
            )
            for _ in range(warmups):
                materializer.materialize(store, layout_plan)
            for repeat in range(repeats):
                seconds, result = _time_call(
                    lambda store=store: materializer.materialize(store, layout_plan),
                    target,
                )
                parity = all(
                    torch.equal(result.layers[layer].key, layout_reference.layers[layer].key)
                    and torch.equal(result.layers[layer].value, layout_reference.layers[layer].value)
                    for layer in layout_reference.layers
                )
                rows.append(
                    {
                        "study": "layout_gather",
                        "mode": layout,
                        "device": str(target),
                        "batch": batch,
                        "candidate_tokens": candidate_tokens,
                        "selected_tokens": layout_plan.unique_tokens,
                        "repeat": repeat,
                        "seconds": seconds,
                        "output_bytes": result.physical_bytes,
                        "index_bytes": store.index_bytes,
                        "parity": parity,
                        "status": "measured",
                    }
                )

    cache = RuntimeKVCache(max_bytes=3 * 1024, max_entries=3)
    for key in ("a", "b", "a", "c", "a", "d", "b"):
        if cache.get(key) is None:
            cache.put(key, key, nbytes=1024)
    cache_snapshot = cache.snapshot()
    return {
        "protocol": {
            "scope": "portable selected-KV mechanism microbenchmark",
            "quality_selection_frozen": True,
            "candidate_tokens": candidate_tokens,
            "selected_tokens": selected_tokens,
            "batches": list(batches),
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "warmups": warmups,
            "repeats": repeats,
            "seed": seed,
            "include_compile": include_compile,
        },
        "capabilities": runtime_capabilities(),
        "rows": rows,
        "summary": _summary(rows),
        "cache": cache_snapshot,
    }


def write_runtime_benchmark(result: Mapping[str, Any], directory: str | Path) -> dict[str, Path]:
    """Persist raw JSON, flat rows, and aggregate rows for paper generation."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "runtime_benchmark.json"
    rows_path = directory / "runtime_rows.csv"
    summary_path = directory / "runtime_summary.csv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    for path, rows in ((rows_path, result["rows"]), (summary_path, result["summary"])):
        fieldnames = sorted({key for row in rows for key in row})
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return {"json": json_path, "rows": rows_path, "summary": summary_path}
