"""Exercise direct-MLX concurrent-session and native-state lifecycle controls.

This is a model-forward experiment, not an HTTP serving benchmark. It measures
two session-isolated native states under sequential and concurrent requests,
then exercises cooperative cancellation before forward, scoped session
termination, device/host tier transition, and inactive LRU eviction.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from pra_hf.subagent_mlx_native import (
    MLXStatePoolError,
    MLXSubagentStatePool,
    encode_mlx_subagent_memory,
    make_mlx_subagent_cache,
)


def resolve_model_snapshot(
    model: str,
    revision: str,
    *,
    snapshot_download=None,
) -> tuple[Path, str]:
    """Resolve the requested Hub revision before MLX-LM loads the checkpoint."""

    if not revision:
        raise ValueError("A pinned model revision is required.")
    if snapshot_download is None:
        from huggingface_hub import snapshot_download as hub_snapshot_download

        snapshot_download = hub_snapshot_download
    path = Path(snapshot_download(repo_id=model, revision=revision)).resolve()
    resolved_revision = path.name if path.parent.name == "snapshots" else revision
    return path, resolved_revision


def _text(tokenizer, tokens: int, marker: str) -> tuple[int, ...]:
    line = (
        f"Session {marker} immutable repository evidence preserves causal identity, "
        "resource validity, and selected native attention state. "
    )
    value = line * max(2, tokens // 12)
    ids = tokenizer.encode(value, add_special_tokens=False)
    while len(ids) < tokens:
        value += line
        ids = tokenizer.encode(value, add_special_tokens=False)
    return tuple(ids[:tokens])


def _forward(model, memory, query_ids: tuple[int, ...]) -> tuple[float, np.ndarray]:
    import mlx.core as mx

    cache = make_mlx_subagent_cache(model, memory)
    started = time.perf_counter()
    logits = model(mx.array(query_ids, dtype=mx.int32)[None], cache=cache)[0, -1]
    mx.eval(logits)
    return (time.perf_counter() - started) * 1000.0, np.asarray(logits.astype(mx.float32))


def _leased_forward(pool, model, session: str, target: str, query_ids):
    lease, memory = pool.acquire(session, "shared", target)
    try:
        elapsed, logits = _forward(model, memory, query_ids)
        return elapsed, logits
    finally:
        pool.release(lease.lease_uuid)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument("--revision", default="73e3e38d")
    parser.add_argument("--shared-tokens", type=int, default=2048)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    model_path, resolved_revision = resolve_model_snapshot(args.model, args.revision)
    model, tokenizer = load(str(model_path))
    query_ids = tuple(
        tokenizer.encode("\nName the state property preserved by this session. Answer:", add_special_tokens=False)
    )
    source_a = _text(tokenizer, args.shared_tokens, "A")
    source_b = _text(tokenizer, args.shared_tokens, "B")
    encode_started = time.perf_counter()
    memory_a = encode_mlx_subagent_memory(model, source_a)
    encode_a_ms = (time.perf_counter() - encode_started) * 1000.0
    encode_started = time.perf_counter()
    memory_b = encode_mlx_subagent_memory(model, source_b)
    encode_b_ms = (time.perf_counter() - encode_started) * 1000.0
    pool = MLXSubagentStatePool(
        device_capacity_bytes=memory_a.nbytes + memory_b.nbytes,
        max_records=2,
    )
    pool.put("session-a", "shared", memory_a)
    pool.put("session-b", "shared", memory_b)

    cross_session_blocked = False
    try:
        pool.acquire("session-c", "shared", "intruder")
    except MLXStatePoolError:
        cross_session_blocked = True

    sequential_ms: list[float] = []
    concurrent_ms: list[float] = []
    parity: list[bool] = []
    concurrent_errors: list[str] = []
    sequential_logits: dict[str, np.ndarray] = {}
    for repetition in range(args.repetitions):
        started = time.perf_counter()
        for session in ("session-a", "session-b"):
            _, logits = _leased_forward(
                pool, model, session, f"seq-{repetition}-{session}", query_ids
            )
            sequential_logits[session] = logits
        sequential_ms.append((time.perf_counter() - started) * 1000.0)

        barrier = threading.Barrier(2)

        def concurrent(session: str):
            barrier.wait()
            try:
                return session, _leased_forward(
                    pool, model, session, f"par-{repetition}-{session}", query_ids
                ), None
            except Exception as error:
                return session, None, f"{type(error).__name__}: {error}"

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(executor.map(concurrent, ("session-a", "session-b")))
        concurrent_ms.append((time.perf_counter() - started) * 1000.0)
        for session, result, error in results:
            if error is not None:
                concurrent_errors.append(error)
                continue
            assert result is not None
            _, logits = result
            parity.append(bool(np.array_equal(logits, sequential_logits[session])))
        mx.clear_cache()

    eviction_active_blocked = False
    guard, _ = pool.acquire("session-a", "shared", "eviction-guard")
    try:
        pool.evict("session-a", "shared")
    except MLXStatePoolError:
        eviction_active_blocked = True
    finally:
        pool.release(guard.lease_uuid)

    # Cooperative cancellation after acquisition but before model submission.
    cancel_acquired = threading.Event()
    cancel_continue = threading.Event()
    cancel_requested = threading.Event()
    cancellation_forward_executed = False

    def cancellable():
        nonlocal cancellation_forward_executed
        pool.acquire("session-a", "shared", "cancel-target")
        cancel_acquired.set()
        cancel_continue.wait(timeout=30)
        if not cancel_requested.is_set():
            cancellation_forward_executed = True

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(cancellable)
        cancel_acquired.wait(timeout=30)
        cancel_requested.set()
        cancellation_released = pool.cancel_target("session-a", "cancel-target")
        cancel_continue.set()
        future.result(timeout=30)

    # Demote and restore one record, then verify exact post-transition logits.
    baseline_ms, baseline_logits = _leased_forward(
        pool, model, "session-a", "tier-baseline", query_ids
    )
    demote_started = time.perf_counter()
    pool.demote("session-a", "shared")
    demote_ms = (time.perf_counter() - demote_started) * 1000.0
    promote_started = time.perf_counter()
    lease, restored = pool.acquire("session-a", "shared", "tier-restored")
    promote_ms = (time.perf_counter() - promote_started) * 1000.0
    restored_ms, restored_logits = _forward(model, restored, query_ids)
    pool.release(lease.lease_uuid)

    pool.acquire("session-b", "shared", "terminate-target")
    terminated = pool.terminate_session("session-b")
    post_termination = pool.snapshot()
    session_termination_scoped = terminated == 1 and any(
        row["session_uuid"] == "session-a" for row in post_termination["records"]
    )

    # Record pressure forces inactive LRU eviction, independently of demotion.
    memory_c = encode_mlx_subagent_memory(model, _text(tokenizer, args.shared_tokens, "C"))
    memory_d = encode_mlx_subagent_memory(model, _text(tokenizer, args.shared_tokens, "D"))
    pool.put("session-c", "shared", memory_c)
    pool.put("session-d", "shared", memory_d)
    final_snapshot = pool.snapshot()

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "paper9-direct-mlx-lifecycle-v1",
        "scope": (
            "direct MLX model-forward concurrency and cooperative pre-forward cancellation; "
            "not HTTP cancellation or mid-kernel preemption"
        ),
        "model": args.model,
        "requested_model_revision": args.revision,
        "resolved_model_revision": resolved_revision,
        "engine": "mlx-lm",
        "mlx_version": getattr(mx, "__version__", None),
        "host": platform.node(),
        "platform": platform.platform(),
        "shared_tokens": args.shared_tokens,
        "query_tokens": len(query_ids),
        "native_bytes_per_session": memory_a.nbytes,
        "encode_ms": {"session_a": encode_a_ms, "session_b": encode_b_ms},
        "concurrent_sessions": {
            "repetitions": args.repetitions,
            "sequential_wall_ms": sequential_ms,
            "concurrent_wall_ms": concurrent_ms,
            "mean_speedup": statistics.mean(sequential_ms) / statistics.mean(concurrent_ms),
            "exact_logit_parity": bool(parity) and all(parity),
            "errors": concurrent_errors,
        },
        "isolation": {"cross_session_lookup_blocked": cross_session_blocked},
        "cancellation": {
            "lease_released": cancellation_released,
            "forward_executed_after_cancel": cancellation_forward_executed,
            "boundary": "cooperative_before_model_submission",
        },
        "tier_transition": {
            "demote_ms": demote_ms,
            "promote_ms": promote_ms,
            "baseline_forward_ms": baseline_ms,
            "restored_forward_ms": restored_ms,
            "exact_logit_parity": bool(np.array_equal(baseline_logits, restored_logits)),
            "max_abs_logit_delta": float(np.max(np.abs(baseline_logits - restored_logits))),
        },
        "termination": {
            "records_removed": terminated,
            "session_scoped": session_termination_scoped,
        },
        "eviction": {
            "record_capacity": pool.max_records,
            "record_capacity_evictions": sum(
                event.get("reason") == "record_capacity" for event in pool.events
            ),
            "active_eviction_blocked": eviction_active_blocked,
        },
        "final_pool": final_snapshot,
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "events.jsonl").open("w", encoding="utf-8") as stream:
        for event in pool.events:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
