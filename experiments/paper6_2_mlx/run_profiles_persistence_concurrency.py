"""Measure MLX consumer-layer profiles, persistence, and request concurrency."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import tempfile
import time
from pathlib import Path

from experiments.paper6_2_mlx.run_native_kv import EXPECTED, SEEDS, _generate, _query, _source


def _profiles(layer_count: int) -> dict[str, tuple[int, ...]]:
    return {
        "all": tuple(range(layer_count)),
        "last_half": tuple(range(layer_count // 2, layer_count)),
        "last_quarter": tuple(range(3 * layer_count // 4, layer_count)),
        "last_4": tuple(range(max(0, layer_count - 4), layer_count)),
        "disabled": (),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument(
        "--revision", default="73e3e38d981303bc594367cd910ea6eb48349da8"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from pra_mlx.native import (
        MLXNativeFingerprint,
        encode_native_memory,
        load_native_memory,
        make_native_prompt_cache,
        save_native_memory,
    )

    model, tokenizer = load(args.model, revision=args.revision)
    layer_count = len(getattr(getattr(model, "model", model), "layers"))
    profile_rows = []
    persistence_rows = []
    concurrency_rows = []

    for seed in SEEDS:
        source = tokenizer.encode(_source(seed), add_special_tokens=False)
        query = tokenizer.encode(_query(), add_special_tokens=False)
        ordinary = make_prompt_cache(model)
        model(mx.array(source, dtype=mx.int32)[None], cache=ordinary)
        ordinary_logits = model(mx.array(query, dtype=mx.int32)[None], cache=ordinary)[
            :, -1, :
        ]
        memory = encode_native_memory(model, source)

        for profile, layers in _profiles(layer_count).items():
            cache = make_native_prompt_cache(model, memory, selected_layers=layers)
            logits = model(mx.array(query, dtype=mx.int32)[None], cache=cache)[:, -1, :]
            mx.eval(logits, ordinary_logits)
            error = mx.abs(logits - ordinary_logits)
            output, elapsed_ms = _generate(
                model,
                tokenizer,
                query,
                make_native_prompt_cache(model, memory, selected_layers=layers),
            )
            profile_rows.append(
                {
                    "seed": seed,
                    "profile": profile,
                    "selected_layers": list(layers),
                    "selected_layer_count": len(layers),
                    "active_native_kv_bytes": memory.selected_nbytes(layers),
                    "retained_native_kv_bytes": memory.nbytes,
                    "completion_latency_ms": elapsed_ms,
                    "max_logit_error_vs_ordinary_split": float(mx.max(error).item()),
                    "mean_logit_error_vs_ordinary_split": float(mx.mean(error).item()),
                    "argmax_matches_ordinary_split": bool(
                        mx.argmax(logits).item() == mx.argmax(ordinary_logits).item()
                    ),
                    "output": output,
                    "exact_recovery": EXPECTED in output,
                }
            )

        fingerprint = MLXNativeFingerprint(
            model_id=args.model,
            model_revision=args.revision,
            tokenizer_revision=args.revision,
            dtype=str(memory.layers[0].keys.dtype),
            position_policy="source_local",
            consumer_profile="all",
            resource_version=f"seed-{seed}",
        )
        with tempfile.TemporaryDirectory(prefix="pra-mlx-") as directory:
            target = Path(directory) / "selected-memory"
            save_started = time.perf_counter()
            arrays_path, _ = save_native_memory(target, memory, fingerprint)
            save_ms = (time.perf_counter() - save_started) * 1000.0
            load_started = time.perf_counter()
            loaded = load_native_memory(target, expected_fingerprint=fingerprint)
            load_ms = (time.perf_counter() - load_started) * 1000.0
            loaded_logits = model(
                mx.array(query, dtype=mx.int32)[None],
                cache=make_native_prompt_cache(model, loaded),
            )[:, -1, :]
            mx.eval(loaded_logits, ordinary_logits)
            persistence_rows.append(
                {
                    "seed": seed,
                    "serialized_bytes": arrays_path.stat().st_size,
                    "save_ms": save_ms,
                    "load_ms": load_ms,
                    "max_logit_error_vs_ordinary_split": float(
                        mx.max(mx.abs(loaded_logits - ordinary_logits)).item()
                    ),
                    "fingerprint_validated": True,
                }
            )

        def run_one(_index: int) -> tuple[str, float]:
            return _generate(
                model,
                tokenizer,
                query,
                make_native_prompt_cache(model, memory),
            )

        for concurrency in (1, 2, 4, 8):
            reset_peak = getattr(mx, "reset_peak_memory", None)
            if reset_peak is not None:
                reset_peak()
            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                results = tuple(pool.map(run_one, range(concurrency)))
            wall_ms = (time.perf_counter() - started) * 1000.0
            concurrency_rows.append(
                {
                    "seed": seed,
                    "concurrency": concurrency,
                    "wall_ms": wall_ms,
                    "requests_per_second": concurrency / max(wall_ms / 1000.0, 1e-9),
                    "exact_recovery_rate": sum(EXPECTED in text for text, _ in results)
                    / concurrency,
                    "mean_request_latency_ms": sum(value for _, value in results)
                    / concurrency,
                    "shared_native_kv_bytes": memory.nbytes,
                    "duplicate_native_kv_bytes": memory.nbytes * concurrency,
                    "sharing_bytes_saved": memory.nbytes * (concurrency - 1),
                    "mlx_active_memory_bytes": int(
                        getattr(mx, "get_active_memory", lambda: 0)()
                    ),
                    "mlx_peak_memory_bytes": int(
                        getattr(mx, "get_peak_memory", lambda: 0)()
                    ),
                }
            )

    payload = {
        "schema_version": "1.0",
        "experiment": "paper6_2_mlx_profiles_persistence_concurrency_v1",
        "evidence_tier": "CONTROLLED",
        "model_id": args.model,
        "model_revision": args.revision,
        "seeds": list(SEEDS),
        "layer_count": layer_count,
        "profile_rows": profile_rows,
        "persistence_rows": persistence_rows,
        "concurrency_rows": concurrency_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
