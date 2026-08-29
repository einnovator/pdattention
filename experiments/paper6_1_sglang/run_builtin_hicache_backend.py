"""Measure PRA objects through SGLang's built-in HiCache file backend."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from experiments.paper6_1_sglang.run_hicache import (
    EXPECTED,
    SEEDS,
    _generate,
    _query,
    _sglang_cache,
    _source,
)


def _storage_config(model: str):
    from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig

    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=True,
        is_page_first_layout=True,
        model_name=model,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument(
        "--revision", default="73e3e38d981303bc594367cd910ea6eb48349da8"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import mlx.core as mx
    import sglang
    from mlx_lm import load
    from pra_mlx.native import encode_native_memory
    from pra_sglang.hicache import PRAHiCacheTier, SGLangPRAHiCache
    from pra_sglang.hicache_backend import SGLangHiCacheStorageBackend
    from pra_sglang.mlx_native import (
        SGLangSelectedKVCache,
        install_selected_kv_attention,
    )
    from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory

    model, tokenizer = load(args.model, revision=args.revision)
    layer_count = len(getattr(getattr(model, "model", model), "layers"))
    patched_layers = install_selected_kv_attention(model)
    query = tokenizer.encode(_query(), add_special_tokens=False)
    rows = []
    with tempfile.TemporaryDirectory(prefix="pra-sglang-builtin-hicache-") as root:
        storage_root = Path(root) / "sglang-storage"
        os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = str(storage_root)
        storage = StorageBackendFactory.create_backend(
            "file", _storage_config(args.model), None
        )
        for seed in SEEDS:
            tokens = tokenizer.encode(_source(seed), add_special_tokens=False)
            memory = encode_native_memory(model, tokens)
            logical_key = f"resource-{seed}-v1"
            writer_backend = SGLangHiCacheStorageBackend(
                storage, namespace="paper6-1-pra"
            )
            started = time.perf_counter()
            stored_bytes = writer_backend.put(logical_key, memory)
            write_ms = (time.perf_counter() - started) * 1000.0

            # A fresh adapter/cache has no process-local object map. Its read
            # therefore verifies identity and bytes through HiCacheStorage.
            reader_backend = SGLangHiCacheStorageBackend(
                storage, namespace="paper6-1-pra"
            )
            reader = SGLangPRAHiCache(
                Path(root) / f"reader-{seed}",
                max_l1_bytes=memory.nbytes,
                max_l2_bytes=memory.nbytes,
                l3_backend=reader_backend,
            )
            started = time.perf_counter()
            restored = reader.get(logical_key, target=PRAHiCacheTier.L1)
            cold_read_to_l1_ms = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            warm = reader.get(logical_key, target=PRAHiCacheTier.L1)
            warm_l1_ms = (time.perf_counter() - started) * 1000.0

            native = [
                SGLangSelectedKVCache(local, layer, position_base=warm.source_tokens)
                for local, layer in zip(_sglang_cache(layer_count), warm.layers)
            ]
            output, generation_ms = _generate(model, tokenizer, query, native)
            mx.eval(*(layer.keys for layer in warm.layers))
            rows.append(
                {
                    "seed": seed,
                    "logical_key_sha256": __import__("hashlib").sha256(
                        logical_key.encode("utf-8")
                    ).hexdigest(),
                    "source_tokens": len(tokens),
                    "native_kv_bytes": memory.nbytes,
                    "stored_blob_bytes": stored_bytes,
                    "write_ms": write_ms,
                    "cold_backend_read_to_l1_ms": cold_read_to_l1_ms,
                    "warm_l1_ms": warm_l1_ms,
                    "generation_ms": generation_ms,
                    "output": output,
                    "exact_recovery": EXPECTED in output,
                    "fresh_adapter_backend_hit": reader_backend.exists(logical_key),
                    "pra_tokens_absent_from_radix_prefix": (
                        native[0].keys is native[0].local_cache.keys
                    ),
                    "scheduler_local_tokens_after_generation": native[0].offset,
                    "hicache": reader.metrics().to_dict(),
                }
            )

    payload = {
        "schema_version": "1.0",
        "experiment": "paper6_1_sglang_builtin_hicache_storage_v1",
        "evidence_tier": "CONTROLLED_BUILTIN_BACKEND",
        "engine": "sglang-mlx",
        "engine_version": getattr(sglang, "__version__", "unknown"),
        "model_id": args.model,
        "model_revision": args.revision,
        "storage_backend": "sglang_hicache_file",
        "storage_blob_compression": "none",
        "off_node_transport": False,
        "ordinary_radix_namespace_used": False,
        "patched_layers": patched_layers,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
