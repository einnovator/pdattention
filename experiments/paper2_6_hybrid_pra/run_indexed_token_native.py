"""Benchmark indexed token-native discovery and tokenizer invariance.

This runner uses only host-model tokenizers.  It separates index construction,
warm query latency, candidate narrowing, and perturbation robustness from the
later native-K/V generation experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pra_hf.hybrid_discovery import (  # noqa: E402
    HybridDiscoveryPolicy,
    TokenChunkRecord,
    TokenNativeIndex,
    _automatic_aliases,
    _ngrams,
    _normalize_piece,
    _word_terms,
)


MODEL_SPECS = {
    "qwen3_0_6b": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    "smollm2_135m": ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"),
}


def _system_manifest() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }


MODES = (
    "token_exact",
    "token_ngram",
    "token_edit",
    "token_approx",
    "token_weighted",
    "cascade",
)
STOP_STRATEGIES = ("none", "fixed", "idf")


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _token_ids(tokenizer, text: str) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if isinstance(values, torch.Tensor):
        values = values.reshape(-1).tolist()
    if values and isinstance(values[0], (list, tuple)):
        values = values[0]
    return tuple(int(value) for value in values)


def _record(tokenizer, index: int, ngram_sizes=(2, 3)) -> TokenChunkRecord:
    name = f"Project Helios{index}"
    code = f"AX-{index:05d}"
    text = (
        f"Archive item {index}. {name} links identifier {code} to North Vale "
        f"and records revision {index % 17}."
    )
    token_ids = _token_ids(tokenizer, text)
    pieces = tokenizer.convert_ids_to_tokens(list(token_ids))
    normalized = tuple(
        value for value in (_normalize_piece(str(piece)) for piece in pieces) if value
    )
    aliases = tuple(
        dict.fromkeys((name.casefold(), code.casefold(), *_automatic_aliases(text)))
    )
    return TokenChunkRecord(
        reference_uri=f"memory://archive/{index}",
        chunk_id=f"archive:{index}",
        layer_id=0,
        token_ids=token_ids,
        normalized_tokens=normalized,
        bm25_terms=_word_terms(tokenizer.decode(list(token_ids))),
        aliases=aliases,
        token_start=0,
        token_end=len(token_ids),
        token_ngrams=_ngrams(token_ids, ngram_sizes),
        normalized_ngrams=_ngrams(normalized, ngram_sizes),
    )


def _build(tokenizer, count: int) -> tuple[TokenNativeIndex, float]:
    started = time.perf_counter()
    records = [_record(tokenizer, index) for index in range(count)]
    index = TokenNativeIndex(
        records,
        special_token_ids=getattr(tokenizer, "all_special_ids", ()),
    )
    return index, time.perf_counter() - started


def _rank(
    index,
    tokenizer,
    query: str,
    mode: str,
    *,
    indexed: bool,
    pool: int,
    stop_strategy: str = "idf",
):
    query_ids = _token_ids(tokenizer, query)
    semantic = torch.zeros(len(index.records), dtype=torch.float32)
    policy = HybridDiscoveryPolicy(
        mode=mode,
        indexed=indexed,
        candidate_pool_size=pool,
        enable_extended_channels=True,
        ngram_sizes=(2, 3),
        approximate_max_distance=1,
        stop_token_strategy=stop_strategy,
    )
    started = time.perf_counter()
    candidates = index.score(
        query_ids,
        semantic,
        tokenizer,
        policy,
        hop=1,
        parent_id="__root__",
        sparse=indexed,
    )
    duration = time.perf_counter() - started
    values = list(candidates.values()) if isinstance(candidates, dict) else candidates
    ranked = sorted(values, key=lambda row: (row.rank or len(values) + 1))
    return ranked, duration, dict(index.last_search_stats)


def _latency_study(tokenizer_name: str, tokenizer, sizes, queries, repeats, pool):
    rows = []
    for size in sizes:
        index, build_seconds = _build(tokenizer, size)
        extra_records = [_record(tokenizer, size + index) for index in range(64)]
        update_started = time.perf_counter()
        extended = index.extended(extra_records)
        rebuild_seconds = time.perf_counter() - update_started
        for mode in ("token_exact", "token_weighted", "token_approx", "cascade"):
            for query_index in queries:
                query = f"Find Project Helios{query_index} using AX-{query_index:05d}."
                measurements = {}
                rankings = {}
                stats = {}
                cold = {}
                for indexed in (False, True):
                    # Preserve the first query separately, then report warm medians.
                    _, cold_duration, _ = _rank(
                        index, tokenizer, query, mode, indexed=indexed, pool=pool
                    )
                    durations = []
                    ranked = None
                    local_stats = None
                    for _ in range(repeats):
                        ranked, duration, local_stats = _rank(
                            index, tokenizer, query, mode, indexed=indexed, pool=pool
                        )
                        durations.append(duration)
                    key = "indexed" if indexed else "exhaustive"
                    cold[key] = cold_duration
                    measurements[key] = statistics.median(durations)
                    rankings[key] = [row.chunk_id for row in ranked[:4]]
                    stats[key] = local_stats
                rows.append(
                    {
                        "tokenizer": tokenizer_name,
                        "corpus_chunks": size,
                        "mode": mode,
                        "query_target": query_index,
                        "build_ms": 1000 * build_seconds,
                        "rebuild_plus_64_ms": 1000 * rebuild_seconds,
                        "updated_index_chunks": len(extended.records),
                        "index_mib": index.memory_bytes() / (1024**2),
                        "exhaustive_ms": 1000 * measurements["exhaustive"],
                        "indexed_ms": 1000 * measurements["indexed"],
                        "exhaustive_cold_ms": 1000 * cold["exhaustive"],
                        "indexed_cold_ms": 1000 * cold["indexed"],
                        "speedup": measurements["exhaustive"]
                        / max(measurements["indexed"], 1e-12),
                        "indexed_candidate_rows": stats["indexed"]["candidate_rows"],
                        "indexed_candidate_fraction": stats["indexed"]["candidate_fraction"],
                        "top4_parity": int(rankings["exhaustive"] == rankings["indexed"]),
                        "target_top1": int(
                            rankings["indexed"][0] == f"archive:{query_index}"
                        ),
                    }
                )
    return rows


def _perturbations(index: int) -> dict[str, str]:
    return {
        "clean": f"Project Helios{index}",
        "case": f"PROJECT HELIOS{index}",
        "leading_space": f"   Project Helios{index}   ",
        "punctuation": f"Project, Helios{index}!",
        "hyphenation": f"Project-Helios{index}",
        "concatenation": f"ProjectHelios{index}",
        "split": f"Project Heli os{index}",
        "typo": f"Project Heliox{index}",
        "abbreviation": f"P. Helios{index}",
        "identifier_alias": f"AX-{index:05d}",
        "stop_insertion": f"the Project of Helios{index}",
        "unicode_fullwidth_nfkc": (
            "\uff30\uff52\uff4f\uff4a\uff45\uff43\uff54 "
            "\uff28\uff45\uff4c\uff49\uff4f\uff53" + str(index)
        ),
        "unicode_math_nfkc": (
            "\U0001d40f\U0001d42b\U0001d428\U0001d423\U0001d41e\U0001d41c\U0001d42d "
            "\U0001d407\U0001d41e\U0001d425\U0001d422\U0001d428\U0001d42c" + str(index)
        ),
    }


def _invariance_study(tokenizer_name: str, tokenizer, count: int, targets, pool):
    index, _ = _build(tokenizer, count)
    rows = []
    for target in targets:
        for perturbation, query in _perturbations(target).items():
            for mode in MODES:
                strategies = (
                    STOP_STRATEGIES
                    if mode in {"token_weighted", "cascade"}
                    else ("idf",)
                )
                for stop_strategy in strategies:
                    ranked, duration, stats = _rank(
                        index,
                        tokenizer,
                        query,
                        mode,
                        indexed=True,
                        pool=pool,
                        stop_strategy=stop_strategy,
                    )
                    target_id = f"archive:{target}"
                    rank = next(
                        (
                            position
                            for position, row in enumerate(ranked, 1)
                            if row.chunk_id == target_id
                        ),
                        len(ranked) + 1,
                    )
                    rows.append(
                        {
                            "tokenizer": tokenizer_name,
                            "target": target,
                            "perturbation": perturbation,
                            "mode": mode,
                            "stop_token_strategy": stop_strategy,
                            "rank": rank,
                            "top1": int(rank == 1),
                            "recall_at_4": int(rank <= 4),
                            "query_ms": 1000 * duration,
                            "candidate_rows": stats["candidate_rows"],
                        }
                    )
    return rows


def _plot_latency(rows, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for tokenizer in sorted({row["tokenizer"] for row in rows}):
        values = [row for row in rows if row["tokenizer"] == tokenizer]
        sizes = sorted({int(row["corpus_chunks"]) for row in values})
        exhaustive = [
            statistics.fmean(float(row["exhaustive_ms"]) for row in values if int(row["corpus_chunks"]) == size)
            for size in sizes
        ]
        indexed = [
            statistics.fmean(float(row["indexed_ms"]) for row in values if int(row["corpus_chunks"]) == size)
            for size in sizes
        ]
        axes[0].plot(sizes, exhaustive, marker="o", linestyle="--", label=f"{tokenizer} exhaustive")
        axes[0].plot(sizes, indexed, marker="o", label=f"{tokenizer} indexed")
        fractions = [
            statistics.fmean(float(row["indexed_candidate_fraction"]) for row in values if int(row["corpus_chunks"]) == size)
            for size in sizes
        ]
        axes[1].plot(sizes, fractions, marker="o", label=tokenizer)
    axes[0].set(xlabel="Indexed chunks", ylabel="Warm routing (ms)", title="Token-native query latency")
    axes[1].set(xlabel="Indexed chunks", ylabel="Scored candidate fraction", title="Posting-list narrowing")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"indexed_token_native_latency.{suffix}", dpi=190)
    plt.close(figure)


def _plot_invariance(rows, output: Path) -> None:
    perturbations = list(_perturbations(0))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for axis, tokenizer in zip(axes, sorted({row["tokenizer"] for row in rows})):
        values = [row for row in rows if row["tokenizer"] == tokenizer]
        for mode in MODES:
            means = [
                statistics.fmean(
                    float(row["recall_at_4"])
                    for row in values
                    if row["mode"] == mode and row["perturbation"] == perturbation
                )
                for perturbation in perturbations
            ]
            axis.plot(range(len(perturbations)), means, marker="o", label=mode)
        axis.set_title(tokenizer)
        axis.set_xticks(range(len(perturbations)), perturbations, rotation=55, ha="right")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Referent recall@4")
    axes[1].legend(fontsize=7, ncol=2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"tokenizer_invariance.{suffix}", dpi=190)
    plt.close(figure)


def _plot_stop_strategies(rows, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    perturbations = list(_perturbations(0))
    for axis, tokenizer in zip(axes, sorted({row["tokenizer"] for row in rows})):
        values = [
            row
            for row in rows
            if row["tokenizer"] == tokenizer and row["mode"] == "token_weighted"
        ]
        for strategy in STOP_STRATEGIES:
            recall = [
                statistics.fmean(
                    float(row["recall_at_4"])
                    for row in values
                    if row["stop_token_strategy"] == strategy
                    and row["perturbation"] == perturbation
                )
                for perturbation in perturbations
            ]
            axis.plot(range(len(perturbations)), recall, marker="o", label=strategy)
        axis.set_title(tokenizer)
        axis.set_xticks(range(len(perturbations)), perturbations, rotation=55, ha="right")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Referent recall@4")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"stop_token_strategies.{suffix}", dpi=190)
    plt.close(figure)


def _refresh_indexed_latency(args) -> dict:
    """Refresh only optimized indexed queries while freezing exhaustive rows."""
    path = args.output / "indexed_latency.csv"
    if not path.exists():
        raise FileNotFoundError("Run the complete latency benchmark before refreshing it.")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for name in args.tokenizers:
        model_id, revision = MODEL_SPECS[name]
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, local_files_only=args.local_files_only
        )
        for size in sorted({int(row["corpus_chunks"]) for row in rows if row["tokenizer"] == name}):
            index, _ = _build(tokenizer, size)
            for row in [
                value
                for value in rows
                if value["tokenizer"] == name and int(value["corpus_chunks"]) == size
            ]:
                query_index = int(row["query_target"])
                query = f"Find Project Helios{query_index} using AX-{query_index:05d}."
                _, cold, _ = _rank(
                    index,
                    tokenizer,
                    query,
                    row["mode"],
                    indexed=True,
                    pool=args.candidate_pool,
                )
                durations = [
                    _rank(
                        index,
                        tokenizer,
                        query,
                        row["mode"],
                        indexed=True,
                        pool=args.candidate_pool,
                    )[1]
                    for _ in range(args.repeats)
                ]
                warm = statistics.median(durations)
                row["indexed_cold_ms"] = 1000 * cold
                row["indexed_ms"] = 1000 * warm
                row["speedup"] = float(row["exhaustive_ms"]) / max(1000 * warm, 1e-12)
            print(f"[{name} refresh] corpus_chunks={size}", flush=True)
    _write_csv(path, rows)
    _plot_latency(rows, args.output)
    findings_path = args.output / "indexed_token_native_findings.json"
    findings = json.loads(findings_path.read_text(encoding="utf-8"))
    findings.update(
        mean_speedup=statistics.fmean(float(row["speedup"]) for row in rows),
        mean_candidate_fraction=statistics.fmean(
            float(row["indexed_candidate_fraction"]) for row in rows
        ),
        warm_repeats=args.repeats,
        optimized_indexed_refresh=True,
        system=_system_manifest(),
    )
    findings_path.write_text(
        json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8"
    )
    return findings


def run(args: argparse.Namespace) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    if args.refresh_indexed_only:
        return _refresh_indexed_latency(args)
    latency_rows = []
    invariance_rows = []
    manifests = {}
    for name in args.tokenizers:
        model_id, revision = MODEL_SPECS[name]
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, local_files_only=args.local_files_only
        )
        latency_rows.extend(
            _latency_study(
                name,
                tokenizer,
                args.sizes,
                args.query_targets,
                args.repeats,
                args.candidate_pool,
            )
        )
        invariance_rows.extend(
            _invariance_study(
                name,
                tokenizer,
                max(args.sizes),
                args.invariance_targets,
                args.candidate_pool,
            )
        )
        manifests[name] = {"model_id": model_id, "revision": revision}
        print(
            f"[{name}] latency_rows={len(latency_rows)} "
            f"invariance_rows={len(invariance_rows)}",
            flush=True,
        )
    _write_csv(args.output / "indexed_latency.csv", latency_rows)
    _write_csv(args.output / "tokenizer_invariance.csv", invariance_rows)
    _plot_latency(latency_rows, args.output)
    _plot_invariance(invariance_rows, args.output)
    _plot_stop_strategies(invariance_rows, args.output)
    findings = {
        "schema_version": "1.0",
        "tokenizers": manifests,
        "corpus_sizes": list(args.sizes),
        "candidate_pool": args.candidate_pool,
        "warm_repeats": args.repeats,
        "latency_rows": len(latency_rows),
        "invariance_rows": len(invariance_rows),
        "top4_parity": statistics.fmean(row["top4_parity"] for row in latency_rows),
        "target_top1": statistics.fmean(row["target_top1"] for row in latency_rows),
        "mean_speedup": statistics.fmean(row["speedup"] for row in latency_rows),
        "mean_candidate_fraction": statistics.fmean(
            row["indexed_candidate_fraction"] for row in latency_rows
        ),
        "stop_token_strategies": list(STOP_STRATEGIES),
        "update_contract": "measured immutable full-statistics rebuild after adding 64 chunks",
        "system": _system_manifest(),
    }
    (args.output / "indexed_token_native_findings.json").write_text(
        json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8"
    )
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizers", nargs="+", choices=tuple(MODEL_SPECS), default=tuple(MODEL_SPECS))
    parser.add_argument("--sizes", nargs="+", type=int, default=(256, 1024, 4096))
    parser.add_argument("--query-targets", nargs="+", type=int, default=(7, 41, 173, 239))
    parser.add_argument("--invariance-targets", nargs="+", type=int, default=(7, 41, 173, 239, 251))
    parser.add_argument("--candidate-pool", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--refresh-indexed-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_6_hybrid_pra/indexed_confirmation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
