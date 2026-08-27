"""Benchmark Paper 7 native-index size gates and lazy selected-region encoding."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pra_hf.adaptive_context_runtime import AdaptiveContextRuntime, ContextPolicy
from pra_hf.context_records import RecordType
from pra_hf.context_store import RecordScope
from pra_hf.progressive_context import NativeIndexState, ProgressiveContextRuntime


OUTPUT = ROOT / "docs/papers/shared/results/paper7_records/native_index_size_gate"
SIZES = (1_024, 4_096, 16_384, 65_536, 262_144)
SEEDS = (11, 23, 37, 53, 71)
NATIVE_LIMIT = 4_096
NATIVE_BYTE_LIMIT = 65_536
LINE_TOKENS = 32
CONDITIONS = (
    "FULL_BODY_NATIVE",
    "SIZE_GATED_CHEAP",
    "SIZE_GATED_LAZY_NATIVE",
    "CURSOR_SEARCH_ONLY",
)
PROTOCOL = "paper7-native-index-size-gate-v1"


class _WhitespaceTokenizer:
    """Deterministic tokenizer used by the bounded systems mechanism profile."""

    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        values = text.split()
        ids = [
            int.from_bytes(hashlib.blake2b(value.encode(), digest_size=4).digest(), "little")
            for value in values
        ]
        return SimpleNamespace(input_ids=ids)


class _InstrumentedNativeEncoder:
    """Fixed-width Torch encoder exposing the PRA reference API for scaling tests.

    It performs real blockwise embedding/projection work but is not a language
    model. The resulting timings characterize the SDK lifecycle and scaling
    safeguard, not Qwen inference latency or routing quality.
    """

    def __init__(self, device: torch.device, seed: int, width: int = 32) -> None:
        self.device = device
        self.width = width
        self.tokenizer = _WhitespaceTokenizer()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        table = torch.randn(8_192, width, generator=generator, dtype=torch.float32)
        projection = torch.randn(width, width, generator=generator, dtype=torch.float32)
        self.table = table.to(device)
        self.projection = projection.to(device)
        self.references: dict[str, dict[str, int]] = {}

    def add_reference(self, reference: str, *, text: str):
        token_ids = self.tokenizer(text).input_ids
        chunk_count = max(1, (len(token_ids) + 63) // 64)
        checksum = torch.zeros(self.width, device=self.device)
        with torch.inference_mode():
            for start in range(0, len(token_ids), 4_096):
                ids = torch.tensor(
                    token_ids[start : start + 4_096], device=self.device, dtype=torch.long
                ) % self.table.shape[0]
                hidden = torch.tanh(self.table[ids] @ self.projection)
                checksum += hidden.sum(dim=0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.references[reference] = {
            "tokens": len(token_ids),
            "chunks": chunk_count,
            "checksum": int(float(checksum[0].cpu()) * 1_000) if token_ids else 0,
        }
        return SimpleNamespace(
            id=reference,
            uri=reference,
            tokens=len(token_ids),
            chunks=chunk_count,
        )

    def remove_reference(self, handle) -> None:
        self.references.pop(handle.uri, None)

    def stats(self) -> dict[str, int]:
        tokens = sum(row["tokens"] for row in self.references.values())
        chunks = sum(row["chunks"] for row in self.references.values())
        return {
            "routing_index_bytes": chunks * self.width * 4,
            "resident_detail_kv_bytes": tokens * self.width * 2 * 4,
        }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _payload(target_tokens: int, seed: int) -> tuple[str, str, int]:
    marker = f"EVIDENCE_KEY_{seed} ANSWER_CODE_ZX_{seed}"
    values = [f"filler_{index % 997}" for index in range(target_tokens)]
    marker_start = min(target_tokens - 2, (target_tokens * 3 // 4) // LINE_TOKENS * LINE_TOKENS)
    values[marker_start : marker_start + 2] = marker.split()
    lines = [
        " ".join(values[start : start + LINE_TOKENS])
        for start in range(0, target_tokens, LINE_TOKENS)
    ]
    return "\n".join(lines), marker, marker_start // LINE_TOKENS


def _cheap_index_bytes(record) -> int:
    return len(json.dumps(record.address_views(), sort_keys=True, default=str).encode("utf-8"))


def _limits(condition: str) -> tuple[int | None, int | None, bool]:
    if condition == "FULL_BODY_NATIVE":
        return None, None, False
    if condition == "CURSOR_SEARCH_ONLY":
        return 0, 0, False
    return NATIVE_LIMIT, NATIVE_BYTE_LIMIT, False


def _run_one(
    *,
    size: int,
    seed: int,
    condition: str,
    device: torch.device,
) -> dict[str, object]:
    payload, marker, marker_line = _payload(size, seed)
    token_limit, byte_limit, deferred = _limits(condition)
    with tempfile.TemporaryDirectory(prefix="paper7-size-gate-") as store_dir:
        policy = ContextPolicy(
            local_store=store_dir,
            max_native_index_tokens=token_limit,
            max_native_index_bytes=byte_limit,
            defer_native_index=deferred,
        )
        runtime = AdaptiveContextRuntime(
            RecordScope("paper7-size-gate", f"{condition}-{size}-{seed}"),
            policy,
        )
        model = _InstrumentedNativeEncoder(device, seed)
        progressive = ProgressiveContextRuntime(runtime, pra_model=model, chunk_tokens=64)

        arrived = time.perf_counter()
        record = progressive.ingest(
            payload,
            record_type=RecordType.TERMINAL_OUTPUT,
            provenance={"experiment": PROTOCOL, "seed": seed},
        )
        compact_ready = time.perf_counter()
        audit = progressive.prepare_native_index(record.record_id)
        index_ready = time.perf_counter()
        ttuc_ms = (index_ready - arrived) * 1_000.0

        search_started = time.perf_counter()
        search = runtime.search_record(record.record_id, marker, limit=1)
        cheap_search_ms = (time.perf_counter() - search_started) * 1_000.0
        matches = search.payload["matches"]
        evidence_recall = int(bool(matches) and marker in str(matches[0]))
        selected_region_tokens = 0
        selected_region_ms = 0.0
        lazy_regions = 0
        if (
            condition == "SIZE_GATED_LAZY_NATIVE"
            and audit.native_index_state == NativeIndexState.SKIPPED_SIZE_LIMIT
        ):
            match_indices = search.payload["match_indices"]
            selected_line = int(match_indices[0]) if match_indices else marker_line
            region = progressive.encode_selected_region_native(
                record.record_id, {"lines": [selected_line, selected_line + 1]}
            )
            selected_region_tokens = region.native_tokens
            selected_region_ms = region.latency_ms
            lazy_regions = 1

        stats = model.stats()
        effective_mode = (
            "PRA_NATIVE"
            if audit.native_index_built
            else (
                "PRA_SIZE_GATED_LAZY"
                if lazy_regions
                else "PRA_SIZE_GATED"
            )
        )
        compact_tokens = len(model.tokenizer(json.dumps(record.compact_view())).input_ids)
        active_kv_tokens = selected_region_tokens or compact_tokens
        return {
            "protocol": PROTOCOL,
            "seed": seed,
            "payload_tokens": size,
            "payload_bytes": record.backing.size_bytes,
            "record_type": record.record_type.value,
            "condition": condition,
            "effective_mode": effective_mode,
            "native_index_state": audit.native_index_state.value,
            "native_index_requested": int(audit.native_index_requested),
            "native_index_built": int(audit.native_index_built),
            "native_index_skipped_reason": audit.native_index_skipped_reason or "",
            "native_index_tokens": audit.native_index_tokens,
            "native_index_bytes": audit.native_index_bytes,
            "native_index_latency_ms": audit.native_index_latency_ms,
            "agent_compression_ttuc_ms": (compact_ready - arrived) * 1_000.0,
            "time_to_usable_context_ms": ttuc_ms,
            "cheap_index_modes_built": ";".join(audit.cheap_index_modes_built),
            "cheap_index_bytes": _cheap_index_bytes(record),
            "cheap_search_latency_ms": cheap_search_ms,
            "selected_region_native_latency_ms": selected_region_ms,
            "selected_region_tokens": selected_region_tokens,
            "lazy_native_regions_encoded": lazy_regions,
            "evidence_recall": evidence_recall,
            "active_kv_tokens": active_kv_tokens,
            "routing_index_bytes": stats["routing_index_bytes"],
            "resident_detail_kv_bytes": stats["resident_detail_kv_bytes"],
            "search_available": 1,
            "cursor_available": 1,
            "full_materialization_available": 1,
        }


def _summaries(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["payload_tokens"]), str(row["condition"]))].append(row)
    output = []
    metrics = (
        "native_index_latency_ms",
        "agent_compression_ttuc_ms",
        "time_to_usable_context_ms",
        "cheap_search_latency_ms",
        "selected_region_native_latency_ms",
        "selected_region_tokens",
        "cheap_index_bytes",
        "routing_index_bytes",
        "resident_detail_kv_bytes",
        "active_kv_tokens",
        "evidence_recall",
    )
    for (size, condition), values in sorted(grouped.items()):
        result: dict[str, object] = {
            "payload_tokens": size,
            "condition": condition,
            "runs": len(values),
            "native_index_state": values[0]["native_index_state"],
        }
        for metric in metrics:
            samples = [float(row[metric]) for row in values]
            result[f"{metric}_median"] = statistics.median(samples)
            result[f"{metric}_mean"] = statistics.fmean(samples)
        output.append(result)
    return output


def _plot(summary: Sequence[Mapping[str, object]], output: Path) -> None:
    labels = {
        "FULL_BODY_NATIVE": "Eager full native",
        "SIZE_GATED_CHEAP": "Size-gated cheap",
        "SIZE_GATED_LAZY_NATIVE": "Size-gated lazy native",
        "CURSOR_SEARCH_ONLY": "Search/cursor only",
    }
    colors = {
        "FULL_BODY_NATIVE": "#b42318",
        "SIZE_GATED_CHEAP": "#155eef",
        "SIZE_GATED_LAZY_NATIVE": "#087443",
        "CURSOR_SEARCH_ONLY": "#6941c6",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for condition in CONDITIONS:
        values = [row for row in summary if row["condition"] == condition]
        x = [int(row["payload_tokens"]) for row in values]
        axes[0].plot(
            x,
            [float(row["native_index_latency_ms_median"]) for row in values],
            marker="o",
            label=labels[condition],
            color=colors[condition],
        )
        axes[1].plot(
            x,
            [float(row["time_to_usable_context_ms_median"]) for row in values],
            marker="o",
            label=labels[condition],
            color=colors[condition],
        )
    for axis, title, ylabel in (
        (axes[0], "Full-body native-index work", "Index latency (ms)"),
        (axes[1], "Compact-first availability", "TTUC (ms)"),
    ):
        axis.set_xscale("log", base=2)
        axis.set_yscale("symlog", linthresh=0.1)
        axis.set_xlabel("Backing payload tokens")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"native_index_ingestion_ttuc.{suffix}", dpi=180)
    plt.close(fig)


def _write_macros(summary: Sequence[Mapping[str, object]], output: Path) -> None:
    largest = max(int(row["payload_tokens"]) for row in summary)
    by_key = {(int(row["payload_tokens"]), str(row["condition"])): row for row in summary}
    eager = by_key[(largest, "FULL_BODY_NATIVE")]
    gated = by_key[(largest, "SIZE_GATED_CHEAP")]
    lazy = by_key[(largest, "SIZE_GATED_LAZY_NATIVE")]
    eager_ttuc = float(eager["time_to_usable_context_ms_median"])
    gated_ttuc = float(gated["time_to_usable_context_ms_median"])
    macros = (
        "% Generated by run_native_index_size_gate.py; do not edit.\n"
        f"\\newcommand{{\\PaperSevenSizeGateSeeds}}{{{len(SEEDS)}}}\n"
        f"\\newcommand{{\\PaperSevenSizeGateLimit}}{{{NATIVE_LIMIT:,}}}\n"
        f"\\newcommand{{\\PaperSevenSizeGateByteLimit}}{{{NATIVE_BYTE_LIMIT:,}}}\n"
        f"\\newcommand{{\\PaperSevenLargestPayload}}{{{largest:,}}}\n"
        f"\\newcommand{{\\PaperSevenEagerLargestIndexMs}}{{{float(eager['native_index_latency_ms_median']):.1f}}}\n"
        f"\\newcommand{{\\PaperSevenEagerLargestTTUC}}{{{eager_ttuc:.1f}}}\n"
        f"\\newcommand{{\\PaperSevenGatedLargestTTUC}}{{{gated_ttuc:.1f}}}\n"
        f"\\newcommand{{\\PaperSevenTTUCReduction}}{{{eager_ttuc / max(gated_ttuc, 1e-9):.1f}}}\n"
        f"\\newcommand{{\\PaperSevenLazyRegionTokens}}{{{float(lazy['selected_region_tokens_median']):.0f}}}\n"
        f"\\newcommand{{\\PaperSevenLazyRegionMs}}{{{float(lazy['selected_region_native_latency_ms_median']):.2f}}}\n"
        f"\\newcommand{{\\PaperSevenGatedRecall}}{{{float(lazy['evidence_recall_mean']):.3f}}}\n"
    )
    (output / "generated_native_index_size_gate_results.tex").write_text(
        macros, encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sizes", type=int, nargs="*", default=SIZES)
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows = []
    for size in args.sizes:
        for seed in args.seeds:
            for condition in CONDITIONS:
                row = _run_one(
                    size=size,
                    seed=seed,
                    condition=condition,
                    device=device,
                )
                rows.append(row)
                print(
                    f"{size:>7} seed={seed} {condition:<24} "
                    f"state={row['native_index_state']:<19} "
                    f"ttuc={row['time_to_usable_context_ms']:.1f}ms",
                    flush=True,
                )
    summary = _summaries(rows)
    _write_csv(output / "native_index_size_gate_results.csv", rows)
    _write_csv(output / "ingestion_latency_by_size.csv", summary)
    _write_csv(output / "time_to_usable_context.csv", [
        {
            "payload_tokens": row["payload_tokens"],
            "condition": row["condition"],
            "ttuc_ms_median": row["time_to_usable_context_ms_median"],
            "compact_ready_ms_median": row["agent_compression_ttuc_ms_median"],
        }
        for row in summary
    ])
    _write_csv(output / "lazy_native_region_results.csv", [
        row for row in rows if int(row["lazy_native_regions_encoded"])
    ])
    _write_csv(output / "oversized_record_policy_results.csv", [
        row for row in summary if int(row["payload_tokens"]) > NATIVE_LIMIT
    ])
    _plot(summary, output)
    _write_macros(summary, output)
    manifest = {
        "protocol": PROTOCOL,
        "device": str(device),
        "torch_version": torch.__version__,
        "sizes": list(args.sizes),
        "seeds": list(args.seeds),
        "conditions": list(CONDITIONS),
        "native_index_token_limit": NATIVE_LIMIT,
        "native_index_byte_limit": NATIVE_BYTE_LIMIT,
        "line_tokens": LINE_TOKENS,
        "encoder": "instrumented fixed-width Torch embedding/projection; not a pretrained LM",
        "claims": "SDK lifecycle and scaling safeguard only; no model-quality or Qwen latency claim",
    }
    (output / "native_index_size_gate_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
