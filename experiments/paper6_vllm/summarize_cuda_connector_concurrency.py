"""Reduce the CUDA connector concurrency and isolation experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


_LABELS = {
    "shared_native": "shared native",
    "mixed_native_ordinary": "mixed native/ordinary",
    "wrong_native": "wrong-memory control",
}


def summarize(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve causal recovery, leakage, throughput, and memory separately."""

    rows = []
    for raw in payload["rows"]:
        expected = int(raw["expected_requests"])
        requests = int(raw["requests"])
        rows.append(
            {
                "condition": str(raw["condition"]),
                "concurrency": int(raw["concurrency"]),
                "requests": requests,
                "recovery_rate": (
                    int(raw["expected_recoveries"]) / expected
                    if expected
                    else None
                ),
                "leakage_rate": int(raw["forbidden_leaks"]) / max(requests, 1),
                "requests_per_second": float(raw["requests_per_second"]),
                "output_tokens_per_second": float(raw["output_tokens_per_second"]),
                "completion_ms": float(raw["completion_ms"]),
                "peak_allocated_mib": int(raw["peak_allocated_bytes"]) / 2**20,
                "peak_reserved_mib": int(raw["peak_reserved_bytes"]) / 2**20,
            }
        )
    return {
        "schema_version": "paper6-vllm-cuda-connector-concurrency-summary-v1",
        "source_schema_version": payload["schema_version"],
        "evidence_tier": payload["evidence_tier"],
        "integration_status": payload["integration_status"],
        "engine_version": payload["engine_version"],
        "model_id": payload["model_id"],
        "device": payload["device"],
        "source_tokens": int(payload["source_tokens"]),
        "source_content_visible": bool(payload["source_content_visible"]),
        "source_slots_scheduler_visible": bool(
            payload["source_slots_scheduler_visible"]
        ),
        "rows": rows,
        "total_requests": sum(row["requests"] for row in rows),
        "all_expected_recovered": all(
            row["recovery_rate"] in (None, 1.0) for row in rows
        ),
        "total_forbidden_leaks": sum(
            int(raw["forbidden_leaks"]) for raw in payload["rows"]
        ),
    }


def render_table(summary: Mapping[str, Any]) -> str:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Condition & $C$ & Recovery & Leakage & requests/s & Peak MiB \\",
        r"\midrule",
    ]
    for row in summary["rows"]:
        recovery = row["recovery_rate"]
        recovery_text = "--" if recovery is None else f"{recovery:.3f}"
        lines.append(
            f"{_LABELS.get(row['condition'], row['condition'])} & "
            f"{row['concurrency']} & {recovery_text} & "
            f"{row['leakage_rate']:.3f} & {row['requests_per_second']:.1f} & "
            f"{row['peak_allocated_mib']:.0f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def render_plot(summary: Mapping[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6.8, 3.8))
    colors = ("#0072B2", "#E69F00", "#009E73")
    for color, condition in zip(colors, _LABELS):
        rows = [row for row in summary["rows"] if row["condition"] == condition]
        rows.sort(key=lambda row: row["concurrency"])
        axis.plot(
            [row["concurrency"] for row in rows],
            [row["requests_per_second"] for row in rows],
            marker="o",
            linewidth=2,
            color=color,
            label=_LABELS[condition],
        )
    axis.set_xlabel("Concurrent requests")
    axis.set_ylabel("Requests/s")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--plot", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    summary = summarize(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.table.write_text(render_table(summary), encoding="utf-8")
    render_plot(summary, args.plot)


if __name__ == "__main__":
    main()
