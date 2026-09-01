"""Explain the measured AirLLM E0/E2 request-path penalty."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping


def _mean(rows: Iterable[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values)


def analyze(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(payload["rows"])
    full = [row for row in rows if row["condition"] == "full_context_e0"]
    selected = [row for row in rows if row["condition"] == "selected_text_e0"]
    native = [row for row in rows if row["condition"] == "native_pra_e2"]
    selected_warm = [row for row in selected if int(row["repeat"]) > 0]
    native_warm = [row for row in native if int(row["repeat"]) > 0]
    encode = [
        float(row["reference_encode_seconds"])
        for row in native
        if row.get("reference_encode_seconds")
    ]
    condition_rows = []
    for name, subset in (
        ("FULL E0", full),
        ("selected E0", selected_warm),
        ("native E2", native_warm),
    ):
        condition_rows.append(
            {
                "condition": name,
                "samples": len(subset),
                "visible_tokens": _mean(subset, "visible_prompt_tokens"),
                "native_tokens": _mean(subset, "selected_native_kv_tokens"),
                "ttft_ms": _mean(subset, "ttft_ms"),
                "itl_ms": _mean(subset, "itl_ms"),
                "completion_seconds": _mean(subset, "completion_seconds"),
                "peak_cuda_mib": _mean(subset, "peak_cuda_bytes") / 2**20,
            }
        )
    selected_row = condition_rows[1]
    native_row = condition_rows[2]
    request_delta = (
        native_row["completion_seconds"] - selected_row["completion_seconds"]
    )
    pooled = {
        "e2_over_e0_ttft": native_row["ttft_ms"] / selected_row["ttft_ms"],
        "e2_over_e0_itl": native_row["itl_ms"] / selected_row["itl_ms"],
        "e2_over_e0_completion": (
            native_row["completion_seconds"] / selected_row["completion_seconds"]
        ),
        "e2_minus_e0_ttft_ms": native_row["ttft_ms"] - selected_row["ttft_ms"],
        "e2_minus_e0_itl_ms": native_row["itl_ms"] - selected_row["itl_ms"],
        "e2_minus_e0_completion_seconds": request_delta,
        "e2_minus_e0_peak_cuda_mib": (
            native_row["peak_cuda_mib"] - selected_row["peak_cuda_mib"]
        ),
        "full_over_selected_visible_tokens": (
            condition_rows[0]["visible_tokens"] / selected_row["visible_tokens"]
        ),
        "full_over_selected_ttft": condition_rows[0]["ttft_ms"]
        / selected_row["ttft_ms"],
        "mean_reference_encode_seconds": statistics.fmean(encode),
        "reuse_break_even_queries": (
            None
            if request_delta >= 0
            else statistics.fmean(encode) / -request_delta
        ),
    }
    return {
        "schema_version": "paper6.6-airllm-request-path-forensics-v1",
        "source_schema_version": payload["schema_version"],
        "evidence_tier": "MEASURED_TRACE_PLUS_CODE_PATH_ATTRIBUTION",
        "model_id": payload["model_id"],
        "device": payload["device"],
        "condition_rows": condition_rows,
        "pooled": pooled,
        "findings": [
            {
                "mechanism": "query_encoding_pass",
                "status": "CODE_PATH_CONFIRMED_TIMING_NOT_YET_ISOLATED",
                "evidence": (
                    "Routed E2 calls _route_once before model.generate; AirLLM streams "
                    "the model weights for both passes."
                ),
            },
            {
                "mechanism": "attention_dispatch_change",
                "status": "CODE_PATH_CONFIRMED_TIMING_NOT_YET_ISOLATED",
                "evidence": (
                    "wrap_airllm_hf_model changes the model attention implementation "
                    "from SDPA to eager before native consumption."
                ),
            },
            {
                "mechanism": "late_layer_memory_attention",
                "status": "MEASURED_COMBINED_EFFECT",
                "evidence": (
                    "ITL remains higher after routing is complete, consistent with "
                    "four late direct-plus-memory consumers and eager dispatch."
                ),
            },
            {
                "mechanism": "reference_encoding",
                "status": "MEASURED_OUTSIDE_REQUEST",
                "evidence": (
                    "Encoding is a separate one-time cost and cannot explain the "
                    "persistent warm request-path penalty."
                ),
            },
        ],
        "favorable_workload_hypothesis": {
            "properties": [
                "selection is explicit or computed without another streamed model pass",
                "one immutable long reference serves many queries",
                "answers are short",
                "one or two late consumer layers preserve quality",
                "selected text is long enough that E0 prefill or sequential K/V pressure dominates",
                "native detail remains host-resident and is prefetched with the corresponding weight layer",
            ],
            "current_trace_has_finite_break_even": pooled["reuse_break_even_queries"]
            is not None,
        },
    }


def _write_table(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Condition & visible & native & TTFT (ms) & ITL (ms) & completion (s) & peak MiB \\",
        r"\midrule",
    ]
    for row in report["condition_rows"]:
        lines.append(
            "{} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.2f} & {:.1f} \\\\".format(
                row["condition"],
                row["visible_tokens"],
                row["native_tokens"],
                row["ttft_ms"],
                row["itl_ms"],
                row["completion_seconds"],
                row["peak_cuda_mib"],
            )
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(report: Mapping[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = report["condition_rows"]
    labels = [row["condition"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.1))
    axes[0].bar(labels, [row["ttft_ms"] / 1000.0 for row in rows], color="#4c78a8")
    axes[0].set_ylabel("TTFT (s)")
    axes[0].set_title("First-token path")
    counts = (1, 2, 4, 8, 16, 32)
    e0 = rows[1]["completion_seconds"]
    e2 = rows[2]["completion_seconds"]
    encode = report["pooled"]["mean_reference_encode_seconds"]
    axes[1].plot(counts, [count * e0 for count in counts], marker="o", label="E0")
    axes[1].plot(
        counts,
        [encode + count * e2 for count in counts],
        marker="s",
        label="E2 + one encode",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("queries reusing one reference")
    axes[1].set_ylabel("cumulative time (s)")
    axes[1].set_title("Measured reuse economics")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = analyze(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "request_path_forensics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _write_table(
        report, args.output_dir / "generated_request_path_forensics_table.tex"
    )
    _plot(report, args.output_dir / "request_path_forensics.png")


if __name__ == "__main__":
    main()
