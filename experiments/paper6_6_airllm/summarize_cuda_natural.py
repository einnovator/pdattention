"""Reduce the selector-frozen AirLLM CUDA natural-QA benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _index(
    aggregates: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    return {
        (str(row["dataset"]), str(row["condition"]), str(row["regime"])): row
        for row in aggregates
    }


def summarize(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Compare selected-text E0 and native E2 without changing selection."""

    if payload.get("status") != "COMPLETE":
        raise ValueError("AirLLM natural benchmark is not complete.")
    aggregates = list(payload.get("aggregates") or ())
    indexed = _index(aggregates)
    output_by_pair: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in payload.get("rows", ()):
        if row.get("condition") not in ("selected_text_e0", "native_pra_e2"):
            continue
        key = (str(row["dataset"]), str(row["example_id"]), int(row["repeat"]))
        output_by_pair.setdefault(key, {})[str(row["condition"])] = str(
            row["output_text"]
        )
    datasets = sorted({str(row["dataset"]) for row in aggregates})
    comparisons: list[dict[str, Any]] = []
    for dataset in datasets:
        for regime in ("cold_one_shot", "warm_repeated"):
            e0 = indexed.get((dataset, "selected_text_e0", regime))
            e2 = indexed.get((dataset, "native_pra_e2", regime))
            if e0 is None or e2 is None:
                continue
            e0_seconds = float(e0["mean_completion_seconds"])
            e2_seconds = float(e2["mean_completion_seconds"])
            e0_ttft = e0.get("mean_ttft_ms")
            e2_ttft = e2.get("mean_ttft_ms")
            e0_itl = e0.get("mean_itl_ms")
            e2_itl = e2.get("mean_itl_ms")
            selected_tokens = float(e0["mean_visible_prompt_tokens"])
            visible_tokens = float(e2["mean_visible_prompt_tokens"])
            repeat_is_warm = regime == "warm_repeated"
            pairs = [
                values
                for (pair_dataset, _example_id, repeat), values in output_by_pair.items()
                if pair_dataset == dataset
                and (repeat > 0) == repeat_is_warm
                and "selected_text_e0" in values
                and "native_pra_e2" in values
            ]
            comparisons.append(
                {
                    "dataset": dataset,
                    "regime": regime,
                    "samples_per_condition": min(int(e0["samples"]), int(e2["samples"])),
                    "e0_token_f1": float(e0["mean_token_f1"]),
                    "e2_token_f1": float(e2["mean_token_f1"]),
                    "token_f1_delta": float(e2["mean_token_f1"])
                    - float(e0["mean_token_f1"]),
                    "paired_output_count": len(pairs),
                    "exact_output_pair_parity": (
                        sum(
                            pair["selected_text_e0"] == pair["native_pra_e2"]
                            for pair in pairs
                        )
                        / len(pairs)
                        if pairs
                        else None
                    ),
                    "e0_answer_containment": float(e0["mean_answer_containment"]),
                    "e2_answer_containment": float(e2["mean_answer_containment"]),
                    "e0_completion_seconds": e0_seconds,
                    "e2_completion_seconds": e2_seconds,
                    "e2_over_e0_completion": e2_seconds / max(e0_seconds, 1e-12),
                    "e0_ttft_ms": None if e0_ttft is None else float(e0_ttft),
                    "e2_ttft_ms": None if e2_ttft is None else float(e2_ttft),
                    "e0_ttft_p95_ms": (
                        None if e0.get("ttft_ms") is None else float(e0["ttft_ms"]["p95"])
                    ),
                    "e2_ttft_p95_ms": (
                        None if e2.get("ttft_ms") is None else float(e2["ttft_ms"]["p95"])
                    ),
                    "e2_over_e0_ttft": (
                        None
                        if e0_ttft is None or e2_ttft is None
                        else float(e2_ttft) / max(float(e0_ttft), 1e-12)
                    ),
                    "e0_itl_ms": None if e0_itl is None else float(e0_itl),
                    "e2_itl_ms": None if e2_itl is None else float(e2_itl),
                    "e0_completion_p95_seconds": (
                        None
                        if e0.get("completion_seconds") is None
                        else float(e0["completion_seconds"]["p95"])
                    ),
                    "e2_completion_p95_seconds": (
                        None
                        if e2.get("completion_seconds") is None
                        else float(e2["completion_seconds"]["p95"])
                    ),
                    "e0_visible_tokens": selected_tokens,
                    "e2_visible_tokens": visible_tokens,
                    "e2_native_tokens": float(e2["mean_native_kv_tokens"]),
                    "visible_token_reduction": 1.0
                    - visible_tokens / max(selected_tokens, 1.0),
                    "e0_peak_cuda_bytes": int(e0["peak_cuda_bytes"]),
                    "e2_peak_cuda_bytes": int(e2["peak_cuda_bytes"]),
                }
            )
    cold_encodes = [
        float(row["reference_encode_seconds"])
        for row in payload.get("rows", ())
        if row.get("condition") == "native_pra_e2"
        and row.get("regime") == "cold_one_shot"
        and row.get("reference_encode_seconds") is not None
    ]
    return {
        "schema_version": "paper6.6-airllm-cuda-natural-summary-v1",
        "source_schema_version": payload.get("schema_version"),
        "evidence_tier": payload.get("evidence_tier"),
        "selector_frozen": bool(payload.get("selector_frozen")),
        "model_id": payload.get("model_id"),
        "device": payload.get("device"),
        "example_count": len(
            {
                (str(row["dataset"]), str(row["example_id"]))
                for row in payload.get("rows", ())
            }
        ),
        "mean_reference_encode_seconds": (
            sum(cold_encodes) / len(cold_encodes) if cold_encodes else None
        ),
        "comparisons": comparisons,
    }


def render_table(summary: Mapping[str, Any]) -> str:
    """Render compact, generated evidence for the AirLLM manuscript."""

    names = {
        "qasper": "QASPER",
        "hotpotqa": "HotpotQA",
        "2wikimultihopqa": "2Wiki",
    }
    regimes = {"cold_one_shot": "cold", "warm_repeated": "warm"}
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dataset & State & E0 F1 & E2 F1 & Pair exact & Completion & TTFT & Visible $\downarrow$ \\",
        r"\midrule",
    ]
    for row in summary["comparisons"]:
        ttft = row.get("e2_over_e0_ttft")
        ttft_text = "--" if ttft is None else f"{ttft:.3f}"
        lines.append(
            f"{names.get(row['dataset'], row['dataset'])} & "
            f"{regimes[row['regime']]} & "
            f"{row['e0_token_f1']:.3f} & {row['e2_token_f1']:.3f} & "
            f"{row['exact_output_pair_parity']:.3f} & "
            f"{row['e2_over_e0_completion']:.3f} & {ttft_text} & "
            f"{100.0 * row['visible_token_reduction']:.1f}\\% \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--table", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    summary = summarize(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(render_table(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
