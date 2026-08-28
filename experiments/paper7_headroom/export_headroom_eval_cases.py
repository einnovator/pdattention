"""Export focused cases through Headroom's released evaluation loaders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from headroom.evals.datasets import load_hotpotqa, load_msmarco, load_tool_output_samples
from headroom.evals.runners.compression_only import CompressionOnlyRunner


def _row(dataset: str, case: Any) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "case_id": str(case.id),
        "context": str(case.context),
        "query": str(case.query),
        "ground_truth": str(case.ground_truth),
        "metadata": dict(case.metadata),
    }


def _load_msmarco_streaming(n: int) -> list[dict[str, Any]]:
    """Reproduce Headroom's MS MARCO adapter over a streaming HF split."""
    from datasets import load_dataset

    rows: list[dict[str, Any]] = []
    dataset = load_dataset("microsoft/ms_marco", "v2.1", split="validation", streaming=True)
    for index, item in enumerate(dataset):
        passages = item.get("passages", {})
        passage_texts = passages.get("passage_text", [])
        selected = passages.get("is_selected", [])
        query = item.get("query", "")
        if not passage_texts or not query:
            continue
        context = "\n\n".join(
            f"{'[RELEVANT] ' if relevant else ''}Passage {offset + 1}: {text}"
            for offset, (text, relevant) in enumerate(zip(passage_texts, selected))
        )
        answers = item.get("answers", [])
        evidence_target = next(
            (text for text, relevant in zip(passage_texts, selected) if relevant),
            str(answers[0]) if answers else "",
        )
        rows.append({
            "dataset": "msmarco",
            "case_id": f"msmarco_{index}",
            "context": context,
            "query": query,
            "ground_truth": str(answers[0]) if answers else "None",
            "evidence_target": evidence_target,
            "metadata": {
                "source": "MS_MARCO",
                "query_type": item.get("query_type", "unknown"),
                "num_passages": len(passage_texts),
                "loader_path": "headroom_adapter_streaming_fallback",
            },
        })
        if len(rows) >= n:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()
    rows = [_row("tool_outputs", case) for case in load_tool_output_samples().cases]

    needle_runner = CompressionOnlyRunner()
    for case in needle_runner.generate_ccr_test_cases(args.n):
        needle = str(case["needles"][0])
        rows.append({
            "dataset": "ccr_needle",
            "case_id": str(case["id"]),
            "context": str(case["content"]),
            "query": f"Find the anomalous error code {needle}.",
            "ground_truth": needle,
            "metadata": {"source": "official_ccr_generator", "needles": case["needles"]},
        })

    failures: list[dict[str, str]] = []
    for name, loader in (("hotpotqa", load_hotpotqa), ("msmarco", load_msmarco)):
        try:
            suite = loader(n=args.n)
            rows.extend(_row(name, case) for case in suite.cases)
        except Exception as exc:  # Official loader incompatibility is an artifact, not a silent skip.
            if name == "msmarco":
                rows.extend(_load_msmarco_streaming(args.n))
                failures.append({
                    "dataset": name,
                    "error": f"official eager loader failed ({type(exc).__name__}: {exc}); "
                    "used a streaming reconstruction of the released adapter",
                    "recovered": "true",
                })
            else:
                failures.append({"dataset": name, "error": f"{type(exc).__name__}: {exc}"})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"rows": rows, "failures": failures}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} cases; loader failures={len(failures)}")


if __name__ == "__main__":
    main()
