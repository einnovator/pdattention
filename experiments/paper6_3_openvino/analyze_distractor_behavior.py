"""Explain OpenVINO distractor effects with paired lexical forensics."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.paper6_3_openvino.run_distractor_ablation import (
    _terms,
    ranked_distractors,
)


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _pearson(left: Iterable[float], right: Iterable[float]) -> float | None:
    x = list(map(float, left))
    y = list(map(float, right))
    if len(x) < 2 or len(x) != len(y):
        return None
    mx = statistics.fmean(x)
    my = statistics.fmean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)
    )
    return None if denominator == 0 else numerator / denominator


def _entry_index(manifest: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (str(entry["dataset"]), str(entry["example_id"])): entry
        for entry in manifest["entries"]
    }


def analyze(
    payload: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    entries = _entry_index(manifest)
    baselines = {
        (str(row["dataset"]), str(row["example_id"])): row
        for row in payload["rows"]
        if row["condition"] == "evidence_only"
    }
    rows = []
    for raw in payload["rows"]:
        condition = str(raw["condition"])
        if condition == "evidence_only":
            continue
        match = re.fullmatch(r"(relevant|irrelevant)_distractors_k(\d+)", condition)
        if match is None:
            continue
        mode, count_text = match.groups()
        count = int(count_text)
        key = (str(raw["dataset"]), str(raw["example_id"]))
        entry = entries[key]
        baseline = baselines[key]
        relevant, irrelevant = ranked_distractors(entry)
        added = (relevant if mode == "relevant" else irrelevant)[:count]
        added_text = "\n\n".join(added)
        question_terms = _terms(str(entry["question"]))
        answer_terms = _terms(str(entry["answer"]))
        added_terms = _terms(added_text)
        answer_normalized = _normalized(str(entry["answer"]))
        rows.append(
            {
                "dataset": raw["dataset"],
                "example_id": raw["example_id"],
                "mode": mode,
                "distractor_count": count,
                "f1_delta": float(raw["token_f1"])
                - float(baseline["token_f1"]),
                "containment_delta": float(raw["answer_containment"])
                - float(baseline["answer_containment"]),
                "output_changed": raw["output_text"] != baseline["output_text"],
                "question_term_overlap": len(question_terms & added_terms),
                "answer_term_overlap": len(answer_terms & added_terms),
                "answer_exact_in_added": bool(answer_normalized)
                and answer_normalized in _normalized(added_text),
                "answer_exact_in_selected": bool(answer_normalized)
                and answer_normalized
                in _normalized(str(entry["selected_source"])),
                "source_tokens": int(raw["source_tokens"]),
                "distractor_tokens": int(raw["distractor_tokens"]),
                "full_limit_reached": int(raw["source_tokens"])
                >= int(payload["max_full_tokens"]),
                "selected_preserved_at_front": True,
                "added_documents_closer_to_question": True,
            }
        )

    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["mode"], row["distractor_count"])].append(row)
    summaries = []
    for (dataset, mode, count), selected in sorted(groups.items()):
        deltas = [float(row["f1_delta"]) for row in selected]
        summaries.append(
            {
                "dataset": dataset,
                "mode": mode,
                "distractor_count": count,
                "samples": len(selected),
                "mean_f1_delta": statistics.fmean(deltas),
                "improved_fraction": statistics.fmean(
                    float(value > 0) for value in deltas
                ),
                "harmed_fraction": statistics.fmean(float(value < 0) for value in deltas),
                "output_changed_fraction": statistics.fmean(
                    float(row["output_changed"]) for row in selected
                ),
                "answer_exact_in_added_fraction": statistics.fmean(
                    float(row["answer_exact_in_added"]) for row in selected
                ),
                "mean_answer_term_overlap": statistics.fmean(
                    float(row["answer_term_overlap"]) for row in selected
                ),
                "mean_question_term_overlap": statistics.fmean(
                    float(row["question_term_overlap"]) for row in selected
                ),
                "full_limit_fraction": statistics.fmean(
                    float(row["full_limit_reached"]) for row in selected
                ),
            }
        )

    stratified = []
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        for mode in ("relevant", "irrelevant"):
            selected = [
                row for row in rows if row["dataset"] == dataset and row["mode"] == mode
            ]
            for answer_present in (False, True):
                subset = [
                    row
                    for row in selected
                    if bool(row["answer_exact_in_added"]) == answer_present
                ]
                if subset:
                    stratified.append(
                        {
                            "dataset": dataset,
                            "mode": mode,
                            "answer_exact_in_added": answer_present,
                            "samples": len(subset),
                            "mean_f1_delta": statistics.fmean(
                                float(row["f1_delta"]) for row in subset
                            ),
                            "improved_fraction": statistics.fmean(
                                float(row["f1_delta"] > 0) for row in subset
                            ),
                        }
                    )

    return {
        "schema_version": "paper6.3-openvino-distractor-forensics-v1",
        "source_schema_version": payload["schema_version"],
        "evidence_tier": "PAIRED_POST_HOC_MECHANISM_AUDIT",
        "model_id": payload["model_id"],
        "device": payload["device"],
        "examples": len(baselines),
        "paired_rows": len(rows),
        "construction_audit": {
            "relevance_terms": "question_plus_gold_answer",
            "candidate_source": "same_example_nonselected_document_blocks",
            "document_order": "selected_evidence_then_added_documents_then_question",
            "selected_evidence_truncated_by_additions": False,
            "interpretation": (
                "The relevant condition is an answer-aware lexical diagnostic, not "
                "a deployable query-only selector or random extra-context baseline."
            ),
        },
        "correlations": {
            "f1_delta_vs_answer_term_overlap": _pearson(
                (row["f1_delta"] for row in rows),
                (row["answer_term_overlap"] for row in rows),
            ),
            "f1_delta_vs_question_term_overlap": _pearson(
                (row["f1_delta"] for row in rows),
                (row["question_term_overlap"] for row in rows),
            ),
            "f1_delta_vs_distractor_tokens": _pearson(
                (row["f1_delta"] for row in rows),
                (row["distractor_tokens"] for row in rows),
            ),
        },
        "summaries": summaries,
        "answer_presence_strata": stratified,
        "rows": rows,
    }


def _write_table(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dataset & order & $k$ & $\Delta$F1 & improved & harmed & answer present & cap hit \\",
        r"\midrule",
    ]
    for row in report["summaries"]:
        lines.append(
            "{} & {} & {} & {:+.3f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
                row["dataset"],
                row["mode"],
                row["distractor_count"],
                row["mean_f1_delta"],
                row["improved_fraction"],
                row["harmed_fraction"],
                row["answer_exact_in_added_fraction"],
                row["full_limit_fraction"],
            )
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(report: Mapping[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    datasets = sorted({row["dataset"] for row in report["summaries"]})
    figure, axes = plt.subplots(1, len(datasets), figsize=(10.2, 3.0), sharey=True)
    for axis, dataset in zip(axes, datasets):
        for mode, color, marker in (
            ("relevant", "#4c78a8", "o"),
            ("irrelevant", "#e15759", "s"),
        ):
            rows = [
                row
                for row in report["summaries"]
                if row["dataset"] == dataset and row["mode"] == mode
            ]
            axis.plot(
                [row["distractor_count"] for row in rows],
                [row["mean_f1_delta"] for row in rows],
                marker=marker,
                color=color,
                label=mode,
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(dataset)
        axis.set_xlabel("added documents")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("paired F1 change")
    axes[-1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = analyze(payload, manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "distractor_forensics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _write_table(report, args.output_dir / "generated_distractor_forensics_table.tex")
    _plot(report, args.output_dir / "distractor_forensics.png")


if __name__ == "__main__":
    main()
