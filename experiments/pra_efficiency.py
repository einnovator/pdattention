"""Shared definitions for normalized PRA retrieval-efficiency reporting."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


METRIC_FIELDS = (
    "N", "E", "K", "recovered", "R_E", "P_E", "C_E", "K_over_N",
    "K_over_E", "rho", "eta",
)


def parse_ids(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    text = str(value).strip()
    if text.startswith("["):
        return [str(item) for item in json.loads(text)]
    separator = "|" if "|" in text else None
    return [item for item in text.split(separator) if item]


def normalized_metrics(candidate_count: int, evidence_ids: Iterable[Any], selected_ids: Iterable[Any]) -> dict[str, float | int]:
    evidence, selected = {str(item) for item in evidence_ids}, {str(item) for item in selected_ids}
    if candidate_count < len(selected):
        raise ValueError("candidate_count cannot be smaller than the selected set")
    recovered, e_count, k_count = len(evidence & selected), len(evidence), len(selected)
    recall = recovered / e_count if e_count else 0.0
    precision = recovered / k_count if k_count else 0.0
    fraction = k_count / candidate_count if candidate_count else 0.0
    return {"N": candidate_count, "E": e_count, "K": k_count, "recovered": recovered,
            "R_E": recall, "P_E": precision, "C_E": int(bool(evidence) and evidence <= selected),
            "K_over_N": fraction, "K_over_E": k_count / e_count if e_count else 0.0,
            "rho": 1.0 - fraction if candidate_count else 0.0, "eta": recall / fraction if fraction else 0.0}


def summarize(rows: Iterable[Mapping[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for group_key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        record = {**dict(zip(keys, group_key)), "examples": len(values)}
        for field in METRIC_FIELDS:
            record[field] = statistics.fmean(float(row[field]) for row in values)
        output.append(record)
    return output
