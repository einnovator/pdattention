"""Durable partial reports for interrupted Paper 4.5 agent campaigns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .schema import CampaignConfig


def write_reports(config: CampaignConfig, state: Mapping[str, Any], root: Path) -> None:
    """Write every required report after each state transition."""

    root.mkdir(parents=True, exist_ok=True)
    cells = state.get("cells", {})
    summary = {
        "campaign_id": config.campaign_id,
        "baselines": [row.model_dump(mode="json") for row in config.baselines],
        "cells": cells,
        "pra_interpretation_allowed": any(
            value.get("reproduction_status") == "BASELINE_REPRODUCED"
            for value in cells.values()
        ),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    result_lines = [
        json.dumps({"cell_id": cell_id, **value}, sort_keys=True)
        for cell_id, value in sorted(cells.items())
        if value.get("result") is not None
    ]
    (root / "results.jsonl").write_text("\n".join(result_lines) + ("\n" if result_lines else ""), encoding="utf-8")

    reproduction = [
        "# Baseline reproduction report", "",
        "PRA and gateway treatments remain locked until the matching no-PRA cell is `BASELINE_REPRODUCED`.", "",
        "| Cell | Model | Harness | Published | Observed | Status | Notes |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    baselines = {row.baseline_id: row for row in config.baselines}
    for cell in config.cells:
        if cell.mode.value != "native":
            continue
        baseline = baselines[cell.baseline_id]
        value = cells.get(cell.cell_id, {})
        review = value.get("review") or {}
        observed = review.get("observed_score")
        notes = "; ".join(review.get("reasons") or cell.notes or ("Not run",))
        reproduction.append(
            f"| `{cell.cell_id}` | `{baseline.model}` | `{baseline.harness}` | "
            f"{baseline.published_score:.1%} | {observed:.1%} | {review.get('status', 'PLANNED')} | {notes} |"
            if observed is not None else
            f"| `{cell.cell_id}` | `{baseline.model}` | `{baseline.harness}` | "
            f"{baseline.published_score:.1%} | - | {review.get('status', 'PLANNED')} | {notes} |"
        )
    (root / "reproduction_report.md").write_text("\n".join(reproduction) + "\n", encoding="utf-8")

    treatments = [cell for cell in config.cells if cell.mode.value != "native"]
    frontier = [
        "# PRA frontier report", "",
        "No PRA frontier is interpreted before a compatible official baseline exists.", "",
        "| Cell | Mode | State | Baseline gate |",
        "| --- | --- | --- | --- |",
    ]
    for cell in treatments:
        value = cells.get(cell.cell_id, {})
        frontier.append(
            f"| `{cell.cell_id}` | `{cell.mode.value}` | {value.get('state', 'PENDING')} | "
            f"`{cell.baseline_cell}` |"
        )
    (root / "frontier_report.md").write_text("\n".join(frontier) + "\n", encoding="utf-8")

    failures = ["# Campaign failures", ""]
    found = False
    for cell_id, value in sorted(cells.items()):
        if value.get("state") in {"FAILED", "BLOCKED"}:
            found = True
            failures.extend((f"## `{cell_id}`", "", str(value.get("error") or value.get("reason") or "Unknown failure"), ""))
    if not found:
        failures.append("No execution failures have been recorded.")
    (root / "failures.md").write_text("\n".join(failures) + "\n", encoding="utf-8")
