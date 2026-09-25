"""Reduce immutable cross-agent session campaigns to the Paper 8.5 matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _official_ids(root: Path) -> tuple[set[str], set[str]] | None:
    reports = sorted(root.glob("official_grade/*.json"))
    reports += sorted(root.glob("official_grade/**/report.json"))
    for path in reports:
        value = _json(path)
        resolved = value.get("resolved_ids")
        unresolved = value.get("unresolved_ids")
        if isinstance(resolved, list) and isinstance(unresolved, list):
            return set(map(str, resolved)), set(map(str, unresolved))
    resolved: set[str] = set()
    unresolved: set[str] = set()
    found = False
    for path in sorted(root.glob("episode-*/official_result.json")):
        value = _json(path)
        instance_id = str(value.get("instance_id") or "")
        outcome = value.get("resolved")
        if instance_id and isinstance(outcome, bool):
            found = True
            (resolved if outcome else unresolved).add(instance_id)
    return (resolved, unresolved) if found else None


def _calls(root: Path) -> int | None:
    values = []
    for path in sorted(root.glob("episode-*/event_summary.json")):
        value = _json(path)
        calls = value.get("assistant_model_calls")
        if calls is None:
            calls = value.get("distinct_llm_response_count")
        if isinstance(calls, int):
            values.append(calls)
    return sum(values) if values else None


def _row(root: Path) -> dict[str, Any] | None:
    relative = root.parts
    try:
        n_index = next(index for index, name in enumerate(relative) if name.startswith("n") and name[1:].isdigit())
    except StopIteration:
        return None
    agent = relative[n_index - 1]
    n = int(relative[n_index][1:])
    arm = relative[n_index + 1]
    campaign = root / "campaign_state.json"
    manifest = root / "run_manifest.json"
    if campaign.is_file():
        value = _json(campaign)
        task_count = int(value.get("task_count_completed") or 0)
        requests = int(value.get("request_count") or 0)
        calls = _calls(root)
        full_tokens = int(value.get("cumulative_full_history_tokens") or 0)
        materialized = int(value.get("cumulative_materialized_history_tokens") or 0)
        saving = float(value.get("own_saving_fraction") or 0.0)
        status = str(value.get("status") or "unknown")
        policy = str(value.get("policy") or arm)
    elif manifest.is_file():
        value = _json(manifest)
        task_count = int(value.get("task_count_completed") or 0)
        requests = int(value.get("request_count") or 0)
        calls = requests
        full_tokens = int(value.get("cumulative_full_history_tokens") or 0)
        materialized = int(value.get("cumulative_materialized_history_tokens") or 0)
        saving = float(value.get("own_saving_fraction") or 0.0)
        status = "execution_complete_pending_grade"
        policy = str(value.get("history_policy") or arm)
    else:
        return None
    official = _official_ids(root)
    return {
        "agent": agent,
        "n": n,
        "arm": arm,
        "policy": policy,
        "status": status,
        "tasks_completed": task_count,
        "resolved": None if official is None else len(official[0]),
        "unresolved": None if official is None else len(official[1]),
        "resolved_ids": [] if official is None else sorted(official[0]),
        "calls": calls,
        "requests": requests,
        "full_history_tokens": full_tokens,
        "materialized_tokens": materialized,
        "own_saving_fraction": saving,
        "paired_saving_fraction": None,
        "lost_full_success_ids": [],
        "evidence_root": str(root),
    }


def reduce_matrix(root: Path) -> dict[str, Any]:
    roots = {path.parent for path in root.glob("*/n*/*/campaign_state.json")}
    roots |= {path.parent for path in root.glob("*/n*/*/run_manifest.json")}
    rows = [row for directory in sorted(roots) if (row := _row(directory))]
    full_by_key = {
        (row["agent"], row["n"]): row
        for row in rows if row["arm"].lower() == "full"
    }
    for row in rows:
        if row["arm"].lower() == "full":
            row["paired_saving_fraction"] = 0.0
            continue
        full = full_by_key.get((row["agent"], row["n"]))
        if full and full["materialized_tokens"]:
            row["paired_saving_fraction"] = (
                1.0 - row["materialized_tokens"] / full["materialized_tokens"]
            )
            row["lost_full_success_ids"] = sorted(
                set(full["resolved_ids"]) - set(row["resolved_ids"])
            )
    return {"schema_version": 1, "rows": rows}


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    columns = [
        "agent", "n", "arm", "policy", "status", "tasks_completed",
        "resolved", "calls", "requests", "full_history_tokens",
        "materialized_tokens", "own_saving_fraction",
        "paired_saving_fraction", "lost_full_success_ids", "evidence_root",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            value = dict(row)
            value["lost_full_success_ids"] = ";".join(row["lost_full_success_ids"])
            writer.writerow(value)


def _percent(value: Any) -> str:
    return "---" if value is None else f"{100 * float(value):.2f}%"


def _write_markdown(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    lines = [
        "# Cross-agent session matrix",
        "",
        "Paired saving includes trajectory-length effects. Missing grades remain pending; they are not failures.",
        "",
        "| Agent | N | Arm | Resolved | Calls | Own saving | Paired saving | Lost FULL successes |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        resolved = "---" if row["resolved"] is None else f"{row['resolved']}/{row['tasks_completed']}"
        calls = "---" if row["calls"] is None else str(row["calls"])
        lost = ", ".join(row["lost_full_success_ids"]) or "none"
        lines.append(
            f"| {row['agent']} | {row['n']} | {row['arm']} | {resolved} | "
            f"{calls} | {_percent(row['own_saving_fraction'])} | "
            f"{_percent(row['paired_saving_fraction'])} | {lost} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = reduce_matrix(Path(args.root).resolve())
    (output / "matrix.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output / "matrix.csv", result["rows"])
    _write_markdown(output / "matrix.md", result["rows"])


if __name__ == "__main__":
    main()
