"""Prepare and score a blinded manual audit of agent-history record labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

from .recordizer import recordize_minisweagent_messages


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    for parent in (resolved.parent, *resolved.parents):
        if (parent / ".git").exists():
            return resolved.relative_to(parent).as_posix()
    return resolved.as_posix()


def _preview(text: str, *, limit: int = 320) -> str:
    """Keep audit sheets readable without hiding both ends of long output."""
    normalized = text.replace("\r\n", "\n")
    if len(normalized) <= limit:
        return normalized
    half = max(1, (limit - 39) // 2)
    return normalized[:half] + "\n...[middle omitted for audit]...\n" + normalized[-half:]


def _trajectory_rows(path: Path) -> list[dict[str, Any]]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    messages = artifact.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{path}: trajectory has no messages")
    history = recordize_minisweagent_messages(messages)
    task_id = str(artifact.get("instance_id") or path.parent.name)
    selectable = [
        record for record in history.records
        if record.turn_id not in {"system", "task"}
    ]
    last_index = max((record.message_index for record in selectable), default=1)
    rows = []
    for record in history.records:
        fraction = record.message_index / max(last_index, 1)
        position = (
            "early" if fraction < 1 / 3
            else "middle" if fraction < 2 / 3
            else "late"
        )
        auto_roles = [role.value for role in record.semantic_roles]
        inferred = "progress" in auto_roles
        rows.append({
            "task_id": task_id,
            "trajectory_path": _portable_path(path),
            "trajectory_sha256": _digest(path),
            "record_id": record.record_id,
            "turn_id": record.turn_id,
            "causal_group_id": record.causal_group_id,
            "message_index": record.message_index,
            "trajectory_position": position,
            "message_role": record.role,
            "auto_primary_role": record.primary_role.value,
            "auto_semantic_roles": ";".join(auto_roles),
            "label_grounding": "mixed_prose_and_tool" if inferred and record.command else (
                "prose_inferred" if inferred else "tool_or_protocol_grounded"
            ),
            "command": record.command or "",
            "return_code": "" if record.return_code is None else record.return_code,
            "resource_ids": ";".join(record.resource_ids),
            "content_chars": len(record.content),
            "content_preview": _preview(record.content),
            "human_primary_role": "",
            "human_semantic_roles": "",
            "human_resource_ids": "",
            "human_causal_group_correct": "",
            "human_label_correct": "",
            "human_notes": "",
        })
    return rows


def stratified_sample(
    trajectory_paths: Sequence[Path], *, sample_size: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [row for path in trajectory_paths for row in _trajectory_rows(path)]
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    rng = random.Random(seed)
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(
            str(row["task_id"]), str(row["auto_primary_role"]),
            str(row["trajectory_position"]),
        )].append(row)
    for members in strata.values():
        rng.shuffle(members)
    selected: list[dict[str, Any]] = []
    ordered = sorted(strata)
    while len(selected) < min(sample_size, len(rows)):
        advanced = False
        for key in ordered:
            if strata[key] and len(selected) < sample_size:
                selected.append(strata[key].pop())
                advanced = True
        if not advanced:
            break
    rng.shuffle(selected)
    for audit_index, row in enumerate(selected, 1):
        row["audit_index"] = audit_index
    manifest = {
        "schema_version": 1,
        "study": "paper8_5_recordizer_manual_audit",
        "seed": seed,
        "requested_sample_size": sample_size,
        "realized_sample_size": len(selected),
        "population_records": len(rows),
        "stratification": ["task_id", "auto_primary_role", "trajectory_position"],
        "trajectory_inputs": [
            {"path": _portable_path(path), "sha256": _digest(path)}
            for path in trajectory_paths
        ],
        "status": "awaiting_blinded_human_labels",
    }
    return selected, manifest


def write_audit(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (output / "recordizer_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output / "recordizer_audit.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    (output / "recordizer_audit_manifest.json").write_text(
        json.dumps(dict(manifest), indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Paper 8.5 recordizer audit\n\n"
        "This is a blinded manual-label worksheet, not a completed reliability result. "
        "Rows are stratified by task, automatic primary role, and trajectory tercile. "
        "Long content is represented by an economical head/tail preview; use the bound "
        "trajectory path and digest to inspect canonical bytes when needed.\n\n"
        "Fill `human_primary_role`, semicolon-delimited `human_semantic_roles`, "
        "`human_resource_ids`, `human_causal_group_correct`, "
        "`human_label_correct` (`yes`/`no`), and notes. "
        "Tool-grounded and prose-inferred labels must be reported separately.\n",
        encoding="utf-8",
    )


def _roles(value: Any) -> set[str]:
    return {part.strip() for part in str(value or "").split(";") if part.strip()}


def score_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    labelled = [
        row for row in rows
        if str(row.get("human_semantic_roles") or "").strip()
    ]
    if not labelled:
        raise ValueError("audit contains no human semantic-role labels")
    roles = sorted(set().union(*(
        _roles(row.get("auto_semantic_roles"))
        | _roles(row.get("human_semantic_roles"))
        for row in labelled
    )))
    per_role = {}
    for role in roles:
        tp = sum(
            role in _roles(row.get("auto_semantic_roles"))
            and role in _roles(row.get("human_semantic_roles"))
            for row in labelled
        )
        fp = sum(
            role in _roles(row.get("auto_semantic_roles"))
            and role not in _roles(row.get("human_semantic_roles"))
            for row in labelled
        )
        fn = sum(
            role not in _roles(row.get("auto_semantic_roles"))
            and role in _roles(row.get("human_semantic_roles"))
            for row in labelled
        )
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        per_role[role] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall,
        }
    boundary_labels = [
        str(row.get("human_causal_group_correct") or "").strip().lower()
        for row in labelled
        if str(row.get("human_causal_group_correct") or "").strip()
    ]
    invalid_boundaries = set(boundary_labels) - {"yes", "no"}
    if invalid_boundaries:
        raise ValueError("causal-group labels must be yes or no")
    return {
        "schema_version": 1,
        "labelled_records": len(labelled),
        "exact_semantic_role_set_accuracy": sum(
            _roles(row.get("auto_semantic_roles"))
            == _roles(row.get("human_semantic_roles"))
            for row in labelled
        ) / len(labelled),
        "causal_group_boundary_labelled_records": len(boundary_labels),
        "causal_group_boundary_accuracy": (
            sum(value == "yes" for value in boundary_labels) / len(boundary_labels)
            if boundary_labels else None
        ),
        "per_role": per_role,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--trajectory", action="append", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--sample-size", type=int, default=150)
    prepare.add_argument("--seed", type=int, default=850)
    score = sub.add_parser("score")
    score.add_argument("--audit-csv", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        rows, manifest = stratified_sample(
            [path.resolve() for path in args.trajectory],
            sample_size=args.sample_size, seed=args.seed,
        )
        write_audit(rows, manifest, args.output.resolve())
        print(json.dumps(manifest))
    else:
        with args.audit_csv.open(encoding="utf-8", newline="") as stream:
            result = score_rows(csv.DictReader(stream))
        args.output.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result))


if __name__ == "__main__":
    main()
