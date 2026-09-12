"""Export full JSON and economical Markdown views of agent trajectories.

The JSON artifact retains every canonical record verbatim.  The Markdown view
is intentionally compact: large observations are represented by bounded head
and tail excerpts plus an explicit omission count and content digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .dag import build_resource_effect_dag
from .recordizer import recordize_minisweagent_messages


_COMMAND_BLOCK = re.compile(
    r"```(?:mswea_bash_command|bash)?\s*\n(.*?)\n```", re.DOTALL
)
_PR_DESCRIPTION = re.compile(
    r"<pr_description>\s*(.*?)\s*</pr_description>", re.DOTALL
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return normalized or "unknown_task"


def _compact_text(
    text: str,
    *,
    head_lines: int,
    tail_lines: int,
    max_line_chars: int,
) -> tuple[str, bool]:
    """Return a bounded, explicit head/tail representation of ``text``."""

    raw_lines = text.splitlines()
    display_lines: list[str] = []
    shortened_line_count = 0
    for line in raw_lines:
        if len(line) <= max_line_chars:
            display_lines.append(line)
            continue
        side = max(1, (max_line_chars - 40) // 2)
        omitted = len(line) - side * 2
        display_lines.append(
            f"{line[:side]} … [{omitted} chars omitted] … {line[-side:]}"
        )
        shortened_line_count += 1

    omitted_lines = max(0, len(display_lines) - head_lines - tail_lines)
    compacted = bool(omitted_lines or shortened_line_count)
    if omitted_lines:
        visible = [
            *display_lines[:head_lines],
            (
                f"… [{omitted_lines} lines omitted; full={len(raw_lines)} lines, "
                f"{len(text)} chars, sha256={_digest(text)}] …"
            ),
            *(display_lines[-tail_lines:] if tail_lines else ()),
        ]
    else:
        visible = display_lines
        if shortened_line_count:
            visible.append(
                f"… [{shortened_line_count} long lines compacted; "
                f"full sha256={_digest(text)}] …"
            )
    return "\n".join(visible), compacted


def _indented(text: str) -> str:
    lines = text.splitlines() or [""]
    return "\n".join(f"    {line}" for line in lines)


def _task_description(content: str) -> str:
    match = _PR_DESCRIPTION.search(content)
    return match.group(1).strip() if match else content.strip()


def _assistant_reasoning(content: str) -> str:
    value = _COMMAND_BLOCK.sub("[command shown below]", content)
    value = re.sub(r"</?format(?:_example)?>", "", value)
    return value.strip()


def _model_summary(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    info = trajectory.get("info")
    info = info if isinstance(info, Mapping) else {}
    config = info.get("config")
    config = config if isinstance(config, Mapping) else {}
    model = config.get("model")
    model = model if isinstance(model, Mapping) else {}
    kwargs = model.get("model_kwargs")
    kwargs = kwargs if isinstance(kwargs, Mapping) else {}
    return {
        "model_name": model.get("model_name"),
        "model_type": config.get("model_type"),
        "temperature": kwargs.get("temperature"),
        "api_base": kwargs.get("api_base"),
        "mini_version": info.get("mini_version"),
    }


def build_review_export(
    trajectory: Mapping[str, Any],
    *,
    source_name: str,
    source_sha256: str,
) -> dict[str, Any]:
    messages = trajectory.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError("trajectory.messages must be a sequence")
    history = recordize_minisweagent_messages(messages)
    dag = build_resource_effect_dag(history)
    info = trajectory.get("info")
    info = info if isinstance(info, Mapping) else {}
    return {
        "schema_version": 1,
        "study": "paper8_5_agent_memory_human_review_history",
        "instance_id": trajectory.get("instance_id"),
        "source": {
            "filename": source_name,
            "trajectory_sha256": source_sha256,
        },
        "outcome": {
            "exit_status": info.get("exit_status"),
            "submission": info.get("submission"),
            "model_stats": info.get("model_stats"),
        },
        "model": _model_summary(trajectory),
        "instrumentation": info.get("paper8_5_instrumentation"),
        "dag": {
            "effect_count": len(dag.effects),
            "edge_count": len(dag.edges),
            "candidate_count": len(dag.exclusion_candidates),
            "certified_excluded_causal_group_ids": list(
                dag.certified_excluded_groups
            ),
            "candidates": [{
                "causal_group_id": row.causal_group_id,
                "classification": row.classification.value,
                "evidence_record_ids": list(row.evidence_record_ids),
                "default_exclusion_eligible": row.default_exclusion_eligible,
                "reason": row.reason,
                "certificate": (
                    {
                        "rule_id": row.certificate.rule_id,
                        "proof_scope": row.certificate.proof_scope,
                        "witness_record_ids": list(
                            row.certificate.witness_record_ids
                        ),
                        "resource_version_fingerprints": [
                            list(value)
                            for value in row.certificate.resource_version_fingerprints
                        ],
                        "assumptions": list(row.certificate.assumptions),
                    }
                    if row.certificate else None
                ),
            } for row in dag.exclusion_candidates],
        },
        "turns": [{
            "turn_id": turn.turn_id,
            "causal_group_id": turn.causal_group_id,
            "record_ids": list(turn.record_ids),
            "first_message_index": turn.first_message_index,
            "complete": turn.complete,
        } for turn in history.turns],
        "records": [{
            "record_id": record.record_id,
            "turn_id": record.turn_id,
            "causal_group_id": record.causal_group_id,
            "message_index": record.message_index,
            "role": record.role,
            "primary_role": record.primary_role.value,
            "semantic_roles": [role.value for role in record.semantic_roles],
            "content": record.content,
            "content_chars": len(record.content),
            "content_sha256": _digest(record.content),
            "command": record.command,
            "return_code": record.return_code,
            "resource_ids": list(record.resource_ids),
            "metadata": dict(record.metadata),
        } for record in history.records],
    }


def render_review_markdown(
    export: Mapping[str, Any],
    *,
    json_filename: str,
    head_lines: int,
    tail_lines: int,
    max_line_chars: int,
) -> str:
    records = list(export["records"])
    record_by_id = {row["record_id"]: row for row in records}
    system = next((row for row in records if row["primary_role"] == "system"), None)
    task = next((row for row in records if row["primary_role"] == "task"), None)
    outcome = export["outcome"]
    model = export["model"]
    dag = export["dag"]
    task_text = _task_description(task["content"]) if task else "[missing]"
    task_excerpt, task_compacted = _compact_text(
        task_text,
        head_lines=max(12, head_lines * 2),
        tail_lines=max(6, tail_lines),
        max_line_chars=max_line_chars,
    )
    lines = [
        f"# Agent history: `{export['instance_id']}`",
        "",
        "This is an economical review view. The adjacent JSON retains every "
        "canonical record verbatim; omitted Markdown text is always reported "
        "with its full-content SHA-256.",
        "",
        f"- Full JSON: [{json_filename}]({json_filename})",
        f"- Source trajectory SHA-256: `{export['source']['trajectory_sha256']}`",
        f"- Outcome: `{outcome.get('exit_status')}`",
        f"- Model: `{model.get('model_name')}`",
        f"- Model class: `{model.get('model_type')}`",
        f"- Temperature: `{model.get('temperature')}`",
        f"- Canonical turns: `{len(export['turns'])}`",
        f"- DAG candidates/certified exclusions: "
        f"`{dag['candidate_count']}/{len(dag['certified_excluded_causal_group_ids'])}`",
        "",
        "## Task",
        "",
        f"Full task: `{len(task_text)}` chars, SHA-256 `{_digest(task_text)}`"
        + ("; excerpted below." if task_compacted else "."),
        "",
        _indented(task_excerpt),
        "",
    ]
    if system:
        excerpt, compacted = _compact_text(
            system["content"],
            head_lines=head_lines,
            tail_lines=tail_lines,
            max_line_chars=max_line_chars,
        )
        lines.extend([
            "## System prompt",
            "",
            f"Full content: `{system['content_chars']}` chars, "
            f"SHA-256 `{system['content_sha256']}`"
            + ("; excerpted below." if compacted else "."),
            "",
            _indented(excerpt),
            "",
        ])

    candidate_by_group: dict[str, list[Mapping[str, Any]]] = {}
    for candidate in dag["candidates"]:
        candidate_by_group.setdefault(candidate["causal_group_id"], []).append(
            candidate
        )
    for ordinal, turn in enumerate(export["turns"], start=1):
        turn_records = [record_by_id[value] for value in turn["record_ids"]]
        action = next(
            (row for row in turn_records if row["primary_role"] == "assistant_action"),
            None,
        )
        observations = [
            row for row in turn_records if "tool_observation" in row["semantic_roles"]
        ]
        lines.extend([
            f"## Turn {ordinal}: `{turn['turn_id']}`",
            "",
            f"Causal group `{turn['causal_group_id']}`; complete=`{turn['complete']}`.",
            "",
        ])
        if action:
            reasoning, reasoning_compacted = _compact_text(
                _assistant_reasoning(action["content"]),
                head_lines=head_lines,
                tail_lines=tail_lines,
                max_line_chars=max_line_chars,
            )
            command = action.get("command") or "[no valid command]"
            command_excerpt, command_compacted = _compact_text(
                command,
                head_lines=max(8, head_lines),
                tail_lines=max(6, tail_lines),
                max_line_chars=max_line_chars,
            )
            lines.extend([
                "Assistant reasoning" + (
                    " (excerpt)" if reasoning_compacted else ""
                ) + ":",
                "",
                _indented(reasoning),
                "",
                "Command" + (" (excerpt)" if command_compacted else "") + ":",
                "",
                _indented(command_excerpt),
                "",
            ])
        for observation in observations:
            metadata = observation.get("metadata") or {}
            semantics = metadata.get("tool_semantics") or {}
            excerpt, compacted = _compact_text(
                observation["content"],
                head_lines=head_lines,
                tail_lines=tail_lines,
                max_line_chars=max_line_chars,
            )
            resources = ", ".join(observation.get("resource_ids") or ()) or "none"
            lines.extend([
                "Observation:",
                "",
                f"- Return code: `{observation.get('return_code')}`",
                f"- Complete/truncated/timed out: "
                f"`{metadata.get('output_complete')}`/"
                f"`{metadata.get('output_truncated')}`/`{metadata.get('timed_out')}`",
                f"- cwd: `{metadata.get('cwd')}`",
                f"- Resources: `{resources}`",
                f"- Tool semantics: `{semantics.get('category')}` / "
                f"`{semantics.get('provenance')}` / complete=`{semantics.get('complete')}`",
                f"- Full content: `{observation['content_chars']}` chars, "
                f"SHA-256 `{observation['content_sha256']}`"
                + ("; excerpted below." if compacted else "."),
                "",
                _indented(excerpt),
                "",
            ])
        candidates = candidate_by_group.get(turn["causal_group_id"], ())
        if candidates:
            lines.append("DAG annotations:")
            lines.append("")
            for candidate in candidates:
                lines.append(
                    f"- `{candidate['classification']}`: "
                    f"eligible=`{candidate['default_exclusion_eligible']}` — "
                    f"{candidate['reason']}"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_trajectory(
    trajectory_path: Path,
    output_directory: Path,
    *,
    head_lines: int = 6,
    tail_lines: int = 6,
    max_line_chars: int = 360,
) -> tuple[Path, Path]:
    raw = trajectory_path.read_bytes()
    trajectory = json.loads(raw.decode("utf-8"))
    instance_id = str(trajectory.get("instance_id") or trajectory_path.stem)
    stem = f"{_safe_name(instance_id)}.history"
    json_path = output_directory / f"{stem}.json"
    markdown_path = output_directory / f"{stem}.md"
    export = build_review_export(
        trajectory,
        source_name=trajectory_path.name,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(export, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_review_markdown(
        export,
        json_filename=json_path.name,
        head_lines=head_lines,
        tail_lines=tail_lines,
        max_line_chars=max_line_chars,
    ), encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, nargs="+", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--head-lines", type=int, default=6)
    parser.add_argument("--tail-lines", type=int, default=6)
    parser.add_argument("--max-line-chars", type=int, default=360)
    args = parser.parse_args()
    if min(args.head_lines, args.tail_lines) < 0 or args.max_line_chars <= 0:
        raise ValueError(
            "head/tail limits must be non-negative and max-line-chars positive"
        )
    for trajectory in args.trajectory:
        json_path, markdown_path = export_trajectory(
            trajectory,
            args.output_directory,
            head_lines=args.head_lines,
            tail_lines=args.tail_lines,
            max_line_chars=args.max_line_chars,
        )
        print(f"{json_path}\n{markdown_path}")


if __name__ == "__main__":
    main()
