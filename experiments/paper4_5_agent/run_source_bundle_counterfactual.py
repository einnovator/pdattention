"""Replay the frozen Task03/04 request-6 source-bundle counterfactual.

The matched full-history prompt is reconstructed from the logical request that
produced the recorded sparse condition.  This matters for Task04: its request-5
reasoning wording differs from the separate 100-percent trajectory even though
the executed command is the same.  Only logical messages 4 and 5 vary between
the sparse and restored prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import socket
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SPARSE_MESSAGE_INDICES = (0, 1, 2, 3, 6, 7, 8, 9, 10, 11)
SOURCE_BUNDLE_MESSAGE_INDICES = (4, 5)
_COMMAND = re.compile(
    r"```mswea_bash_command\s*\n(?P<command>[\s\S]*?)\n```"
)


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_digest(value: object) -> str:
    return _text_digest(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _command(value: str) -> str | None:
    match = _COMMAND.search(value.replace("\r\n", "\n"))
    return match.group("command").strip() if match else None


def load_request(path: Path, request_index: int = 6) -> dict[str, Any]:
    """Load one one-based request from an interaction-history JSONL file."""

    requests = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") == "request":
                requests.append(row)
    if len(requests) < request_index:
        raise ValueError(f"{path} has only {len(requests)} requests")
    row = requests[request_index - 1]
    recorded_index = int(row.get("request_index", request_index))
    if recorded_index != request_index:
        raise ValueError(
            f"request slot {request_index} contains request_index={recorded_index}"
        )
    return row


def load_action(path: Path, action_index: int = 6) -> dict[str, Any]:
    """Load one one-based assistant action from a mini-swe-agent trajectory."""

    trajectory = json.loads(path.read_text(encoding="utf-8"))
    actions = [
        row for row in trajectory["messages"] if row.get("role") == "assistant"
    ]
    content = str(actions[action_index - 1].get("content", ""))
    command = _command(content)
    return {
        "content": content,
        "content_sha256": _text_digest(content),
        "command": command,
        "command_sha256": _text_digest(command) if command is not None else None,
    }


def build_fixture(request_row: Mapping[str, Any]) -> dict[str, Any]:
    """Construct the three matched prompts without consulting another run."""

    logical_messages = [
        {"role": str(row["role"]), "content": str(row.get("content", ""))}
        for row in request_row["logical_payload"]["messages"]
    ]
    if len(logical_messages) != 12:
        raise ValueError(
            f"request-6 fixture requires 12 logical messages, got {len(logical_messages)}"
        )
    manifest = [
        {
            "message_index": index,
            "role": message["role"],
            "content_sha256": _text_digest(message["content"]),
            "characters": len(message["content"]),
        }
        for index, message in enumerate(logical_messages)
    ]
    index_sets = {
        "recorded_sparse_subset": list(SPARSE_MESSAGE_INDICES),
        "restore_only_source_bundle_4_5": sorted(
            (*SPARSE_MESSAGE_INDICES, *SOURCE_BUNDLE_MESSAGE_INDICES)
        ),
        "matched_full_history": list(range(len(logical_messages))),
    }
    conditions = []
    for name, indices in index_sets.items():
        messages = [logical_messages[index] for index in indices]
        conditions.append(
            {
                "name": name,
                "selected_message_indices": indices,
                "selected_identities": [manifest[index] for index in indices],
                "messages": messages,
                "prompt_sha256": _json_digest(messages),
                "source_bundle_included": all(
                    index in indices for index in SOURCE_BUNDLE_MESSAGE_INDICES
                ),
            }
        )
    by_name = {row["name"]: row for row in conditions}
    if (
        by_name["restore_only_source_bundle_4_5"]["prompt_sha256"]
        != by_name["matched_full_history"]["prompt_sha256"]
    ):
        raise AssertionError("restored prompt is not the matched logical request")
    return {
        "logical_message_manifest": manifest,
        "conditions": conditions,
        "request5_action_message_sha256": manifest[10]["content_sha256"],
    }


def _post(
    url: str, payload: Mapping[str, Any], timeout: float
) -> tuple[dict[str, Any], float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(dict(payload)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = json.load(response)
    return raw, time.perf_counter() - started


def _get_json(url: str, timeout: float) -> Mapping[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        value = json.load(response)
    return value if isinstance(value, Mapping) else {"value": value}


def _checkpoint(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(artifact), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run(
    *,
    tasks: Sequence[Mapping[str, Any]],
    output: Path,
    base_url: str,
    model: str,
    repeats: int,
    max_tokens: int,
    timeout: float,
    source_commit: str,
) -> dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    artifact: dict[str, Any] = {
        "schema": "paper4.5-request06-source-bundle-counterfactual-v1",
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "endpoint": endpoint,
        "endpoint_models": {"status": "pending"},
        "model": model,
        "sampling": {
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
            "max_tokens": max_tokens,
        },
        "design": {
            "request_index": 6,
            "source_bundle_indices": list(SOURCE_BUNDLE_MESSAGE_INDICES),
            "sparse_indices": list(SPARSE_MESSAGE_INDICES),
            "matched_full_history_source": (
                "recorded sparse run logical request-6 before selection"
            ),
            "task04_request5_wording_frozen": True,
            "generated_commands_executed_against_repository": False,
        },
        "tasks": [],
    }
    try:
        artifact["endpoint_models"] = _get_json(
            base_url.rstrip("/") + "/v1/models", min(timeout, 30)
        )
    except Exception as error:  # the generation endpoint remains authoritative
        artifact["endpoint_models"] = {
            "status": "not_reported",
            "error": f"{type(error).__name__}: {error}",
        }
    _checkpoint(output, artifact)
    for spec in tasks:
        history_path = Path(spec["history"])
        request_row = load_request(history_path)
        fixture = build_fixture(request_row)
        full_reference = load_action(Path(spec["full_trajectory"]))
        sparse_reference = load_action(Path(spec["sparse_trajectory"]))
        task_result: dict[str, Any] = {
            "task": spec["task"],
            "instance_id": spec["instance_id"],
            "history_file_sha256": _text_digest(
                history_path.read_text(encoding="utf-8")
            ),
            "logical_request_input_sha256": request_row.get("request_input_sha256"),
            "logical_message_manifest": fixture["logical_message_manifest"],
            "request5_action_message_sha256": fixture[
                "request5_action_message_sha256"
            ],
            "recorded_full_history_action_6": full_reference,
            "recorded_sparse_action_6": sparse_reference,
            "conditions": [],
        }
        artifact["tasks"].append(task_result)
        for fixture_condition in fixture["conditions"]:
            messages = fixture_condition["messages"]
            condition = {
                key: value
                for key, value in fixture_condition.items()
                if key != "messages"
            }
            condition["runs"] = []
            task_result["conditions"].append(condition)
            for repeat in range(repeats):
                raw, elapsed = _post(
                    endpoint,
                    {
                        "model": model,
                        "messages": messages,
                        "temperature": 0,
                        "top_p": 1,
                        "seed": 0,
                        "max_tokens": max_tokens,
                        "stream": False,
                    },
                    timeout,
                )
                choice = raw["choices"][0]
                content = str(choice.get("message", {}).get("content") or "")
                command = _command(content)
                usage = raw.get("usage") or {}
                prompt_details = usage.get("prompt_tokens_details") or {}
                timings = raw.get("timings") or {}
                reported_cache_tokens = prompt_details.get(
                    "cached_tokens", timings.get("cache_n")
                )
                condition["runs"].append(
                    {
                        "repeat": repeat + 1,
                        "elapsed_seconds": elapsed,
                        "action_text": content,
                        "action_sha256": _text_digest(content),
                        "executed_command": command,
                        "executed_command_sha256": (
                            _text_digest(command) if command is not None else None
                        ),
                        "finish_reason": choice.get("finish_reason"),
                        "usage": usage,
                        "kv_telemetry": raw.get("pra")
                        or {
                            "status": (
                                "prefix_cache_counters_reported"
                                if reported_cache_tokens is not None
                                else "not_reported_by_endpoint"
                            ),
                            "cached_prompt_tokens": reported_cache_tokens,
                            "newly_evaluated_prompt_tokens": timings.get("prompt_n"),
                            "prompt_evaluation_ms": timings.get("prompt_ms"),
                            "note": (
                                "Endpoint counters describe ordinary contiguous prefix reuse, "
                                "not sparse PRA K/V attachment."
                            ),
                        },
                        "matches_recorded_full_action_text": (
                            _text_digest(content) == full_reference["content_sha256"]
                        ),
                        "matches_recorded_full_executed_command": (
                            command is not None and command == full_reference["command"]
                        ),
                        "matches_recorded_sparse_action_text": (
                            _text_digest(content) == sparse_reference["content_sha256"]
                        ),
                        "matches_recorded_sparse_executed_command": (
                            command is not None and command == sparse_reference["command"]
                        ),
                        "raw_response_metadata": {
                            key: value for key, value in raw.items() if key != "choices"
                        },
                    }
                )
                _checkpoint(output, artifact)
            condition["deterministic_action_text"] = (
                len({row["action_sha256"] for row in condition["runs"]}) == 1
            )
            condition["deterministic_executed_command"] = (
                len(
                    {
                        row["executed_command_sha256"]
                        for row in condition["runs"]
                    }
                )
                == 1
            )
            condition["all_match_recorded_full_command"] = all(
                row["matches_recorded_full_executed_command"]
                for row in condition["runs"]
            )
            _checkpoint(output, artifact)
        by_name = {row["name"]: row for row in task_result["conditions"]}
        sparse = by_name["recorded_sparse_subset"]
        restored = by_name["restore_only_source_bundle_4_5"]
        full = by_name["matched_full_history"]
        task_result["causal_recovery"] = {
            "restored_prompt_equals_matched_full_prompt": (
                restored["prompt_sha256"] == full["prompt_sha256"]
            ),
            "sparse_prompt_differs_only_by_messages_4_5": True,
            "restored_all_match_recorded_full_command": restored[
                "all_match_recorded_full_command"
            ],
            "sparse_all_match_recorded_full_command": sparse[
                "all_match_recorded_full_command"
            ],
            "supported": (
                restored["all_match_recorded_full_command"]
                and not sparse["all_match_recorded_full_command"]
            ),
        }
        _checkpoint(output, artifact)
    artifact["status"] = "complete"
    artifact["completed_utc"] = datetime.now(timezone.utc).isoformat()
    _checkpoint(output, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-specs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3-coder:30b")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--source-commit", default="cbf190e2")
    args = parser.parse_args()
    task_specs = json.loads(args.task_specs.read_text(encoding="utf-8"))
    run(
        tasks=task_specs,
        output=args.output,
        base_url=args.base_url,
        model=args.model,
        repeats=args.repeats,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        source_commit=args.source_commit,
    )


if __name__ == "__main__":
    main()
