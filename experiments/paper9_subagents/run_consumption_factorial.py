"""Run a frozen cross-model selected-record consumption factorial.

Selection is immutable across every arm. The experiment changes only how the
same selected record is consumed: direct injection, attribution-aware
presentation, or attribution-aware presentation with mandatory tool
verification against a pinned read-only repository snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from experiments.paper9_subagents.run_autonomous_repository_campaign import (
    OllamaChatClient,
    TOOL_SCHEMAS,
    _textual_tool_calls,
)


@dataclass(frozen=True)
class ConsumptionCondition:
    name: str
    attribution_aware: bool
    require_tool_verification: bool


CONDITIONS = (
    ConsumptionCondition("direct_injection", False, False),
    ConsumptionCondition("attribution_aware", True, False),
    ConsumptionCondition("mandatory_tool_verification", True, True),
)


class FrozenRepositoryTools:
    """Read/search only the candidate files embedded in the frozen manifest."""

    def __init__(self, files: Mapping[str, Mapping[str, object]]) -> None:
        self.files = files
        self.calls = 0

    def execute(self, name: str, arguments: Mapping[str, object]) -> str:
        self.calls += 1
        if name == "list_files":
            prefix = str(arguments.get("path", ".")).strip("./")
            paths = sorted(self.files)
            if prefix:
                paths = [path for path in paths if path.startswith(prefix)]
            return "\n".join(paths)
        if name == "read_text":
            path = str(arguments.get("path", "")).replace("\\", "/")
            record = self.files.get(path)
            return str(record["text"]) if record else f"ERROR: file not found: {path}"
        if name == "search_text":
            query = str(arguments.get("query", "")).strip().lower()
            prefix = str(arguments.get("path", ".")).strip("./")
            if not query:
                return "ERROR: query is required"
            matches: list[str] = []
            for path, record in sorted(self.files.items()):
                if prefix and not path.startswith(prefix):
                    continue
                for line_number, line in enumerate(str(record["text"]).splitlines(), 1):
                    if query in line.lower():
                        matches.append(f"{path}:{line_number}:{line.strip()}")
                        if len(matches) >= 80:
                            return "\n".join(matches)
            return "\n".join(matches) if matches else "NO MATCHES"
        return f"ERROR: invalid or unavailable tool call: {name}"


def _record_block(
    case: Mapping[str, object],
    selected_text: str,
    condition: ConsumptionCondition,
) -> str:
    if condition.attribution_aware:
        header = (
            "ROUTER-SELECTED RECORD (not independently verified)\n"
            f"repository: {case['repository_id']}\n"
            f"revision: {case['revision']}\n"
            f"source_path: {case['selected_path']}\n"
            f"sha256: {case['selected_sha256']}\n"
            "The path identifies the supplied record, not necessarily the answer. "
            "State whether its contents actually support your conclusion."
        )
    else:
        header = f"SELECTED CONTEXT\nsource_path: {case['selected_path']}"
    return f"{header}\n\n```text\n{selected_text}\n```"


def run_case(
    client,
    manifest: Mapping[str, object],
    case: Mapping[str, object],
    condition: ConsumptionCondition,
    *,
    model: str,
    max_steps: int,
) -> dict[str, object]:
    repository = manifest["repositories"][case["repository_id"]]
    files = repository["files"]
    selected = files[case["selected_path"]]
    selected_text = str(selected["text"])
    digest = hashlib.sha256(selected_text.encode("utf-8")).hexdigest()
    if digest != case["selected_sha256"]:
        raise ValueError(f"Selected record hash mismatch for {case['query_id']}")

    system = (
        "You are a repository evidence analyst. Answer the question concisely, name "
        "the exact relative source path and relevant class or function, and do not guess."
    )
    if condition.require_tool_verification:
        system += (
            " You must make at least one read-only tool call before answering. Check whether "
            "the router-selected record really supports the answer; if it does not, use the "
            "tools to locate the correct candidate file."
        )
    messages: list[dict[str, object]] = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": "/no_think " + str(case["query"]) + "\n\n" + _record_block(
                case, selected_text, condition
            ),
        },
    ]
    tools = FrozenRepositoryTools(files)
    totals = {
        "model_calls": 0,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "model_seconds": 0.0,
    }
    trace: list[dict[str, object]] = []
    answer = ""
    error: str | None = None
    started = time.perf_counter()
    try:
        for step in range(max_steps):
            assistant, metrics = client.chat(
                messages,
                TOOL_SCHEMAS if condition.require_tool_verification else (),
            )
            totals["model_calls"] += 1
            totals["prompt_tokens"] += int(metrics.get("prompt_tokens", 0))
            totals["generated_tokens"] += int(metrics.get("generated_tokens", 0))
            totals["model_seconds"] += float(metrics.get("model_seconds", 0.0))
            calls = assistant.get("tool_calls", ())
            if not condition.require_tool_verification:
                calls = ()
            elif not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
                calls = ()
            if condition.require_tool_verification and not calls:
                calls = _textual_tool_calls(assistant.get("content", ""))
            if calls:
                messages.append({"role": "assistant", "content": "", "tool_calls": calls})
                for call in calls:
                    function = call.get("function", {}) if isinstance(call, Mapping) else {}
                    name = str(function.get("name", ""))
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    if not isinstance(arguments, Mapping):
                        arguments = {}
                    output = tools.execute(name, arguments)
                    trace.append(
                        {
                            "step": step,
                            "tool": name,
                            "arguments": dict(arguments),
                            "output_chars": len(output),
                        }
                    )
                    messages.append(
                        {"role": "tool", "tool_name": name, "content": output[:16_000]}
                    )
                continue
            if condition.require_tool_verification and tools.calls == 0:
                messages.append(dict(assistant))
                messages.append(
                    {
                        "role": "user",
                        "content": "Verification is mandatory. Use a read-only tool before answering.",
                    }
                )
                continue
            answer = str(assistant.get("content", "")).strip()
            break
        else:
            error = "max_steps_exhausted"
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"

    normalized = answer.replace("\\", "/").lower()
    relevant_path_hit = str(case["relevant_path"]).lower() in normalized
    selected_path_hit = str(case["selected_path"]).lower() in normalized
    return {
        "model": model,
        "condition": condition.name,
        "repository_id": case["repository_id"],
        "revision": case["revision"],
        "query_id": case["query_id"],
        "query": case["query"],
        "relevant_path": case["relevant_path"],
        "selected_path": case["selected_path"],
        "selected_sha256": case["selected_sha256"],
        "selection_correct": case["selection_correct"],
        "relevant_path_hit": relevant_path_hit,
        "selected_path_hit": selected_path_hit,
        "recovered_from_wrong_selection": bool(
            not case["selection_correct"] and relevant_path_hit
        ),
        "verification_required": condition.require_tool_verification,
        "verification_compliant": tools.calls > 0,
        "tool_calls": tools.calls,
        "answer": answer,
        "failed": error is not None,
        "error": error,
        "wall_seconds": time.perf_counter() - started,
        "trace": trace,
        **totals,
    }


def summarize(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["model"]), str(row["condition"])), []).append(row)
    summaries: list[dict[str, object]] = []
    for (model, condition), values in sorted(grouped.items()):
        correctly_selected = [row for row in values if row["selection_correct"]]
        wrongly_selected = [row for row in values if not row["selection_correct"]]
        summaries.append(
            {
                "model": model,
                "condition": condition,
                "cases": len(values),
                "failures": sum(bool(row["failed"]) for row in values),
                "path_accuracy": sum(bool(row["relevant_path_hit"]) for row in values)
                / len(values),
                "conditional_consumption_accuracy": sum(
                    bool(row["relevant_path_hit"]) for row in correctly_selected
                )
                / max(1, len(correctly_selected)),
                "wrong_selection_recovery": sum(
                    bool(row["recovered_from_wrong_selection"]) for row in wrongly_selected
                )
                / max(1, len(wrongly_selected)),
                "verification_compliance": sum(
                    bool(row["verification_compliant"]) for row in values
                )
                / len(values),
                "tool_calls": sum(int(row["tool_calls"]) for row in values),
                "prompt_tokens": sum(int(row["prompt_tokens"]) for row in values),
                "generated_tokens": sum(int(row["generated_tokens"]) for row in values),
                "wall_seconds": sum(float(row["wall_seconds"]) for row in values),
            }
        )
    return summaries


def _ollama_metadata(base_url: str, models: Sequence[str]) -> dict[str, object]:
    def get(path: str) -> Mapping[str, object]:
        with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=30) as response:
            value = json.loads(response.read())
        return value if isinstance(value, Mapping) else {}

    version = get("/api/version").get("version")
    tags = get("/api/tags").get("models", ())
    by_name = {
        str(row.get("name")): row
        for row in tags
        if isinstance(row, Mapping) and row.get("name")
    }
    return {
        "engine": "ollama",
        "engine_version": version,
        "base_url": base_url,
        "runner_host": platform.node(),
        "runner_platform": platform.platform(),
        "models": {
            model: {
                "digest": by_name.get(model, {}).get("digest"),
                "size": by_name.get(model, {}).get("size"),
                "modified_at": by_name.get(model, {}).get("modified_at"),
            }
            for model in models
        },
    }


def _model_digests(runtime: Mapping[str, object]) -> dict[str, str | None]:
    models = runtime.get("models", {})
    if not isinstance(models, Mapping):
        return {}
    return {
        str(name): (
            str(metadata.get("digest"))
            if isinstance(metadata, Mapping) and metadata.get("digest")
            else None
        )
        for name, metadata in models.items()
    }


def _identity_digest(identity: Mapping[str, object]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--models", default="qwen3-coder:30b,qwen3:14b,gemma3:4b-it-qat")
    parser.add_argument("--conditions", default=",".join(row.name for row in CONDITIONS))
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "paper9.consumption_factorial_manifest.v1":
        raise ValueError("Unsupported consumption-factorial manifest schema.")
    condition_names = {value.strip() for value in args.conditions.split(",") if value.strip()}
    conditions = [row for row in CONDITIONS if row.name in condition_names]
    if condition_names != {row.name for row in conditions}:
        raise ValueError(f"Unknown conditions: {sorted(condition_names - {row.name for row in conditions})}")
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    cases = list(manifest["cases"][: args.max_cases])
    if not models or not conditions or not cases:
        raise ValueError("At least one model, condition, and case are required.")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    runtime_before = _ollama_metadata(args.base_url, models)
    missing_models = [
        model for model, digest in _model_digests(runtime_before).items() if digest is None
    ]
    if missing_models:
        raise RuntimeError(f"Requested Ollama model identities are unavailable: {missing_models}")
    identity = {
        "protocol": "paper9-frozen-cross-model-consumption-factorial-v1",
        "manifest_sha256": manifest_sha256,
        "models": models,
        "model_digests": _model_digests(runtime_before),
        "conditions": [row.name for row in conditions],
        "case_count": len(cases),
        "max_steps": args.max_steps,
        "max_new_tokens": args.max_new_tokens,
    }
    run_manifest = {
        **identity,
        "execution_identity_sha256": _identity_digest(identity),
        "runtime_before": runtime_before,
    }
    run_manifest_path = args.output / "run_manifest.json"
    if run_manifest_path.exists():
        previous = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if previous.get("execution_identity_sha256") != run_manifest["execution_identity_sha256"]:
            raise RuntimeError("Refusing to resume a consumption factorial with changed identity.")
    else:
        run_manifest_path.write_text(
            json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
        )
    rows_path = args.output / "rows.jsonl"
    rows: list[dict[str, object]] = []
    if rows_path.exists():
        rows = [
            json.loads(line)
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    completed = {
        (row["model"], row["condition"], row["repository_id"], row["query_id"])
        for row in rows
    }
    for model in models:
        client = OllamaChatClient(
            args.base_url,
            model,
            max_new_tokens=args.max_new_tokens,
            timeout_seconds=args.timeout_seconds,
        )
        for condition in conditions:
            for case in cases:
                key = (model, condition.name, case["repository_id"], case["query_id"])
                if key in completed:
                    continue
                row = run_case(
                    client,
                    manifest,
                    case,
                    condition,
                    model=model,
                    max_steps=args.max_steps,
                )
                rows.append(row)
                with rows_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                completed.add(key)
                print(
                    json.dumps(
                        {
                            "model": model,
                            "condition": condition.name,
                            "query_id": case["query_id"],
                            "selection_correct": case["selection_correct"],
                            "path_hit": row["relevant_path_hit"],
                            "tool_calls": row["tool_calls"],
                            "failed": row["failed"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    runtime_after = _ollama_metadata(args.base_url, models)
    model_digests_stable = _model_digests(runtime_before) == _model_digests(runtime_after)
    report = {
        "protocol": "paper9-frozen-cross-model-consumption-factorial-v1",
        "evidence_scope": "read-only answer/path task; not autonomous patch utility",
        "manifest_schema": manifest["schema_version"],
        "selection_policy": manifest["selection_policy"],
        "models": models,
        "runtime_before": runtime_before,
        "runtime_after": runtime_after,
        "model_digests_stable": model_digests_stable,
        "conditions": [row.name for row in conditions],
        "case_count": len(cases),
        "selected_record_hashes_identical_across_arms": all(
            len({row["selected_sha256"] for row in rows if row["query_id"] == case["query_id"]})
            == 1
            for case in cases
        ),
        "summaries": summarize(rows),
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if not model_digests_stable:
        raise RuntimeError("Ollama model digest changed during the consumption factorial.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
