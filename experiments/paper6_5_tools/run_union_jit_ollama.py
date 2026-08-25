"""Low-memory five-presentation runner for Paper 6.5 E4--E6.

Discovery and callable ingestion are frozen before this runner starts. It reads
their JSON artifacts and therefore needs neither PyTorch nor Transformers.
Qwen3 Q4 inference is delegated to a CPU-only local Ollama runner; host-side
schema validation and every materialization metric remain explicit here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "docs/papers/shared/results/paper6_5_tools/auto_union_records"
HF_SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
SEEDS = (11, 23, 37, 53, 71)
CONDITIONS = ("top1_jit", "union_jit_k2", "union_jit_k4", "union_jit_k6", "union_jit_k8", "static_oracle", "static_graph", "all_tools")
CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


TASKS = (
    {
        "task_id": "m4-user-3",
        "query": "Find alice@example.com, validate the account, then set its status to reviewed.",
        "unsafe": ("delete_user",),
        "steps": (
            ("search_user", {"email": "alice@example.com"}, {"user_id": "u17"}),
            ("validate_user", {"user_id": "u17"}, {"valid": True}),
            ("update_user", {"user_id": "u17", "status": "reviewed"}, {"changed": True}),
        ),
    },
    {
        "task_id": "m4-doc-4",
        "query": "Find the PRA Notes document, read it, export it as pdf, and create a report titled PRA digest.",
        "unsafe": ("purge_archive",),
        "steps": (
            ("search_document", {"title": "PRA Notes"}, {"document_id": "d42"}),
            ("read_document", {"document_id": "d42"}, {"document_id": "d42", "text": "PRA notes"}),
            ("export_document", {"document_id": "d42", "format": "pdf"}, {"artifact_id": "artifact-d42-pdf"}),
            ("create_report", {"artifact_id": "artifact-d42-pdf", "title": "PRA digest"}, {"report_id": "report-7"}),
        ),
    },
    {
        "task_id": "m4-repo-5",
        "query": "Find repository pra-core, retrieve it, create issue Routing audit, mark that issue open, then notify user u17 about the change.",
        "unsafe": ("delete_user",),
        "steps": (
            ("search_repository", {"name": "pra-core"}, {"repository_id": "repo9"}),
            ("get_repository", {"repository_id": "repo9"}, {"repository_id": "repo9", "owner": "team-a"}),
            ("create_issue", {"repository_id": "repo9", "title": "Routing audit"}, {"issue_id": "issue-4"}),
            ("update_issue", {"issue_id": "issue-4", "status": "open"}, {"changed": True}),
            ("notify_user", {"user_id": "u17", "message": "Routing audit issue is open"}, {"notified": True}),
        ),
    },
)

SINGLE_TASKS = (
    ("m2-user", "Retrieve user u17.", "get_user", {"user_id": "u17"}),
    ("m2-document", "Read document d42.", "read_document", {"document_id": "d42"}),
    ("m2-repository", "Retrieve repository repo9.", "get_repository", {"repository_id": "repo9"}),
    ("m2-export", "Export document d42 as pdf.", "export_document", {"document_id": "d42", "format": "pdf"}),
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json_atomic(path: Path, value: dict) -> None:
    """Persist a restart checkpoint without exposing a half-written JSON file."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _save_checkpoint(
    path: Path,
    *,
    model: str,
    seeds: tuple[int, ...],
    e4_rows: list[dict],
    atomic_rows: list[dict],
    materialization_rows: list[dict],
    elapsed_seconds_completed: float,
) -> None:
    _write_json_atomic(path, {
        "schema_version": "1.0",
        "model": model,
        "seeds": list(seeds),
        "e4_rows": e4_rows,
        "atomic_rows": atomic_rows,
        "materialization_rows": materialization_rows,
        "elapsed_seconds_completed": elapsed_seconds_completed,
    })


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_resources(results: Path) -> tuple[dict, ...]:
    rows = []
    for line in (results / "auto_tool_records.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        properties = {}
        required = []
        for parameter in record["schema"]["inputs"]:
            field = dict(parameter["json_schema"])
            if parameter["description"]:
                field["description"] = parameter["description"]
            properties[parameter["name"]] = field
            if parameter["required"]:
                required.append(parameter["name"])
        uri = f"!!ref:tool:{record['namespace']}:{record['qualified_name']}:{record['version']}!!"
        rows.append({
            "uri": uri,
            "name": record["name"],
            "description": record["description"],
            "schema": {
                "type": "function",
                "function": {
                    "name": record["name"],
                    "description": record["description"],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            },
            "version": record["version"],
        })
    return tuple(rows)


def _load_candidates(results: Path) -> dict[tuple[str, int, str], dict]:
    payload = json.loads((results / "union_jit_candidate_sets.json").read_text(encoding="utf-8"))
    return {(row["task_id"], int(row["step"]), row["condition"]): row for row in payload["rows"]}


def _prompt(query: str, seed: int) -> str:
    prefixes = (
        "Use exactly one available tool. ",
        "Call the matching function with all required arguments. ",
        "Complete this request with one function call. ",
        "Choose the correct available function and call it. ",
        "Return a single tool call for this request. ",
    )
    return prefixes[SEEDS.index(seed)] + query


def _parse_call(text: str) -> dict | None:
    match = CALL_PATTERN.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload.get("name"), str) and isinstance(payload.get("arguments"), dict) else None


class OllamaClient:
    def __init__(self, model: str, tokenizer: Tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, messages: list[dict], resources: list[dict], max_new_tokens: int = 48) -> tuple[str, dict]:
        formatted = []
        for message in messages:
            value = dict(message)
            if value["role"] == "user" and "/no_think" not in value["content"]:
                value["content"] = "/no_think " + value["content"]
            call = _parse_call(value.get("content", "")) if value["role"] == "assistant" else None
            if call:
                value = {"role": "assistant", "content": "", "tool_calls": [{"function": call}]}
            formatted.append(value)
        payload = {
            "model": self.model,
            "stream": False,
            "keep_alive": 0,
            "messages": formatted,
            "tools": [row["schema"] for row in resources],
            "options": {"temperature": 0, "num_predict": max_new_tokens, "num_gpu": 0},
        }
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error = None
        for attempt in range(3):
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    result = json.loads(response.read())
                break
            except urllib.error.HTTPError as exc:
                last_error = exc.read().decode("utf-8", errors="replace")
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"Ollama generation failed after retries: {last_error}")
        message = result.get("message", {})
        calls = message.get("tool_calls", ())
        if calls:
            function = calls[0]["function"]
            text = "<tool_call>" + json.dumps({"name": function.get("name"), "arguments": function.get("arguments", {})}, separators=(",", ":")) + "</tool_call>"
        else:
            text = str(message.get("content", ""))
        return text, {
            "prompt_tokens": int(result.get("prompt_eval_count", 0)),
            "generated_tokens": int(result.get("eval_count", 0)),
            "generation_seconds": time.perf_counter() - started,
            "tool_definition_tokens": sum(len(self.tokenizer.encode(json.dumps(row["schema"])).ids) for row in resources),
        }


def _validate(call: dict | None, expected_name: str, expected_arguments: dict, original: dict) -> tuple[bool, bool, str, set[str]]:
    if call is None:
        return False, False, "malformed_call", set(expected_arguments)
    tool_ok = call["name"] == expected_name
    arguments = call["arguments"]
    required = set(original["schema"]["function"]["parameters"]["required"])
    missing = required - set(arguments)
    if not tool_ok:
        return False, False, "wrong_tool", missing
    if missing:
        return True, False, "missing_required_argument", missing
    if arguments != expected_arguments:
        return True, False, "arguments_mismatch", missing
    return True, True, "executed", missing


def _record_text(resource: dict, provenance: dict | None = None) -> str:
    payload = {"uri": resource["uri"], "version": resource["version"], "schema": resource["schema"], "side_effect": "none"}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    header = json.dumps({"id": resource["uri"], "type": "tool_definition", "version": resource["version"], "fingerprint": fingerprint}, sort_keys=True)
    return f"<<<PRA_RECORD {header}>>>\n{json.dumps(payload, sort_keys=True)}\n<<<END_PRA_RECORD {resource['uri']}>>>"


def _materialization(candidate: dict, resources: list[dict], tokenizer: Tokenizer, *, overflow_case: str = "fits") -> dict:
    texts = [_record_text(row) for row in resources]
    selected_bytes = sum(len(value.encode()) for value in texts) + max(len(texts) - 1, 0)
    fits = overflow_case == "fits"
    payload = "\n".join(texts) if fits else ""
    tokens = len(tokenizer.encode(payload).ids) if payload else 0
    config = json.loads((HF_SNAPSHOT / "config.json").read_text(encoding="utf-8"))
    head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
    kv_per_token = 2 * config["num_hidden_layers"] * config["num_key_value_heads"] * head_dim * 2
    return {
        "status": "materialized" if fits else "narrow_required",
        "selected_record_ids": "|".join(row["uri"] for row in resources),
        "materialized_record_ids": "|".join(row["uri"] for row in resources) if fits else "",
        "records_selected": 1 + len(resources),
        "records_materialized": 1 + len(resources) if fits else 0,
        "child_records_selected": len(resources),
        "child_records_materialized": len(resources) if fits else 0,
        "record_coverage": 1.0 if fits else 0.0,
        "serialized_bytes": len(payload.encode()),
        "serialized_tokens": tokens,
        "native_kv_bytes": tokens * kv_per_token,
        "partial_record_count": 0,
        "atomicity_violations": 0,
        "upstream_selection_preserved": int(fits),
        "overflow_behavior": "request_narrow",
        "declared_budget_bytes": selected_bytes if fits else max(selected_bytes - 1, 0),
    }


def _generic_schema_chunk(query: str, resource: dict) -> str:
    """Select one internal schema segment using a content-only lexical router."""

    function = resource["schema"]["function"]
    query_terms = set(re.findall(r"[a-z0-9]+", query.casefold()))
    segments = {
        "description": f"{function['name']} {function.get('description', '')}",
        "parameters": json.dumps(function.get("parameters", {}), sort_keys=True),
    }
    scores = {
        name: len(query_terms & set(re.findall(r"[a-z0-9]+", text.casefold())))
        for name, text in segments.items()
    }
    return max(("description", "parameters"), key=lambda name: (scores[name], name == "description"))


def _partial(resource: dict, condition: str, query: str = "") -> dict:
    value = json.loads(json.dumps(resource))
    function = value["schema"]["function"]
    if condition == "generic_chunk_reroute":
        condition = "description_only" if _generic_schema_chunk(query, resource) == "description" else "parameters_only"
    if condition == "parameters_only":
        function["description"] = ""
    elif condition == "description_only":
        function["parameters"] = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    elif condition == "drop_required_field":
        parameters = function["parameters"]
        if parameters["required"]:
            removed = parameters["required"].pop()
            parameters["properties"].pop(removed, None)
    return value


def _summaries(e4_rows: list[dict], atomic_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    e4 = []
    for condition in CONDITIONS:
        summaries = [row for row in e4_rows if row["row_type"] == "summary" and row["condition"] == condition]
        steps = [row for row in e4_rows if row["row_type"] == "step" and row["condition"] == condition]
        e4.append({
            "condition": condition,
            "workflows": len(summaries),
            "task_success": sum(row["task_success"] for row in summaries) / len(summaries),
            "candidate_required_recall": sum(row["expected_tool"] in row["candidate_names"].split("|") for row in steps) / len(steps),
            "wrong_tool_calls": sum(row["wrong_tool_calls"] for row in summaries) / len(summaries),
            "unsafe_calls": sum(row["unsafe_calls"] for row in summaries) / len(summaries),
            "mean_jit_steps": sum(row["jit_steps"] for row in summaries) / len(summaries),
            "mean_context_tokens": sum(row["definition_tokens"] for row in summaries) / len(summaries),
            "mean_disclosed_tools": sum(row["total_disclosed_tools"] for row in summaries) / len(summaries),
        })
    atomic = []
    for condition in sorted({row["condition"] for row in atomic_rows}):
        subset = [row for row in atomic_rows if row["condition"] == condition]
        atomic.append({
            "condition": condition,
            "calls": len(subset),
            "call_validity": sum(row["call_valid"] for row in subset) / len(subset),
            "argument_accuracy": sum(row["arguments_correct"] for row in subset) / len(subset),
            "execution_acceptance": sum(row["execution_accepted"] for row in subset) / len(subset),
            "required_argument_omission": sum(row["required_argument_omission"] for row in subset) / len(subset),
            "schema_error": sum(row["schema_error"] for row in subset) / len(subset),
        })
    return e4, atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:0.6b")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--fresh", action="store_true", help="Discard a partial condition-level checkpoint.")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    resources = _load_resources(args.output_dir)
    by_uri = {row["uri"]: row for row in resources}
    by_name = {row["name"]: row for row in resources}
    candidates = _load_candidates(args.output_dir)
    tokenizer = Tokenizer.from_file(str(HF_SNAPSHOT / "tokenizer.json"))
    client = OllamaClient(args.model, tokenizer)
    started = time.perf_counter()
    checkpoint_path = args.output_dir / "union_jit_ollama_checkpoint.json"
    if args.fresh and checkpoint_path.exists():
        checkpoint_path.unlink()
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("model") != args.model or tuple(checkpoint.get("seeds", ())) != seeds:
            raise ValueError("Checkpoint model/seeds differ; pass --fresh to start a new matrix.")
        e4_rows = checkpoint.get("e4_rows", [])
        atomic_rows = checkpoint.get("atomic_rows", [])
        materialization_rows = checkpoint.get("materialization_rows", [])
        prior_elapsed_seconds = float(checkpoint.get("elapsed_seconds_completed", 0.0))
        print(f"Resuming checkpoint with {len(e4_rows)} E4 and {len(atomic_rows)} E5/E6 rows.", flush=True)
    else:
        e4_rows = []
        atomic_rows = []
        materialization_rows = []
        prior_elapsed_seconds = 0.0
    completed_e4 = {
        (int(row["seed"]), row["task_id"], row["condition"])
        for row in e4_rows if row.get("row_type") == "summary"
    }
    for seed in seeds:
        for task in TASKS:
            for condition in CONDITIONS:
                if (seed, task["task_id"], condition) in completed_e4:
                    continue
                messages = [{"role": "user", "content": _prompt(task["query"], seed) + " Execute one step at a time."}]
                observations = []
                completed = wrong = unsafe = definition_tokens = prompt_tokens = 0
                generation_seconds = 0.0
                disclosed_unique = set()
                failure = ""
                attempted = 0
                for step_index, (expected_name, expected_args, output) in enumerate(task["steps"]):
                    attempted += 1
                    candidate = candidates[(task["task_id"], step_index, condition)]
                    disclosed = [by_uri[uri] for uri in candidate["candidate_uris"]]
                    randomizer = __import__("random").Random(seed * 100 + step_index)
                    randomizer.shuffle(disclosed)
                    disclosed_unique.update(row["uri"] for row in disclosed)
                    materialization_rows.append({
                        "experiment": "E4", "seed": seed, "task_id": task["task_id"], "condition": condition, "step": step_index + 1,
                        **_materialization(candidate, disclosed, tokenizer),
                    })
                    text, cost = client.generate(messages, disclosed)
                    call = _parse_call(text)
                    tool_ok, arguments_ok, reason, _ = _validate(call, expected_name, expected_args, by_name[expected_name])
                    wrong += int(not tool_ok)
                    unsafe += int(call is not None and call["name"] in task["unsafe"])
                    definition_tokens += cost["tool_definition_tokens"]
                    prompt_tokens += cost["prompt_tokens"]
                    generation_seconds += cost["generation_seconds"]
                    e4_rows.append({
                        "row_type": "step", "seed": seed, "task_id": task["task_id"], "condition": condition, "step": step_index + 1,
                        "expected_tool": expected_name, "generated_tool": "" if call is None else call["name"], "tool_correct": int(tool_ok),
                        "arguments_correct": int(arguments_ok), "executed": int(arguments_ok), "failure_reason": "" if arguments_ok else reason,
                        "candidate_count": len(disclosed), "candidate_names": "|".join(row["name"] for row in disclosed),
                        "prompt_tokens": cost["prompt_tokens"], "definition_tokens": cost["tool_definition_tokens"], "generation_seconds": cost["generation_seconds"],
                    })
                    if not arguments_ok:
                        failure = reason
                        break
                    completed += 1
                    observation = json.dumps(output, sort_keys=True)
                    observations.append(observation)
                    messages.extend((
                        {"role": "assistant", "content": text},
                        {"role": "tool", "content": observation},
                        {"role": "user", "content": "Continue the workflow with exactly one next tool call."},
                    ))
                success = completed == len(task["steps"])
                e4_rows.append({
                    "row_type": "summary", "seed": seed, "task_id": task["task_id"], "condition": condition, "step": 0,
                    "expected_tool": "", "generated_tool": "", "tool_correct": int(success), "arguments_correct": int(success), "executed": int(success),
                    "failure_reason": failure, "candidate_count": "", "candidate_names": "", "prompt_tokens": prompt_tokens,
                    "definition_tokens": definition_tokens, "generation_seconds": generation_seconds, "task_success": int(success),
                    "wrong_tool_calls": wrong, "unsafe_calls": unsafe, "plan_revisions": 0, "jit_steps": attempted,
                    "total_disclosed_tools": len(disclosed_unique),
                })
                print(f"[E4 {seed} {task['task_id']} {condition}] {completed}/{len(task['steps'])}", flush=True)
                _save_checkpoint(
                    checkpoint_path, model=args.model, seeds=seeds, e4_rows=e4_rows,
                    atomic_rows=atomic_rows, materialization_rows=materialization_rows,
                    elapsed_seconds_completed=prior_elapsed_seconds + time.perf_counter() - started,
                )

    atomic_conditions = (
        "record_full", "flat_serialization", "parameters_only", "description_only",
        "generic_chunk_reroute", "drop_required_field",
    )
    completed_atomic = {(int(row["seed"]), row["task_id"], row["condition"]) for row in atomic_rows}
    for seed in seeds:
        for task_id, query, expected_name, expected_args in SINGLE_TASKS:
            original = by_name[expected_name]
            for condition in atomic_conditions:
                if (seed, task_id, condition) in completed_atomic:
                    continue
                selected_schema_chunk = _generic_schema_chunk(query, original) if condition == "generic_chunk_reroute" else ""
                visible = _partial(original, condition, query)
                if condition == "flat_serialization":
                    prompt = query + "\nThe available function is serialized as ordinary text. Reply with <tool_call>{\"name\":...,\"arguments\":{...}}</tool_call>.\n" + json.dumps(original["schema"], sort_keys=True)
                    disclosed = []
                else:
                    prompt = _prompt(query, seed)
                    disclosed = [visible]
                text, cost = client.generate([{"role": "user", "content": prompt}], disclosed)
                call = _parse_call(text)
                tool_ok, arguments_ok, reason, missing = _validate(call, expected_name, expected_args, original)
                atomic_rows.append({
                    "seed": seed, "task_id": task_id, "condition": condition, "record_aware": int(condition == "record_full"),
                    "partial_tool": int(condition in {"description_only", "generic_chunk_reroute", "drop_required_field"}),
                    "selected_schema_chunk": selected_schema_chunk, "call_valid": int(call is not None),
                    "tool_correct": int(tool_ok), "arguments_correct": int(arguments_ok), "execution_accepted": int(arguments_ok),
                    "execution_reason": reason, "required_argument_omission": int(bool(missing)), "omitted_arguments": "|".join(sorted(missing)),
                    "schema_error": int(reason == "missing_required_argument"), "incorrect_call": int(not arguments_ok), "unsafe_call": 0,
                    "prompt_tokens": cost["prompt_tokens"], "definition_tokens": cost["tool_definition_tokens"], "generation_seconds": cost["generation_seconds"],
                })
                if condition == "record_full":
                    candidate = {"candidate_uris": [original["uri"]]}
                    for overflow_case in ("fits", "narrow_required"):
                        materialization_rows.append({
                            "experiment": "E5", "seed": seed, "task_id": task_id, "condition": condition, "step": 0, "overflow_case": overflow_case,
                            **_materialization(candidate, [original], tokenizer, overflow_case=overflow_case),
                        })
                print(f"[E5/E6 {seed} {task_id} {condition}] accepted={int(arguments_ok)}", flush=True)
                _save_checkpoint(
                    checkpoint_path, model=args.model, seeds=seeds, e4_rows=e4_rows,
                    atomic_rows=atomic_rows, materialization_rows=materialization_rows,
                    elapsed_seconds_completed=prior_elapsed_seconds + time.perf_counter() - started,
                )

    e4_summary, atomic_summary = _summaries(e4_rows, atomic_rows)
    _write_csv(args.output_dir / "union_jit_results.csv", e4_rows)
    _write_csv(args.output_dir / "union_jit_summary.csv", e4_summary)
    _write_csv(args.output_dir / "record_materialization_results.csv", materialization_rows)
    _write_csv(args.output_dir / "tool_atomicity_controls.csv", atomic_rows)
    _write_csv(args.output_dir / "tool_atomicity_summary.csv", atomic_summary)
    frontier = list(csv.DictReader((args.output_dir / "union_recall_frontier.csv").open(encoding="utf-8")))
    recall_gate = all(
        float(next(row["required_recall"] for row in frontier if row["strategy"] == "diversity_union" and int(row["max_candidates"]) == k))
        > float(next(row["required_recall"] for row in frontier if row["strategy"] == "fused_score" and int(row["max_candidates"]) == k))
        for k in (2, 4, 6, 8)
    )
    full = next(row for row in atomic_summary if row["condition"] == "record_full")
    partial = [
        row for row in atomic_summary
        if row["condition"] in {"description_only", "generic_chunk_reroute", "drop_required_field"}
    ]
    findings = {
        "schema_version": "1.0", "model_id": args.model, "model_frozen": True, "generation_backend": "ollama",
        "weight_precision": "Q4_K_M", "seeds": list(seeds), "e4": e4_summary, "e5_e6": atomic_summary,
        "gates": {
            "union_default": False,
            "union_recall_dominates_fusion": recall_gate,
            "record_atomic_default": full["execution_acceptance"] >= max(row["execution_acceptance"] for row in partial),
            "paper7_progressive_detail": "not_implemented",
        },
        "runtime": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": platform.python_version(),
            "git_sha": _git("rev-parse", "HEAD"), "git_branch": _git("branch", "--show-current"), "processor": platform.processor(),
            "ollama_model": args.model, "ollama_options": {"num_gpu": 0, "temperature": 0, "keep_alive": 0},
        },
        "elapsed_seconds": prior_elapsed_seconds + time.perf_counter() - started,
        "execution_boundary": "registered deterministic handlers; discovery never authorizes execution",
    }
    (args.output_dir / "union_record_findings.json").write_text(json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8")
    checkpoint_path.unlink(missing_ok=True)
    print(json.dumps(findings, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
