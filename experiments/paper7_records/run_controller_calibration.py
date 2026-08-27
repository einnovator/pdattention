"""Calibrate Paper 7 context-control descriptions, model capacity, and protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import time
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.full_pra_context_cases import FullPRAContextCase, full_pra_context_cases
from pra_hf.adaptive_context_runtime import (
    AdaptiveContextRuntime,
    ContextPolicy,
    CursorOperation,
    CursorPolicy,
    DeploymentTopology,
    StoragePolicy,
    TypeContextPolicy,
)
from pra_hf.context_records import RecordType
from pra_hf.context_store import RecordScope
from pra_hf.progressive_context import (
    ControllerConfig,
    ControllerDescriptionLevel,
    ControllerProtocol,
    ContextAction,
    ProgressiveContextRuntime,
)


DEFAULT_OUTPUT = ROOT / "docs/papers/shared/results/paper7_records/full_pra_calibrated"
SEEDS = (11, 23, 37, 53, 71)
PROTOCOL_REVISION = "paper7-controller-calibration-v1"


@dataclass(frozen=True)
class Response:
    value: Mapping[str, object]
    raw: str
    latency_seconds: float
    prompt_tokens: int
    generated_tokens: int
    cache_hit: bool


class OllamaController:
    """Content-addressed Ollama adapter retaining token and latency accounting."""

    def __init__(self, endpoint: str, cache_path: Path) -> None:
        self.endpoint = endpoint.rstrip("/") + "/api/chat"
        self.cache_path = cache_path
        self.cache = (
            json.loads(cache_path.read_text(encoding="utf-8-sig"))
            if cache_path.is_file() else {}
        )

    def chat(
        self,
        config: ControllerConfig,
        prompt: str,
        *,
        seed: int,
        max_tokens: int = 64,
    ) -> Response:
        thinking = "" if config.thinking else " /no_think"
        request = {
            "model": config.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": "Return one valid JSON object only." + thinking},
                {"role": "user", "content": prompt + thinking},
            ],
            "options": {"temperature": 0, "seed": seed, "num_predict": max_tokens},
            "keep_alive": "30m",
        }
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        cached = self.cache.get(key)
        if cached is not None:
            return Response(
                cached["value"], str(cached["raw"]),
                float(cached["latency_seconds"]), int(cached["prompt_tokens"]),
                int(cached["generated_tokens"]), True,
            )
        started = time.perf_counter()
        http_request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(request).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(http_request, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
        latency = time.perf_counter() - started
        raw = str(body.get("message", {}).get("content", "{}"))
        value = _json_object(raw)
        row = {
            "value": value,
            "raw": raw,
            "latency_seconds": latency,
            "prompt_tokens": int(body.get("prompt_eval_count", 0)),
            "generated_tokens": int(body.get("eval_count", 0)),
        }
        self.cache[key] = row
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, indent=2, sort_keys=True), encoding="utf-8"
        )
        return Response(**row, cache_hit=False)


def _json_object(raw: str) -> Mapping[str, object]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        try:
            value = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            value = {}
    return value if isinstance(value, Mapping) else {}


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _runtime(case: FullPRAContextCase, output: Path) -> ProgressiveContextRuntime:
    policies = {record_type: TypeContextPolicy(unit_limit=3) for record_type in RecordType}
    runtime = AdaptiveContextRuntime(
        RecordScope("paper7-controller", case.case_id),
        ContextPolicy(
            topology=DeploymentTopology.REMOTE_MODEL,
            storage=StoragePolicy.ON_DEMAND,
            local_store=output / ".controller_stores" / case.case_id,
            record_policies=policies,
            cursor_policy=CursorPolicy(page_size=4, max_page_size=16),
            persistent_store=False,
        ),
    )
    progressive = ProgressiveContextRuntime(runtime, chunk_tokens=32)
    record = progressive.ingest(
        case.payload,
        record_type=case.record_type,
        capabilities=case.capabilities,
        provenance={"source": "paper7_controller_validation"},
    )
    if case.case_class == "C3_CURSOR":
        cursor = runtime.open_cursor(record.record_id, collection=case.cursor_collection)
        first_page = runtime.fetch_cursor(cursor.cursor_id)
        operations = (
            (CursorOperation.NEXT,)
            if case.cursor_query is None else (CursorOperation.SEARCH,)
        )
        capabilities = replace(
            case.capabilities,
            cursor_available=True,
            cursor_id=cursor.cursor_id,
            has_more=first_page.has_more,
            allowed_cursor_operations=operations,
        )
        progressive.registry.capabilities[record.record_id] = capabilities
    return progressive


def _prompt_safe(value: object) -> object:
    """Remove host-bound opaque IDs that the action-only decoder never emits."""

    if isinstance(value, Mapping):
        return {
            str(key): _prompt_safe(item)
            for key, item in value.items()
            if str(key) not in {"record_id", "cursor_id"}
        }
    if isinstance(value, list):
        return [_prompt_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_prompt_safe(item) for item in value]
    return value


def _visible_context(progressive: ProgressiveContextRuntime) -> list[object]:
    return [
        _prompt_safe(document.payload)
        for document in progressive.registry.documents.values()
    ]


def _allowed_actions(case: FullPRAContextCase, progressive: ProgressiveContextRuntime) -> list[str]:
    record_id = next(iter(progressive.runtime.records))
    capabilities = progressive.registry.capabilities[record_id]
    actions = [ContextAction.CONTINUE, ContextAction.MATERIALIZE_FULL]
    if capabilities.partial_selectors:
        actions.append(ContextAction.MATERIALIZE_MORE)
    if capabilities.searchable:
        actions.append(ContextAction.SEARCH_RECORD)
    if capabilities.cursor_available:
        if CursorOperation.NEXT in capabilities.allowed_cursor_operations:
            actions.append(ContextAction.CURSOR_NEXT)
        else:
            actions.append(ContextAction.CURSOR_QUERY)
    if case.tool_name:
        actions.append(ContextAction.CALL_TOOL)
    return [action.value for action in actions]


def _base_prompt(
    config: ControllerConfig,
    case: FullPRAContextCase,
    progressive: ProgressiveContextRuntime,
) -> str:
    record_id = next(iter(progressive.runtime.records))
    capabilities = progressive.registry.capabilities[record_id]
    return (
        "You control progressive context for a typed PRA result. PRA context operations are "
        "fixed runtime capabilities and are always available as described below. Never invent "
        "hidden content. CALL_TOOL is different from retrieving current backing state.\n\n"
        f"PRA operations:\n{config.description_block}\n\n"
        f"Task: {case.query}\n"
        f"Visible typed context: {json.dumps(_visible_context(progressive), sort_keys=True, default=str)}\n"
        f"Record capabilities: {json.dumps(_prompt_safe(capabilities.prompt_descriptor()), sort_keys=True)}\n"
        f"Allowed actions for this record: {json.dumps(_allowed_actions(case, progressive))}\n"
    )


def _parse_action(value: Mapping[str, object], allowed: Sequence[str]) -> str:
    candidate = str(value.get("context_action", value.get("action", ""))).upper()
    return candidate if candidate in set(allowed) else "INVALID"


def _decide(
    client: OllamaController,
    config: ControllerConfig,
    case: FullPRAContextCase,
    progressive: ProgressiveContextRuntime,
    seed: int,
) -> tuple[str, str, list[Response]]:
    base = _base_prompt(config, case, progressive)
    allowed = _allowed_actions(case, progressive)
    if config.protocol == ControllerProtocol.FLAT:
        choices = json.dumps(allowed)
        response = client.chat(
            config,
            base + (
                "\nChoose exactly one value from this list: " + choices + ". Return one JSON "
                'object such as {"context_action":"CONTINUE"}. Do not echo placeholder words.'
            ),
            seed=seed,
        )
        return _parse_action(response.value, allowed), "flat", [response]

    first = client.chat(
        config,
        base + (
            '\nFirst decide only whether current visible context is sufficient. Return '
            '{"sufficiency":"SUFFICIENT"}, {"sufficiency":"NEED_MORE"}, or '
            '{"sufficiency":"UNCERTAIN"}.'
        ),
        seed=seed,
    )
    sufficiency = str(first.value.get("sufficiency", "UNCERTAIN")).upper()
    if sufficiency == "SUFFICIENT":
        return ContextAction.CONTINUE.value, sufficiency, [first]
    second = client.chat(
        config,
        base + (
            "\nCurrent context was classified as insufficient or uncertain. Select the one "
            "allowed operation that most narrowly obtains the missing evidence. Choose exactly "
            f"one value from {json.dumps([value for value in allowed if value != 'CONTINUE'])}. "
            'Return one JSON object such as {"context_action":"SEARCH_RECORD"}. Do not echo '
            "placeholder words."
        ),
        seed=seed,
    )
    return _parse_action(second.value, allowed), sufficiency, [first, second]


def _configs() -> tuple[ControllerConfig, ...]:
    return (
        ControllerConfig("qwen3:0.6b", "D0", protocol="flat"),
        ControllerConfig("qwen3:0.6b", "D2", protocol="flat"),
        ControllerConfig("llama3.2:3b", "D0", protocol="flat"),
        ControllerConfig("llama3.2:3b", "D1", protocol="flat"),
        ControllerConfig("llama3.2:3b", "D2", protocol="flat"),
        ControllerConfig("llama3.2:3b", "D2", protocol="hierarchical"),
    )


def _row_key(row: Mapping[str, object]) -> tuple[str, int, str]:
    return str(row["case_id"]), int(row["seed"]), str(row["controller_fingerprint"])


def _load_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _append(path: Path, row: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _aggregate(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    result = []
    for key, values in sorted(grouped.items()):
        required = [row for row in values if int(row["need_more_target"])]
        sufficient = [row for row in values if not int(row["need_more_target"])]
        predicted_more = [row for row in values if int(row["need_more_predicted"])]
        result.append({
            **dict(zip(fields, key)),
            "n": len(values),
            "decision_accuracy": statistics.fmean(float(row["decision_correct"]) for row in values),
            "need_more_recall": statistics.fmean(float(row["need_more_predicted"]) for row in required),
            "false_escalation": statistics.fmean(float(row["need_more_predicted"]) for row in sufficient),
            "operation_accuracy_given_need_more": (
                statistics.fmean(float(row["operation_correct"]) for row in predicted_more)
                if predicted_more else 0.0
            ),
            "prompt_tokens": statistics.fmean(float(row["prompt_tokens"]) for row in values),
            "generated_tokens": statistics.fmean(float(row["generated_tokens"]) for row in values),
            "latency_seconds": statistics.fmean(float(row["latency_seconds"]) for row in values),
            "fixed_description_token_estimate": values[0]["fixed_description_token_estimate"],
        })
    return result


def _postprocess(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    config_fields = ("model", "description_level", "controller_protocol", "thinking", "controller_fingerprint")
    configs = _aggregate(rows, config_fields)
    _write_csv(output / "controller_description_calibration.csv", configs)
    _write_csv(output / "controller_model_calibration.csv", _aggregate(rows, ("model",)))
    _write_csv(
        output / "controller_protocol_calibration.csv",
        _aggregate(rows, ("model", "description_level", "controller_protocol")),
    )
    _write_csv(output / "controller_calibration_rows.csv", list(rows))
    best_accuracy = max(float(row["decision_accuracy"]) for row in configs)
    pareto = [row for row in configs if float(row["decision_accuracy"]) >= best_accuracy - 0.02]
    selected = min(
        pareto,
        key=lambda row: (
            float(row["false_escalation"]),
            -float(row["need_more_recall"]),
            -float(row["operation_accuracy_given_need_more"]),
            float(row["latency_seconds"]),
            float(row["prompt_tokens"]),
        ),
    )
    config = ControllerConfig(
        str(selected["model"]),
        str(selected["description_level"]),
        protocol=str(selected["controller_protocol"]),
        thinking=bool(selected["thinking"]),
    )
    (output / "selected_controller.json").write_text(
        json.dumps({
            "selection_rule": "within 0.02 of best validation decision accuracy, then minimize false escalation and cost",
            "config": {
                "model": config.model,
                "description_level": config.description_level.value,
                "protocol": config.protocol.value,
                "thinking": config.thinking,
                "fingerprint": config.fingerprint,
                "description_block": config.description_block,
            },
            "validation_metrics": selected,
            "validation_cases": len({str(row["case_id"]) for row in rows}),
            "seeds": list(SEEDS),
        }, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--seed-limit", type=int)
    parser.add_argument("--config-limit", type=int)
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "controller_calibration_checkpoint.jsonl"
    rows = _load_rows(checkpoint)
    if not args.postprocess_only:
        client = OllamaController(
            args.endpoint, args.output_dir / "controller_response_cache.json"
        )
        cases = [case for case in full_pra_context_cases() if case.partition == "validation"]
        seeds = SEEDS[: args.seed_limit]
        configs = _configs()[: args.config_limit]
        if args.case_limit is not None:
            cases = cases[: args.case_limit]
        completed = {_row_key(row) for row in rows}
        total = len(cases) * len(seeds) * len(configs)
        count = 0
        for config in configs:
            for case in cases:
                for seed in seeds:
                    count += 1
                    key = (case.case_id, seed, config.fingerprint)
                    if key in completed:
                        continue
                    progressive = _runtime(case, args.output_dir)
                    action, sufficiency, responses = _decide(
                        client, config, case, progressive, seed
                    )
                    need_target = case.expected_action != ContextAction.CONTINUE
                    need_predicted = action != ContextAction.CONTINUE.value
                    row = {
                        "protocol_revision": PROTOCOL_REVISION,
                        "case_id": case.case_id,
                        "case_class": case.case_class,
                        "omission_stratum": case.omission_stratum.value,
                        "seed": seed,
                        "model": config.model,
                        "description_level": config.description_level.value,
                        "controller_protocol": config.protocol.value,
                        "thinking": int(config.thinking),
                        "controller_fingerprint": config.fingerprint,
                        "expected_action": case.expected_action.value,
                        "predicted_action": action,
                        "sufficiency_output": sufficiency,
                        "need_more_target": int(need_target),
                        "need_more_predicted": int(need_predicted),
                        "insufficiency_correct": int(need_target == need_predicted),
                        "operation_correct": int(action == case.expected_action.value),
                        "decision_correct": int(action == case.expected_action.value),
                        "model_passes": len(responses),
                        "prompt_tokens": sum(response.prompt_tokens for response in responses),
                        "generated_tokens": sum(response.generated_tokens for response in responses),
                        "latency_seconds": sum(response.latency_seconds for response in responses),
                        "fixed_description_token_estimate": math.ceil(
                            len(config.description_block.encode("utf-8")) / 4
                        ),
                        "raw": " | ".join(response.raw for response in responses),
                    }
                    rows.append(row)
                    _append(checkpoint, row)
                    progressive.runtime.store.close()
                    print(
                        f"[{count}/{total}] {config.model} {config.description_level.value} "
                        f"{config.protocol.value} {case.case_id} {action}",
                        flush=True,
                    )
    _postprocess(rows, args.output_dir)


if __name__ == "__main__":
    main()
