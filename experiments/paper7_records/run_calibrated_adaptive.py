"""Evaluate frozen Paper 7 routing and controller choices on held-out cases.

Production Qwen routing is run once by ``run_full_pra_reachability.py``. This
runner replays those immutable original-chunk selections across controller
seeds, so controller uncertainty is not confounded with repeated indexing.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.full_pra_context_cases import FullPRAContextCase, full_pra_context_cases
from experiments.paper7_records.run_controller_calibration import (
    OllamaController,
    SEEDS,
    _decide,
    _runtime,
)
from pra_hf.progressive_context import (
    ContextAction,
    ContextDecision,
    ControllerConfig,
    ProgressiveContextRuntime,
)


DEFAULT_OUTPUT = ROOT / "docs/papers/shared/results/paper7_records/full_pra_calibrated"
POLICIES = (
    "FULL",
    "COMPACT_ONLY",
    "PRA_COMPACT",
    "PRA_FALLBACK",
    "PRA_NATIVE",
    "MODEL_ONLY",
    "PRA_ADAPTIVE",
    "PRA_ADAPTIVE_ORACLE",
    "CCR_TOOL",
)
PROTOCOL_REVISION = "paper7-calibrated-adaptive-v1"


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["case_id"]: row for row in csv.DictReader(handle)}


def _load_checkpoint(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _append_checkpoint(path: Path, row: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _payload_text(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(
        value, sort_keys=True, ensure_ascii=True, default=str
    )


def _visible_text(runtime: ProgressiveContextRuntime) -> str:
    return "\n".join(
        _payload_text(document.payload)
        for document in runtime.registry.documents.values()
    )


def _contains_answer(value: object, case: FullPRAContextCase) -> bool:
    return case.expected_answer.casefold() in _payload_text(value).casefold()


def _selected_chunks(row: Mapping[str, str], field: str) -> tuple[dict, ...]:
    value = json.loads(row[field])
    return tuple(dict(chunk) for chunk in value)


def _controller_config(path: Path) -> ControllerConfig:
    value = json.loads(path.read_text(encoding="utf-8"))["config"]
    config = ControllerConfig(
        value["model"], value["description_level"],
        protocol=value["protocol"], thinking=bool(value["thinking"]),
    )
    if config.fingerprint != value["fingerprint"]:
        raise ValueError("Selected controller fingerprint does not match its frozen config.")
    return config


def _record_id(runtime: ProgressiveContextRuntime) -> str:
    return next(iter(runtime.runtime.records))


def _execute_action(
    runtime: ProgressiveContextRuntime,
    case: FullPRAContextCase,
    action: str,
) -> tuple[object | None, int, int, int, float, bool]:
    """Execute a model-selected operation using fixture-owned bounded arguments."""

    record_id = _record_id(runtime)
    capabilities = runtime.registry.capabilities[record_id]
    try:
        parsed = ContextAction(action)
    except ValueError:
        return None, 0, 0, 1, 0.0, False
    if parsed == ContextAction.CONTINUE:
        result = runtime.execute(ContextDecision(parsed))
    elif parsed == ContextAction.MATERIALIZE_FULL:
        result = runtime.materialize_full(record_id)
    elif parsed == ContextAction.MATERIALIZE_MORE and case.selector:
        result = runtime.materialize_more(record_id, case.selector)
    elif parsed == ContextAction.SEARCH_RECORD and case.search_query:
        result = runtime.search_record(record_id, case.search_query)
    elif parsed == ContextAction.CURSOR_NEXT and capabilities.cursor_id:
        result = runtime.cursor_next(capabilities.cursor_id)
    elif parsed == ContextAction.CURSOR_QUERY and capabilities.cursor_id and case.cursor_query:
        result = runtime.cursor_query(capabilities.cursor_id, case.cursor_query)
    elif parsed == ContextAction.CALL_TOOL and case.tool_payload is not None:
        payload = case.tool_payload
        size = len(_payload_text(payload).encode("utf-8"))
        return payload, size, size, 1, 0.0, True
    else:
        return None, 0, 0, 1, 0.0, False
    return (
        result.payload,
        result.payload_bytes,
        result.network_bytes,
        result.model_passes,
        result.latency_seconds,
        result.success,
    )


def _controller_decision(
    client: OllamaController,
    config: ControllerConfig,
    case: FullPRAContextCase,
    runtime: ProgressiveContextRuntime,
    seed: int,
) -> tuple[str, int, int, float, int]:
    action, _, responses = _decide(client, config, case, runtime, seed)
    return (
        action,
        sum(response.prompt_tokens for response in responses),
        sum(response.generated_tokens for response in responses),
        sum(response.latency_seconds for response in responses),
        len(responses),
    )


def _new_row(
    policy: str,
    case: FullPRAContextCase,
    seed: int,
    config: ControllerConfig,
    address: Mapping[str, str],
) -> dict[str, object]:
    compact_bytes = int(address["compact_bytes"])
    return {
        "protocol_revision": PROTOCOL_REVISION,
        "case_id": case.case_id,
        "partition": case.partition,
        "case_class": case.case_class,
        "omission_stratum": case.omission_stratum.value,
        "seed": seed,
        "policy": policy,
        "controller_fingerprint": config.fingerprint if policy in {
            "MODEL_ONLY", "PRA_ADAPTIVE", "CCR_TOOL"
        } else "",
        "expected_action": case.expected_action.value,
        "predicted_action": "",
        "need_more_target": int(case.expected_action != ContextAction.CONTINUE),
        "need_more_predicted": 0,
        "operation_correct": 0,
        "runtime_recovery": 0,
        "evidence_visible": 0,
        "final_use_given_visible": 0,
        "task_success": 0,
        "under_expansion": 0,
        "over_expansion": 0,
        "compact_bytes": compact_bytes,
        "compact_tokens_estimate": (compact_bytes + 3) // 4,
        "materialized_bytes": 0,
        "materialized_tokens": 0,
        "active_kv_tokens": 0,
        "native_kv_tokens": 0,
        "ingestion_seconds": 0.0,
        "routing_seconds": 0.0,
        "controller_seconds": 0.0,
        "runtime_seconds": 0.0,
        "controller_prompt_tokens": 0,
        "controller_generated_tokens": 0,
        "model_passes": 1,
        "tool_roundtrips": 0,
        "network_bytes": 0,
    }


def _evaluate(
    policy: str,
    case: FullPRAContextCase,
    seed: int,
    config: ControllerConfig,
    client: OllamaController,
    address: Mapping[str, str],
    output: Path,
    controller_decisions: dict[tuple[str, int, str], tuple[str, int, int, float, int]],
) -> dict[str, object]:
    runtime = _runtime(case, output)
    row = _new_row(policy, case, seed, config, address)
    extra_payload: object | None = None
    expected_now = case.expected_action.value

    if policy == "FULL":
        extra_payload, size, network, passes, elapsed, ok = _execute_action(
            runtime, case, ContextAction.MATERIALIZE_FULL.value
        )
        row.update(materialized_bytes=size, materialized_tokens=(size + 3) // 4,
                   active_kv_tokens=(size + 3) // 4, network_bytes=network,
                   model_passes=passes, runtime_seconds=elapsed,
                   predicted_action=ContextAction.MATERIALIZE_FULL.value,
                   operation_correct=int(case.expected_action == ContextAction.MATERIALIZE_FULL))
    elif policy in {"COMPACT_ONLY", "PRA_COMPACT"}:
        row["predicted_action"] = ContextAction.CONTINUE.value
        row["operation_correct"] = int(case.expected_action == ContextAction.CONTINUE)
        if policy == "PRA_COMPACT":
            row["routing_seconds"] = 0.0
    elif policy in {"PRA_FALLBACK", "PRA_NATIVE"}:
        field = "fallback_selected_chunks" if policy == "PRA_FALLBACK" else "native_selected_chunks"
        chunks = _selected_chunks(address, field)
        result = runtime.materialize_backing_chunks(
            _record_id(runtime), case.query, chunks,
            selection_policy=("PRA_FALLBACK" if policy == "PRA_FALLBACK" else address["native_selection_policy"]),
        )
        extra_payload = result.payload
        row.update(
            predicted_action="AUTO_ROUTE",
            runtime_recovery=int(_contains_answer(extra_payload, case)),
            materialized_bytes=result.payload_bytes,
            materialized_tokens=(result.payload_bytes + 3) // 4,
            active_kv_tokens=(
                int(address["native_materialized_kv_tokens"])
                if policy == "PRA_NATIVE" else (result.payload_bytes + 3) // 4
            ),
            native_kv_tokens=(
                int(address["native_materialized_kv_tokens"])
                if policy == "PRA_NATIVE" else 0
            ),
            ingestion_seconds=float(address["indexing_seconds"]) if policy == "PRA_NATIVE" else 0.0,
            routing_seconds=float(address["routing_seconds"]) if policy == "PRA_NATIVE" else 0.0,
            runtime_seconds=result.latency_seconds,
            network_bytes=result.network_bytes,
            tool_roundtrips=result.round_trips,
        )
    else:
        if policy in {"PRA_ADAPTIVE", "PRA_ADAPTIVE_ORACLE"}:
            chunks = _selected_chunks(address, "native_selected_chunks")
            native = runtime.materialize_backing_chunks(
                _record_id(runtime), case.query, chunks,
                selection_policy=address["native_selection_policy"],
            )
            row.update(
                ingestion_seconds=float(address["indexing_seconds"]),
                routing_seconds=float(address["routing_seconds"]),
                materialized_bytes=native.payload_bytes,
                materialized_tokens=(native.payload_bytes + 3) // 4,
                active_kv_tokens=int(address["native_materialized_kv_tokens"]),
                native_kv_tokens=int(address["native_materialized_kv_tokens"]),
                runtime_seconds=native.latency_seconds,
                network_bytes=native.network_bytes,
                tool_roundtrips=native.round_trips,
            )
            extra_payload = native.payload
            native_visible = _contains_answer(extra_payload, case)
            expected_now = (
                ContextAction.CONTINUE.value if native_visible else case.expected_action.value
            )
            row["need_more_target"] = int(expected_now != ContextAction.CONTINUE.value)
        if policy == "PRA_ADAPTIVE_ORACLE":
            action = expected_now
            prompt_tokens = generated_tokens = passes = 0
            controller_seconds = 0.0
        else:
            state = "native" if policy == "PRA_ADAPTIVE" else "compact"
            decision_key = (case.case_id, seed, state)
            if decision_key not in controller_decisions:
                controller_decisions[decision_key] = _controller_decision(
                    client, config, case, runtime, seed
                )
            action, prompt_tokens, generated_tokens, controller_seconds, passes = (
                controller_decisions[decision_key]
            )
        row.update(
            predicted_action=action,
            need_more_predicted=int(action != ContextAction.CONTINUE.value),
            operation_correct=int(action == expected_now),
            controller_prompt_tokens=prompt_tokens,
            controller_generated_tokens=generated_tokens,
            controller_seconds=controller_seconds,
            model_passes=int(row["model_passes"]) + passes,
        )
        if policy == "CCR_TOOL":
            if action == ContextAction.CONTINUE.value:
                payload, size, network, runtime_passes, elapsed, ok = (None, 0, 0, 1, 0.0, True)
            elif action == "INVALID":
                payload, size, network, runtime_passes, elapsed, ok = (None, 0, 0, 1, 0.0, False)
            elif action == ContextAction.CALL_TOOL.value and case.tool_payload is not None:
                payload, size, network, runtime_passes, elapsed, ok = _execute_action(
                    runtime, case, action
                )
            else:
                payload, size, network, runtime_passes, elapsed, ok = _execute_action(
                    runtime, case, ContextAction.MATERIALIZE_FULL.value
                )
            extra_payload = payload
            row.update(materialized_bytes=size, materialized_tokens=(size + 3) // 4,
                       active_kv_tokens=(size + 3) // 4, network_bytes=network,
                       tool_roundtrips=int(action != ContextAction.CONTINUE.value),
                       runtime_seconds=elapsed,
                       model_passes=int(row["model_passes"]) + runtime_passes)
        elif policy in {"MODEL_ONLY", "PRA_ADAPTIVE", "PRA_ADAPTIVE_ORACLE"}:
            if action != ContextAction.CONTINUE.value:
                payload, size, network, runtime_passes, elapsed, ok = _execute_action(
                    runtime, case, action
                )
                extra_payload = payload if payload is not None else extra_payload
                row["materialized_bytes"] = int(row["materialized_bytes"]) + size
                row["materialized_tokens"] = int(row["materialized_tokens"]) + (size + 3) // 4
                row["active_kv_tokens"] = int(row["active_kv_tokens"]) + (size + 3) // 4
                row["network_bytes"] = int(row["network_bytes"]) + network
                row["runtime_seconds"] = float(row["runtime_seconds"]) + elapsed
                row["model_passes"] = int(row["model_passes"]) + runtime_passes

    visible = _visible_text(runtime)
    if extra_payload is not None:
        visible += "\n" + _payload_text(extra_payload)
    evidence_visible = case.expected_answer.casefold() in visible.casefold()
    row["runtime_recovery"] = int(evidence_visible)
    row["evidence_visible"] = int(evidence_visible)
    row["final_use_given_visible"] = int(evidence_visible)
    row["task_success"] = int(evidence_visible)
    predicted_more = str(row["predicted_action"]) not in {
        "", ContextAction.CONTINUE.value, "AUTO_ROUTE"
    }
    target_more = bool(row["need_more_target"])
    row["under_expansion"] = int(target_more and not predicted_more and not evidence_visible)
    row["over_expansion"] = int(not target_more and predicted_more)
    runtime.runtime.store.close()
    return row


def _aggregate(rows: Sequence[Mapping[str, object]]) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy"])].append(row)
    frontier = []
    failures = []
    for policy in POLICIES:
        values = grouped[policy]
        required = [row for row in values if int(row["need_more_target"])]
        sufficient = [row for row in values if not int(row["need_more_target"])]
        predicted = [row for row in values if int(row["need_more_predicted"])]
        visible = [row for row in values if int(row["evidence_visible"])]
        frontier.append({
            "policy": policy,
            "n": len(values),
            "task_success": statistics.fmean(float(row["task_success"]) for row in values),
            "evidence_visible": statistics.fmean(float(row["evidence_visible"]) for row in values),
            "materialized_tokens": statistics.fmean(float(row["materialized_tokens"]) for row in values),
            "active_kv_tokens": statistics.fmean(float(row["active_kv_tokens"]) for row in values),
            "native_kv_tokens": statistics.fmean(float(row["native_kv_tokens"]) for row in values),
            "total_latency_seconds": statistics.fmean(
                float(row["routing_seconds"]) + float(row["controller_seconds"]) + float(row["runtime_seconds"])
                for row in values
            ),
            "model_passes": statistics.fmean(float(row["model_passes"]) for row in values),
            "tool_roundtrips": statistics.fmean(float(row["tool_roundtrips"]) for row in values),
            "network_bytes": statistics.fmean(float(row["network_bytes"]) for row in values),
            "controller_prompt_tokens": statistics.fmean(
                float(row["controller_prompt_tokens"]) for row in values
            ),
        })
        failures.append({
            "policy": policy,
            "n": len(values),
            "need_more_recall": (
                statistics.fmean(float(row["need_more_predicted"]) for row in required)
                if required else 0.0
            ),
            "false_escalation": (
                statistics.fmean(float(row["need_more_predicted"]) for row in sufficient)
                if sufficient else 0.0
            ),
            "operation_accuracy_given_escalation": (
                statistics.fmean(float(row["operation_correct"]) for row in predicted)
                if predicted else 0.0
            ),
            "runtime_recovery": statistics.fmean(float(row["runtime_recovery"]) for row in values),
            "final_use_given_visible": (
                statistics.fmean(float(row["final_use_given_visible"]) for row in visible)
                if visible else 0.0
            ),
            "under_expansion": statistics.fmean(float(row["under_expansion"]) for row in values),
            "over_expansion": statistics.fmean(float(row["over_expansion"]) for row in values),
        })
    return frontier, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--seed-limit", type=int)
    parser.add_argument("--policy-limit", type=int)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    output = args.output_dir
    address = _read_csv(output / "compact_vs_backing_addressability.csv")
    config = _controller_config(output / "selected_controller.json")
    client = OllamaController(args.endpoint, output / "controller_response_cache.json")
    cases = [case for case in full_pra_context_cases() if case.partition == "test"]
    if args.case_limit is not None:
        cases = cases[:args.case_limit]
    seeds = SEEDS[:args.seed_limit]
    policies = POLICIES[:args.policy_limit]
    checkpoint = output / "adaptive_oracle_checkpoint.jsonl"
    rows = _load_checkpoint(checkpoint)
    completed = {
        (str(row["case_id"]), int(row["seed"]), str(row["policy"])) for row in rows
    }
    total = len(cases) * len(seeds) * len(policies)
    count = 0
    controller_decisions: dict[
        tuple[str, int, str], tuple[str, int, int, float, int]
    ] = {}
    for case in cases:
        for seed in seeds:
            for policy in policies:
                count += 1
                if (case.case_id, seed, policy) in completed:
                    continue
                row = _evaluate(
                    policy, case, seed, config, client, address[case.case_id], output,
                    controller_decisions,
                )
                rows.append(row)
                _append_checkpoint(checkpoint, row)
                print(f"[{count}/{total}] {case.case_id} seed={seed} {policy} success={row['task_success']}", flush=True)
    fingerprints = {
        str(row["controller_fingerprint"])
        for row in rows if row["policy"] in {"MODEL_ONLY", "PRA_ADAPTIVE"}
    }
    if fingerprints != {config.fingerprint}:
        raise AssertionError("MODEL_ONLY and PRA_ADAPTIVE did not share one frozen controller config.")
    _write_csv(output / "adaptive_oracle_results.csv", rows)
    frontier, failures = _aggregate(rows)
    _write_csv(output / "quality_cost_frontier.csv", frontier)
    _write_csv(output / "adaptive_failure_decomposition.csv", failures)


if __name__ == "__main__":
    main()
