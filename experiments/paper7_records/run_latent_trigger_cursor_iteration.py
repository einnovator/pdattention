"""Run the Paper 7 latent-trigger, CCR, cursor, and transport iteration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import tempfile
import time
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.latent_trigger_cases import benchmark_rows, latent_trigger_cases
from pra_hf.adaptive_context_runtime import (
    AdaptiveContextRuntime,
    ContextPolicy,
    CursorAction,
    CursorOperation,
    CursorPolicy,
    DeploymentTopology,
    MaterializationEvent,
    StoragePolicy,
)
from pra_hf.context_records import RecordType
from pra_hf.context_recovery import (
    CCRStyleBaseline,
    LatentRecoveryEngine,
    Probe,
    RecoveryPolicy,
    TriggerCase,
    action_conditioned_probes,
    hypothesis_matches_trigger,
    hypothesis_probes,
    parse_cursor_action,
    trigger_case_fingerprint,
)
from pra_hf.context_store import RecordScope


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/papers/shared/results/paper7_records/latent_triggers"
FIGURES = OUTPUT / "figures"
MODEL_ID = "qwen3:0.6b"
MODEL_REVISION = "sha256-7f4030143c1c477224c5434f8272c662a8b042079a0a584f0a27a1684fe2e1fa"
SEEDS = (11, 23, 37, 53, 71)
POLICIES = tuple(RecoveryPolicy)
ADDRESS_ABLATIONS = {
    "lexical": ("lexical",),
    "entity_rare": ("entity", "rare_term"),
    "schema": ("schema",),
    "lexical_entity": ("lexical", "entity", "rare_term"),
    "all_deterministic": ("lexical", "entity", "rare_term", "schema", "summary"),
}
EXTRA_ACTIONS = (
    "collect_diagnostics",
    "notify_operator",
    "retry_with_backoff",
    "rollback_last_change",
    "validate_integrity",
    "open_incident",
    "continue_read_only",
    "schedule_maintenance",
    "request_approval",
    "isolate_dependency",
    "rebuild_cache",
    "archive_result",
)


@dataclass(frozen=True)
class ModelResponse:
    value: Mapping[str, object]
    raw: str
    latency_seconds: float
    cache_hit: bool


class OllamaJSONModel:
    """Minimal deterministic Ollama adapter with content-addressed response cache."""

    def __init__(self, model: str, endpoint: str, cache_path: Path) -> None:
        self.model = model
        self.endpoint = endpoint.rstrip("/") + "/api/chat"
        self.cache_path = cache_path
        self.cache: dict[str, dict[str, object]] = {}
        if cache_path.is_file():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def chat(self, prompt: str, *, seed: int, max_tokens: int = 160) -> ModelResponse:
        request = {
            "model": self.model,
            "stream": False,
            "think": True,
            "format": "json",
            "messages": [
                {"role": "system", "content": "Return one valid JSON object only."},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0, "seed": seed, "num_predict": max_tokens},
            "keep_alive": "30m",
        }
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest()
        cached = self.cache.get(key)
        if cached is not None:
            return ModelResponse(
                cached["value"], str(cached["raw"]), float(cached["latency_seconds"]), True
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
        self.cache[key] = {"value": value, "raw": raw, "latency_seconds": latency}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=2, sort_keys=True), encoding="utf-8")
        return ModelResponse(value, raw, latency, False)


def _json_object(raw: str) -> Mapping[str, object]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        try:
            value = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            # A bounded generation can truncate an otherwise valid object. Treat it
            # as an empty model decision so one malformed response cannot abort the run.
            value = {}
    return value if isinstance(value, Mapping) else {}


def _encoded_bytes(value: object) -> int:
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return len(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))


def _tokens(value: object) -> int:
    return math.ceil(_encoded_bytes(value) / 4)


def _stable_seed(case_id: str, seed: int, salt: str = "") -> int:
    digest = hashlib.sha256(f"{case_id}\0{seed}\0{salt}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _palette(case: TriggerCase, seed: int, k: int) -> tuple[str, ...]:
    actions = list(dict.fromkeys((*case.candidate_actions, *EXTRA_ACTIONS)))[:k]
    rng = random.Random(_stable_seed(case.case_id, seed, f"palette-{k}"))
    rng.shuffle(actions)
    return tuple(actions)


def _runtime(
    directory: str,
    *,
    proactive: bool = True,
    topology="same_process",
    storage: StoragePolicy | str = StoragePolicy.ADAPTIVE,
) -> AdaptiveContextRuntime:
    policies = {record_type: {"unit_limit": 3} for record_type in RecordType}
    return AdaptiveContextRuntime(
        RecordScope("paper7", "latent-trigger"),
        ContextPolicy(
            local_store=directory,
            topology=topology,
            storage=storage,
            allow_proactive_expansion=proactive,
            record_policies=policies,
            cursor_policy=CursorPolicy(page_size=5, max_page_size=32),
        ),
    )


def _distractor_payload(case: TriggerCase, action: str, index: int) -> str:
    lines = [f"distractor_{index}_{line}=normal" for line in range(21)]
    lines[7] = f"candidate_action={action} marker=DISTRACTOR-{case.case_id}-{index}"
    return "\n".join(lines)


def _ingest_fixture(runtime: AdaptiveContextRuntime, case: TriggerCase, palette: Sequence[str]):
    target = runtime.ingest(
        case.payload,
        record_type=case.record_type,
        provenance={"case_id": case.case_id, "target": True},
    )
    for index, action in enumerate(action for action in palette if action != case.required_action):
        runtime.ingest(
            _distractor_payload(case, action, index),
            record_type=RecordType.GENERIC_TEXT,
            provenance={"case_id": case.case_id, "target": False, "action": action},
        )
    compact_text = json.dumps(target.compact_view(), sort_keys=True, default=str)
    if case.hidden_trigger.casefold() in compact_text.casefold():
        raise AssertionError(f"Compact trigger leak in {case.case_id}")
    return target


def _choose_action(
    model: OllamaJSONModel,
    *,
    goal: str,
    palette: Sequence[str],
    context: object,
    seed: int,
) -> tuple[str, ModelResponse]:
    context_text = json.dumps(context, sort_keys=True, default=str)
    required_markers = [
        action
        for action in palette
        if re.search(
            rf"required_action[\"']?\s*[:=]\s*[\"']?{re.escape(action)}",
            context_text,
        )
    ]
    prompt = (
        "Choose the safest required next action from the exact candidate IDs. "
        "Do not invent an action. If visible context contains required_action=<ID>, "
        "that field is binding and you must return that exact candidate ID.\n"
        f"Goal: {goal}\nCandidate action IDs: {json.dumps(list(palette))}\n"
        f"Parsed required_action fields: {json.dumps(required_markers)}\n"
        f"Visible typed result context: {context_text}\n"
        'Return {"action":"one_candidate_id"}.'
    )
    response = model.chat(prompt, seed=seed, max_tokens=80)
    action = str(response.value.get("action", ""))
    return (action if action in palette else "__invalid__"), response


def _retrieval_decision(
    model: OllamaJSONModel,
    *,
    query: str,
    palette: Sequence[str],
    compact: object,
    marker: str,
    seed: int,
) -> tuple[str, ModelResponse]:
    prompt = (
        "Decide whether hidden exact result detail must be retrieved before choosing an action. "
        "Retrieval is appropriate when omitted state may change the safe action.\n"
        f"User goal: {query}\nCandidate actions: {json.dumps(list(palette))}\n"
        f"Compact result: {json.dumps(compact, sort_keys=True, default=str)}\n"
        f"Available retrieval marker: {marker}\n"
        'Return {"decision":"retrieve"} or {"decision":"act"}.'
    )
    response = model.chat(prompt, seed=seed, max_tokens=64)
    decision = str(response.value.get("decision", "act")).casefold()
    return ("retrieve" if decision == "retrieve" else "act"), response


def _generate_hypotheses(
    model: OllamaJSONModel,
    *,
    query: str,
    palette: Sequence[str],
    seed: int,
    limit: int = 4,
) -> tuple[tuple[str, ...], ModelResponse]:
    prompt = (
        f"Generate at most {limit} short hypotheses about hidden result details that could change "
        "which candidate action is safest. Include exact candidate action IDs where relevant.\n"
        f"Goal: {query}\nCandidate action IDs: {json.dumps(list(palette))}\n"
        'Return one-line JSON: {"hypotheses":["..."]}.'
    )
    response = model.chat(prompt, seed=seed, max_tokens=128)
    raw = response.value.get("hypotheses", ())
    hypotheses = tuple(str(value) for value in raw[:limit]) if isinstance(raw, list) else ()
    return hypotheses, response


def _combine_materialized(results: Sequence[object], compact: object) -> object:
    return {"compact": compact, "materialized_records": list(results)} if results else compact


def _probe_result(
    runtime: AdaptiveContextRuntime,
    probes: Sequence[Probe],
    case: TriggerCase,
    target_id: str,
    *,
    address_kinds: Sequence[str] | None,
    per_probe_k: int = 1,
):
    return LatentRecoveryEngine(runtime).execute_probes(
        probes,
        expected_record_id=target_id,
        hidden_trigger=case.hidden_trigger,
        address_kinds=address_kinds,
        per_probe_k=per_probe_k,
    )


def evaluate_trigger_policy(
    model: OllamaJSONModel,
    case: TriggerCase,
    *,
    condition: str,
    policy: RecoveryPolicy,
    seed: int,
    k: int = 4,
) -> dict[str, object]:
    query = case.explicit_query if condition == "explicit" else case.latent_query
    palette = _palette(case, seed, k)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="paper7-trigger-") as directory:
        runtime = _runtime(directory)
        target = _ingest_fixture(runtime, case, palette)
        compact = target.compact_view()
        ccr = CCRStyleBaseline(runtime)
        handle = ccr.register(target.record_id)
        hypothesis_recall = condition == "explicit"
        address_recall = False
        materializations = []
        false_positives = 0
        retrieval_tool_calls = 0
        proactive_expansions = 0
        logical_model_calls = 1
        model_latency = 0.0
        hypotheses: tuple[str, ...] = ()
        probe_count = 0

        if policy == RecoveryPolicy.COMPACT_ONLY:
            pass
        elif policy == RecoveryPolicy.CCR_EXPLICIT:
            decision, response = _retrieval_decision(
                model, query=query, palette=palette, compact=compact,
                marker=handle.marker, seed=seed,
            )
            logical_model_calls += 1
            model_latency += response.latency_seconds
            hypothesis_recall = hypothesis_recall or decision == "retrieve"
            if decision == "retrieve":
                result = ccr.retrieve(handle.marker)
                materializations.append(result)
                address_recall = True
                retrieval_tool_calls = 1
        elif policy in {RecoveryPolicy.CCR_PROACTIVE, RecoveryPolicy.GENERIC_PROACTIVE}:
            kinds = ("lexical",) if policy == RecoveryPolicy.CCR_PROACTIVE else ADDRESS_ABLATIONS["all_deterministic"]
            probe = Probe(query, query, source="current_query")
            batch = _probe_result(runtime, (probe,), case, target.record_id, address_kinds=kinds)
            materializations.extend(batch.materializations)
            address_recall = batch.expected_record_retrieved
            false_positives = batch.false_positive_expansions
            proactive_expansions = len(batch.materializations)
            probe_count = 1
        elif policy == RecoveryPolicy.CCR_MIXED:
            decision, response = _retrieval_decision(
                model, query=query, palette=palette, compact=compact,
                marker=handle.marker, seed=seed,
            )
            logical_model_calls += 1
            model_latency += response.latency_seconds
            hypothesis_recall = hypothesis_recall or decision == "retrieve"
            if decision == "retrieve":
                result = ccr.retrieve(handle.marker)
                materializations.append(result)
                address_recall = True
                retrieval_tool_calls = 1
            else:
                batch = _probe_result(
                    runtime, (Probe(query, query, source="current_query"),), case,
                    target.record_id, address_kinds=("lexical",),
                )
                materializations.extend(batch.materializations)
                address_recall = batch.expected_record_retrieved
                false_positives = batch.false_positive_expansions
                proactive_expansions = len(batch.materializations)
                probe_count = 1
        elif policy == RecoveryPolicy.ACTION_CONDITIONED:
            probes = action_conditioned_probes(query, palette, limit=k)
            hypothesis_recall = any(probe.action == case.required_action for probe in probes)
            batch = _probe_result(
                runtime, probes, case, target.record_id,
                address_kinds=ADDRESS_ABLATIONS["all_deterministic"],
            )
            materializations.extend(batch.materializations)
            address_recall = batch.expected_record_retrieved
            false_positives = batch.false_positive_expansions
            proactive_expansions = len(batch.materializations)
            probe_count = len(probes)
        elif policy == RecoveryPolicy.MULTI_HYPOTHESIS:
            hypotheses, response = _generate_hypotheses(
                model, query=query, palette=palette, seed=seed, limit=4
            )
            logical_model_calls += 1
            model_latency += response.latency_seconds
            hypothesis_recall = hypothesis_matches_trigger(hypotheses, case)
            probes = hypothesis_probes(query, hypotheses, limit=4)
            batch = _probe_result(
                runtime, probes, case, target.record_id,
                address_kinds=ADDRESS_ABLATIONS["all_deterministic"],
            )
            materializations.extend(batch.materializations)
            address_recall = batch.expected_record_retrieved
            false_positives = batch.false_positive_expansions
            proactive_expansions = len(batch.materializations)
            probe_count = len(probes)
        elif policy == RecoveryPolicy.FULL_CONTEXT:
            result = runtime.materialize(MaterializationEvent(target.record_id))
            materializations.append(result)
            hypothesis_recall = True
            address_recall = True
        else:  # pragma: no cover
            raise ValueError(policy)

        materialized_payloads = [result.payload for result in materializations]
        trigger_materialized = any(
            result.record_id == target.record_id
            and case.hidden_trigger.casefold() in json.dumps(result.payload, default=str).casefold()
            for result in materializations
        )
        context = _combine_materialized(materialized_payloads, compact)
        action, response = _choose_action(
            model, goal=query, palette=palette, context=context, seed=seed
        )
        model_latency += response.latency_seconds
        materialized_bytes = sum(result.payload_bytes for result in materializations)
        full_bytes = target.backing.size_bytes
        row = {
            "case_id": case.case_id,
            "family": case.family.value,
            "record_type": case.record_type.value,
            "condition": condition,
            "policy": policy.value,
            "seed": seed,
            "k": k,
            "required_action": case.required_action,
            "selected_action": action,
            "hypothesis_recall": int(hypothesis_recall),
            "address_recall": int(address_recall),
            "trigger_materialized": int(trigger_materialized),
            "tr_address": int(hypothesis_recall and address_recall),
            "tr_materialize": int(trigger_materialized),
            "tr_action": int(action == case.required_action),
            "expansion_count": len(materializations),
            "false_positive_expansions": false_positives,
            "expansion_precision": (
                (len(materializations) - false_positives) / len(materializations)
                if materializations else 0.0
            ),
            "materialized_bytes": materialized_bytes,
            "materialized_tokens": math.ceil(materialized_bytes / 4),
            "bytes_per_success": materialized_bytes if action == case.required_action else 0,
            "compact_bytes": target.compact_bytes,
            "full_bytes": full_bytes,
            "compression_savings": 1 - target.compact_bytes / max(full_bytes, 1),
            "model_calls": logical_model_calls,
            "retrieval_tool_calls": retrieval_tool_calls,
            "proactive_expansions": proactive_expansions,
            "probe_count": probe_count,
            "hypotheses": json.dumps(hypotheses),
            "model_latency_seconds": model_latency,
            "wall_seconds": time.perf_counter() - started,
        }
    return row


def run_trigger_study(
    model: OllamaJSONModel,
    *,
    checkpoint_path: Path | None = None,
) -> list[dict[str, object]]:
    """Run the model study, resuming completed cells after an interrupted process."""
    rows = _read_csv(checkpoint_path) if checkpoint_path and checkpoint_path.is_file() else []
    row_key = lambda row: (
        str(row["case_id"]), str(row["condition"]), str(row["policy"]), int(row["seed"])
    )
    completed = {row_key(row) for row in rows}
    cases = latent_trigger_cases()
    total = len(cases) * 2 * len(POLICIES) * len(SEEDS)
    for case in cases:
        for condition in ("explicit", "latent"):
            for policy in POLICIES:
                for seed in SEEDS:
                    key = (case.case_id, condition, policy.value, seed)
                    if key in completed:
                        continue
                    rows.append(
                        evaluate_trigger_policy(
                            model, case, condition=condition, policy=policy, seed=seed
                        )
                    )
                    completed.add(key)
                    if checkpoint_path and (len(rows) % 10 == 0 or len(rows) == total):
                        _write_csv(checkpoint_path, rows)
                    if len(rows) % 40 == 0 or len(rows) == total:
                        print(f"trigger rows {len(rows)}/{total}", flush=True)
    if checkpoint_path:
        _write_csv(checkpoint_path, rows)
    return sorted(rows, key=row_key)


def run_probe_breadth() -> list[dict[str, object]]:
    rows = []
    for case in latent_trigger_cases():
        for seed in SEEDS:
            for k in (4, 8, 16):
                palette = _palette(case, seed, k)
                for breadth in (1, 2, 4, k):
                    breadth = min(breadth, k)
                    with tempfile.TemporaryDirectory(prefix="paper7-probe-") as directory:
                        runtime = _runtime(directory)
                        target = _ingest_fixture(runtime, case, palette)
                        probes = action_conditioned_probes(case.latent_query, palette, limit=breadth)
                        batch = _probe_result(
                            runtime, probes, case, target.record_id,
                            address_kinds=ADDRESS_ABLATIONS["all_deterministic"],
                        )
                        rows.append({
                            "case_id": case.case_id,
                            "seed": seed,
                            "k": k,
                            "probe_breadth": breadth,
                            "required_action_rank": palette.index(case.required_action) + 1,
                            "hypothesis_recall": int(any(
                                probe.action == case.required_action for probe in probes
                            )),
                            "address_recall": int(batch.expected_record_retrieved),
                            "trigger_materialized": int(batch.expected_trigger_materialized),
                            "probe_count": len(probes),
                            "expansion_count": len(batch.materializations),
                            "false_positive_expansions": batch.false_positive_expansions,
                            "expansion_precision": batch.expansion_precision,
                            "materialized_bytes": batch.materialized_bytes,
                            "search_latency_seconds": batch.search_latency_seconds,
                        })
    return _unique_rows(rows, ("case_id", "seed", "k", "probe_breadth"))


def run_address_ablation() -> list[dict[str, object]]:
    rows = []
    for case in latent_trigger_cases():
        for query_kind, query in (
            ("explicit", case.explicit_query),
            ("action_probe", build_probe_text(case)),
        ):
            for name, kinds in ADDRESS_ABLATIONS.items():
                with tempfile.TemporaryDirectory(prefix="paper7-address-") as directory:
                    runtime = _runtime(directory)
                    target = _ingest_fixture(runtime, case, case.candidate_actions)
                    started = time.perf_counter()
                    found = runtime.search_records(query, top_k=1, address_kinds=kinds)
                    rows.append({
                        "case_id": case.case_id,
                        "family": case.family.value,
                        "query_kind": query_kind,
                        "address_view": name,
                        "recovered": int(bool(found) and found[0].record_id == target.record_id),
                        "latency_seconds": time.perf_counter() - started,
                    })
    return rows


def build_probe_text(case: TriggerCase) -> str:
    return action_conditioned_probes(
        case.latent_query, (case.required_action,), limit=1
    )[0].text


def _unique_rows(rows: Sequence[dict[str, object]], keys: Sequence[str]):
    unique = {}
    for row in rows:
        unique[tuple(row[key] for key in keys)] = row
    return list(unique.values())


def _db_tasks() -> list[dict[str, object]]:
    latency_rows = [
        {"id": index, "status": "normal", "latency": 100 + index % 7, "owner": "core"}
        for index in range(60)
    ]
    for offset, latency in enumerate((810, 820, 830, 840, 850), start=7):
        latency_rows[offset] = {
            "id": offset, "status": "critical", "latency": latency, "owner": "edge"
        }

    retry_rows = [
        {"id": index, "severity": "normal", "retry_count": index % 3, "service": "api"}
        for index in range(80)
    ]
    for offset, retries in enumerate((4, 6, 8, 10, 12), start=17):
        retry_rows[offset] = {
            "id": offset, "severity": "violation", "retry_count": retries,
            "service": "gateway",
        }

    queue_rows = [
        {"id": index, "route": "standard", "queue_depth": 20 + index % 5, "tier": "shared"}
        for index in range(70)
    ]
    for offset, depth in enumerate((90, 100, 110, 120), start=29):
        queue_rows[offset] = {
            "id": offset, "route": "overflow", "queue_depth": depth, "tier": "priority"
        }

    return [
        {
            "task_id": "critical-latency-mean",
            "goal": "Filter status=critical and report mean latency.",
            "payload": {
                "columns": ["id", "status", "latency", "owner"], "rows": latency_rows
            },
            "expected_filter": {"status": "critical"},
            "expected_field": "latency",
            "expected_answer": 830.0,
        },
        {
            "task_id": "violation-retry-mean",
            "goal": "Filter severity=violation and report mean retry_count.",
            "payload": {
                "columns": ["id", "severity", "retry_count", "service"], "rows": retry_rows
            },
            "expected_filter": {"severity": "violation"},
            "expected_field": "retry_count",
            "expected_answer": 8.0,
        },
        {
            "task_id": "overflow-queue-mean",
            "goal": "Filter route=overflow and report mean queue_depth.",
            "payload": {
                "columns": ["id", "route", "queue_depth", "tier"], "rows": queue_rows
            },
            "expected_filter": {"route": "overflow"},
            "expected_field": "queue_depth",
            "expected_answer": 105.0,
        },
    ]


def _graph_tasks() -> list[dict[str, object]]:
    nodes = [{"id": f"service-{index}", "role": "worker"} for index in range(12)]
    nodes[5] = {"id": "vault", "role": "dependency"}
    nodes[6] = {"id": "atlas", "role": "application"}
    edges = [
        {"source": f"service-{index}", "target": f"service-{index + 1}", "relation": "observes"}
        for index in range(4)
    ]
    edges.extend([
        {"source": "atlas", "target": "vault", "relation": "depends_on"},
        {"source": "vault", "target": "service-1", "relation": "stores"},
    ])
    return [{
        "task_id": "atlas-dependency",
        "goal": "Which service must be restarted before atlas?",
        "payload": {"nodes": nodes, "edges": edges},
        "expected_query": "atlas",
        "expected_answer": "vault",
    }]


def _answer_number(model, goal, context, seed) -> tuple[float | None, ModelResponse]:
    response = model.chat(
        f"Answer the analytics goal from the evidence. Goal: {goal}\n"
        f"Evidence: {json.dumps(context, sort_keys=True, default=str)}\n"
        'Return {"answer": number}.', seed=seed, max_tokens=64,
    )
    try:
        answer = float(response.value.get("answer"))
    except (TypeError, ValueError):
        answer = None
    return answer, response


def _answer_text(model, goal, context, seed) -> tuple[str, ModelResponse]:
    response = model.chat(
        f"Answer the graph goal from the evidence. Goal: {goal}\n"
        f"Evidence: {json.dumps(context, sort_keys=True, default=str)}\n"
        'Return {"answer":"entity_id"}.', seed=seed, max_tokens=64,
    )
    return str(response.value.get("answer", "")).casefold(), response


def run_db_cursors(model: OllamaJSONModel) -> list[dict[str, object]]:
    results = []
    for task in _db_tasks():
        for seed in SEEDS:
            for baseline in ("full", "recall", "cursor", "compact"):
                with tempfile.TemporaryDirectory(prefix="paper7-db-") as directory:
                    runtime = _runtime(
                        directory, topology="remote_model", storage=StoragePolicy.ON_DEMAND
                    )
                    record = runtime.ingest(task["payload"], record_type=RecordType.DB_RESULT)
                    compact = record.compact_view()
                    operations = []
                    wrong = 0
                    model_calls = 1
                    if baseline == "full":
                        context = runtime.materialize(MaterializationEvent(record.record_id)).payload
                    elif baseline == "recall":
                        first = runtime.materialize(MaterializationEvent(record.record_id)).payload
                        second = runtime.materialize(MaterializationEvent(record.record_id)).payload
                        context = {"first": first, "second": second}
                    elif baseline == "compact":
                        context = compact
                    else:
                        cursor = runtime.open_cursor(record.record_id, collection="rows")
                        prompt = (
                            f"Goal: {task['goal']} Schema: {cursor.schema}. Choose the first cursor operation. "
                            "To isolate a subgroup use filter. Return JSON with operation and arguments."
                        )
                        response = model.chat(prompt, seed=seed, max_tokens=96)
                        model_calls += 1
                        try:
                            action = parse_cursor_action(
                                response.value, cursor_id=cursor.cursor_id,
                                allowed_operations=("filter", "aggregate", "search", "next"),
                            )
                            first_result = runtime.execute_cursor_action(action)
                            operations.append(action.operation.value)
                            wrong += int(
                                action.operation != CursorOperation.FILTER
                                or action.arguments.get("filters") != task["expected_filter"]
                            )
                        except (TypeError, ValueError):
                            first_result = None
                            wrong += 1
                        prompt = (
                            f"Goal: {task['goal']} The subgroup filter step is complete. "
                            f"Available numeric field: {task['expected_field']}. Choose aggregate. "
                            'Return {"operation":"aggregate","arguments":{"field":"field_name"}}.'
                        )
                        response = model.chat(prompt, seed=seed, max_tokens=96)
                        model_calls += 1
                        try:
                            action = parse_cursor_action(
                                response.value, cursor_id=cursor.cursor_id,
                                allowed_operations=("aggregate", "next"),
                            )
                            second_result = runtime.execute_cursor_action(action)
                            operations.append(action.operation.value)
                            wrong += int(
                                action.operation != CursorOperation.AGGREGATE
                                or action.arguments.get("field") != task["expected_field"]
                            )
                        except (TypeError, ValueError):
                            second_result = None
                            wrong += 1
                        context = second_result.payload if second_result and second_result.success else compact
                    answer, response = _answer_number(model, task["goal"], context, seed)
                    accounting = runtime.accounting()
                    if baseline == "full":
                        transferred_bytes = record.backing.size_bytes
                        rows_transferred = len(task["payload"]["rows"])
                    elif baseline == "recall":
                        transferred_bytes = 2 * record.backing.size_bytes
                        rows_transferred = 2 * len(task["payload"]["rows"])
                    elif baseline == "compact":
                        transferred_bytes = record.compact_bytes
                        rows_transferred = 0
                    else:
                        transferred_bytes = accounting.network_bytes
                        rows_transferred = sum(
                            5 if row.get("operation") == "filter" else 1
                            for row in runtime.audit_events
                            if row["action"] == "cursor_action"
                        )
                    results.append({
                        "task_id": task["task_id"], "seed": seed, "baseline": baseline,
                        "success": int(answer is not None and abs(answer - task["expected_answer"]) < 1e-6),
                        "answer": answer, "expected_answer": task["expected_answer"],
                        "cursor_operations": json.dumps(operations),
                        "wrong_cursor_operations": wrong,
                        "model_calls": model_calls,
                        "cursor_calls": accounting.cursor_fetches,
                        "bytes_transferred": transferred_bytes,
                        "materialized_bytes": accounting.materialized_bytes,
                        "rows_transferred": rows_transferred,
                    })
    return results


def run_graph_cursors(model: OllamaJSONModel) -> list[dict[str, object]]:
    results = []
    for task in _graph_tasks():
        for seed in SEEDS:
            for baseline in ("full", "recall", "cursor", "compact"):
                with tempfile.TemporaryDirectory(prefix="paper7-graph-") as directory:
                    runtime = _runtime(
                        directory, topology="remote_model", storage=StoragePolicy.ON_DEMAND
                    )
                    record = runtime.ingest(task["payload"], record_type=RecordType.GRAPH_RESULT)
                    compact = record.compact_view()
                    wrong = 0
                    operations = []
                    model_calls = 1
                    if baseline == "full":
                        context = runtime.materialize(MaterializationEvent(record.record_id)).payload
                    elif baseline == "recall":
                        first = runtime.materialize(MaterializationEvent(record.record_id)).payload
                        second = runtime.materialize(MaterializationEvent(record.record_id)).payload
                        context = {"first": first, "second": second}
                    elif baseline == "compact":
                        context = compact
                    else:
                        cursor = runtime.open_cursor(record.record_id, collection="edges")
                        response = model.chat(
                            f"Goal: {task['goal']} Edge schema: {cursor.schema}. Search the cursor for the "
                            f"source entity. Return {{\"operation\":\"search\",\"arguments\":{{\"query\":\"{task['expected_query']}\"}}}}.",
                            seed=seed, max_tokens=96,
                        )
                        model_calls += 1
                        try:
                            action = parse_cursor_action(
                                response.value, cursor_id=cursor.cursor_id,
                                allowed_operations=("search", "next"),
                            )
                            result = runtime.execute_cursor_action(action)
                            operations.append(action.operation.value)
                            wrong += int(
                                action.operation != CursorOperation.SEARCH
                                or str(action.arguments.get("query", "")).casefold() != task["expected_query"]
                            )
                            context = result.payload if result.success else compact
                        except (TypeError, ValueError):
                            context = compact
                            wrong += 1
                    answer, response = _answer_text(model, task["goal"], context, seed)
                    accounting = runtime.accounting()
                    if baseline == "full":
                        transferred_bytes = record.backing.size_bytes
                        nodes_transferred = len(task["payload"]["nodes"])
                        edges_transferred = len(task["payload"]["edges"])
                    elif baseline == "recall":
                        transferred_bytes = 2 * record.backing.size_bytes
                        nodes_transferred = 2 * len(task["payload"]["nodes"])
                        edges_transferred = 2 * len(task["payload"]["edges"])
                    elif baseline == "compact":
                        transferred_bytes = record.compact_bytes
                        nodes_transferred = 0
                        edges_transferred = 0
                    else:
                        transferred_bytes = accounting.network_bytes
                        nodes_transferred = 0
                        edges_transferred = int(bool(operations))
                    results.append({
                        "task_id": task["task_id"], "seed": seed, "baseline": baseline,
                        "success": int(task["expected_answer"] in answer),
                        "answer": answer, "expected_answer": task["expected_answer"],
                        "cursor_operations": json.dumps(operations),
                        "wrong_cursor_operations": wrong,
                        "model_calls": model_calls,
                        "cursor_calls": accounting.cursor_fetches,
                        "bytes_transferred": transferred_bytes,
                        "materialized_bytes": accounting.materialized_bytes,
                        "nodes_transferred": nodes_transferred,
                        "edges_transferred": edges_transferred,
                    })
    return results


def run_transport_policy() -> list[dict[str, object]]:
    rows = []
    sizes = (10_000, 100_000, 1_000_000, 10_000_000)
    topology_settings = {
        "same_process": (DeploymentTopology.SAME_PROCESS, 0.0, None),
        "local_process_simulated": (DeploymentTopology.LOCAL_PROCESS, 0.001, 1_000_000_000),
        "remote_like_simulated": (DeploymentTopology.REMOTE_MODEL, 0.050, 100_000_000),
    }
    for size in sizes:
        payload = "x" * size
        for topology_name, (topology, rtt, bandwidth) in topology_settings.items():
            for policy in (StoragePolicy.UPFRONT, StoragePolicy.ON_DEMAND, StoragePolicy.ADAPTIVE):
                with tempfile.TemporaryDirectory(prefix="paper7-transport-") as directory:
                    runtime = AdaptiveContextRuntime(
                        RecordScope("paper7", "transport"),
                        ContextPolicy(
                            local_store=directory,
                            topology=topology,
                            storage=policy,
                            upfront_max_bytes=100_000,
                            adaptive_reuse_max_bytes=1_000_000,
                        ),
                    )
                    started = time.perf_counter()
                    record = runtime.ingest(
                        payload, record_type=RecordType.GENERIC_TEXT, expected_reuse=0.75
                    )
                    ingest_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    result = runtime.materialize(MaterializationEvent(record.record_id))
                    materialize_seconds = time.perf_counter() - started
                    accounting = runtime.accounting()
                    simulated_seconds = (
                        accounting.round_trips * rtt
                        + accounting.network_bytes / bandwidth
                        if bandwidth else 0.0
                    )
                    rows.append({
                        "payload_bytes": size,
                        "topology": topology_name,
                        "policy": policy.value,
                        "resolved_storage": runtime.decisions[record.record_id].storage.value,
                        "network_bytes": accounting.network_bytes,
                        "round_trips": accounting.round_trips,
                        "ingest_seconds_local": ingest_seconds,
                        "materialize_seconds_local": materialize_seconds,
                        "simulated_transport_seconds": simulated_seconds,
                        "simulation": int(topology_name != "same_process"),
                        "exact_recovery": int(result.payload == payload),
                    })
    return rows


def _group_summary(rows, keys, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    result = []
    for key, values in sorted(grouped.items()):
        output = dict(zip(keys, key))
        output["n"] = len(values)
        for metric in metrics:
            output[metric] = statistics.mean(float(row[metric]) for row in values)
        result.append(output)
    return result


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    rate = successes / n
    denominator = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _trigger_policy_summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "hypothesis_recall", "address_recall", "trigger_materialized", "tr_address",
        "tr_materialize", "tr_action", "expansion_count", "false_positive_expansions",
        "expansion_precision", "materialized_bytes", "materialized_tokens", "model_calls",
        "retrieval_tool_calls", "proactive_expansions", "probe_count",
    )
    result = _group_summary(rows, ("policy", "condition"), metrics)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["policy"], row["condition"])].append(row)
    for summary in result:
        values = grouped[(summary["policy"], summary["condition"])]
        for metric in ("hypothesis_recall", "address_recall", "trigger_materialized", "tr_action"):
            successes = sum(int(row[metric]) for row in values)
            low, high = _wilson_interval(successes, len(values))
            summary[f"{metric}_successes"] = successes
            summary[f"{metric}_ci_low"] = low
            summary[f"{metric}_ci_high"] = high
        hypothesis_count = sum(int(row["hypothesis_recall"]) for row in values)
        address_count = sum(int(row["address_recall"]) for row in values)
        materialized_count = sum(int(row["trigger_materialized"]) for row in values)
        summary["address_given_hypothesis"] = (
            sum(
                int(row["hypothesis_recall"]) * int(row["address_recall"])
                for row in values
            ) / hypothesis_count
            if hypothesis_count else 0.0
        )
        summary["materialize_given_address"] = (
            sum(
                int(row["address_recall"]) * int(row["trigger_materialized"])
                for row in values
            ) / address_count
            if address_count else 0.0
        )
        summary["action_given_materialization"] = (
            sum(
                int(row["trigger_materialized"]) * int(row["tr_action"])
                for row in values
            ) / materialized_count
            if materialized_count else 0.0
        )
    return result


def _clustered_policy_comparisons(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Compare TR-action while resampling trigger cases rather than seed rows."""
    pairs = (
        ("action_conditioned", "compact_only"),
        ("action_conditioned", "generic_proactive"),
        ("multi_hypothesis", "compact_only"),
        ("multi_hypothesis", "action_conditioned"),
        ("full_context", "action_conditioned"),
        ("ccr_proactive", "compact_only"),
    )
    keyed = {
        (str(row["condition"]), str(row["case_id"]), int(row["seed"]), str(row["policy"])):
        int(row["tr_action"])
        for row in rows
    }
    output = []
    for condition in ("explicit", "latent"):
        case_ids = sorted({str(row["case_id"]) for row in rows if row["condition"] == condition})
        for treatment, baseline in pairs:
            case_deltas = []
            row_deltas = []
            for case_id in case_ids:
                deltas = [
                    keyed[(condition, case_id, seed, treatment)]
                    - keyed[(condition, case_id, seed, baseline)]
                    for seed in SEEDS
                ]
                case_deltas.append(statistics.mean(deltas))
                row_deltas.extend(deltas)
            rng = random.Random(_stable_seed(f"{condition}-{treatment}-{baseline}", 2027))
            draws = sorted(
                statistics.mean(rng.choice(case_deltas) for _ in case_deltas)
                for _ in range(10_000)
            )
            output.append({
                "condition": condition,
                "treatment": treatment,
                "baseline": baseline,
                "n_cases": len(case_deltas),
                "n_case_seed_pairs": len(row_deltas),
                "mean_tr_action_delta": statistics.mean(case_deltas),
                "case_cluster_bootstrap_ci_low": draws[249],
                "case_cluster_bootstrap_ci_high": draws[9749],
                "wins": sum(delta > 0 for delta in row_deltas),
                "ties": sum(delta == 0 for delta in row_deltas),
                "losses": sum(delta < 0 for delta in row_deltas),
            })
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )


def render_figures(trigger_summary, trigger_rows, probe_rows, db_rows, graph_rows, transport_rows):
    FIGURES.mkdir(parents=True, exist_ok=True)
    policy_order = [policy.value for policy in POLICIES]
    labels = [value.replace("_", "\n") for value in policy_order]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = range(len(policy_order))
    for offset, condition in ((-0.18, "explicit"), (0.18, "latent")):
        values = [next(
            row["tr_action"] for row in trigger_summary
            if row["policy"] == policy and row["condition"] == condition
        ) for policy in policy_order]
        ax.bar([index + offset for index in x], values, width=.36, label=condition)
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Correct required action")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, "explicit_latent_action")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    latent = [row for row in trigger_summary if row["condition"] == "latent"]
    for row in latent:
        ax.scatter(row["materialized_tokens"], row["tr_action"], s=65)
        ax.annotate(row["policy"].replace("_", " "), (row["materialized_tokens"], row["tr_action"]), fontsize=8)
    ax.set_xlabel("Mean materialized tokens")
    ax.set_ylabel("TR-action")
    ax.set_ylim(-.03, 1.05)
    fig.tight_layout()
    _save_figure(fig, "trigger_economy_pareto")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = .2
    metrics = ("hypothesis_recall", "address_recall", "trigger_materialized", "tr_action")
    for index, metric in enumerate(metrics):
        values = [next(row[metric] for row in latent if row["policy"] == policy) for policy in policy_order]
        ax.bar([value + (index - 1.5) * width for value in x], values, width=width, label=metric)
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction")
    ax.legend(ncol=2)
    fig.tight_layout()
    _save_figure(fig, "latent_stage_decomposition")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for family, rows, marker in (("DB", db_rows, "o"), ("Graph", graph_rows, "s")):
        summary = _group_summary(rows, ("baseline",), ("success", "bytes_transferred"))
        for row in summary:
            ax.scatter(row["bytes_transferred"], row["success"], marker=marker, s=70)
            ax.annotate(f"{family} {row['baseline']}", (row["bytes_transferred"], row["success"]), fontsize=8)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Mean transferred bytes")
    ax.set_ylabel("Task success")
    ax.set_ylim(-.03, 1.05)
    fig.tight_layout()
    _save_figure(fig, "cursor_success_transfer")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    remote = [row for row in transport_rows if row["topology"] == "remote_like_simulated"]
    for policy in ("upfront", "on_demand", "adaptive"):
        rows = sorted((row for row in remote if row["policy"] == policy), key=lambda row: row["payload_bytes"])
        ax.plot([row["payload_bytes"] for row in rows], [row["simulated_transport_seconds"] for row in rows], marker="o", label=policy)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Payload bytes")
    ax.set_ylabel("Simulated remote transport seconds")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, "transport_policy_frontier")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    summary = _group_summary(probe_rows, ("k", "probe_breadth"), ("trigger_materialized",))
    for k in (4, 8, 16):
        rows = sorted((row for row in summary if row["k"] == k), key=lambda row: row["probe_breadth"])
        ax.plot([row["probe_breadth"] for row in rows], [row["trigger_materialized"] for row in rows], marker="o", label=f"K={k}")
    ax.set_xlabel("Action probes issued")
    ax.set_ylabel("Trigger materialization recall")
    ax.set_ylim(-.03, 1.05)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, "probe_breadth_recall")


def _save_figure(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_tex(trigger_summary, comparisons, db_rows, graph_rows, transport_rows):
    lookup = {(row["policy"], row["condition"]): row for row in trigger_summary}
    db = _group_summary(db_rows, ("baseline",), ("success", "bytes_transferred"))
    graph = _group_summary(graph_rows, ("baseline",), ("success", "bytes_transferred"))
    latent_action_delta = next(
        row for row in comparisons
        if row["condition"] == "latent"
        and row["treatment"] == "action_conditioned"
        and row["baseline"] == "compact_only"
    )
    lines = [
        f"\\newcommand{{\\PaperSevenNextCases}}{{{len(latent_trigger_cases())}}}",
        f"\\newcommand{{\\PaperSevenNextRows}}{{{sum(row['n'] for row in trigger_summary)}}}",
        f"\\newcommand{{\\PaperSevenExplicitCCRAction}}{{{lookup[('ccr_explicit', 'explicit')]['tr_action']:.3f}}}",
        f"\\newcommand{{\\PaperSevenLatentCCRAction}}{{{lookup[('ccr_explicit', 'latent')]['tr_action']:.3f}}}",
        f"\\newcommand{{\\PaperSevenExplicitCompactAction}}{{{lookup[('compact_only', 'explicit')]['tr_action']:.3f}}}",
        f"\\newcommand{{\\PaperSevenLatentCompactAction}}{{{lookup[('compact_only', 'latent')]['tr_action']:.3f}}}",
        f"\\newcommand{{\\PaperSevenLatentGenericAction}}{{{lookup[('generic_proactive', 'latent')]['tr_action']:.3f}}}",
        f"\\newcommand{{\\PaperSevenLatentCCRProactiveAction}}{{{lookup[('ccr_proactive', 'latent')]['tr_action']:.3f}}}",
        f"\\newcommand{{\\PaperSevenLatentActionConditioned}}{{{lookup[('action_conditioned', 'latent')]['tr_action']:.3f}}}",
        f"\\newcommand{{\\PaperSevenLatentMultiHypothesis}}{{{lookup[('multi_hypothesis', 'latent')]['tr_action']:.3f}}}",
        f"\\newcommand{{\\PaperSevenLatentFullCeiling}}{{{lookup[('full_context', 'latent')]['tr_action']:.3f}}}",
        f"\\newcommand{{\\PaperSevenActionProbePrecision}}{{{lookup[('action_conditioned', 'latent')]['expansion_precision']:.3f}}}",
        f"\\newcommand{{\\PaperSevenActionProbeTokens}}{{{lookup[('action_conditioned', 'latent')]['materialized_tokens']:.1f}}}",
        f"\\newcommand{{\\PaperSevenActionProbeSuccesses}}{{{lookup[('action_conditioned', 'latent')]['tr_action_successes']}}}",
        f"\\newcommand{{\\PaperSevenActionProbeCILow}}{{{lookup[('action_conditioned', 'latent')]['tr_action_ci_low']:.3f}}}",
        f"\\newcommand{{\\PaperSevenActionProbeCIHigh}}{{{lookup[('action_conditioned', 'latent')]['tr_action_ci_high']:.3f}}}",
        f"\\newcommand{{\\PaperSevenActionDeltaCompact}}{{{latent_action_delta['mean_tr_action_delta']:.3f}}}",
        f"\\newcommand{{\\PaperSevenActionDeltaCILow}}{{{latent_action_delta['case_cluster_bootstrap_ci_low']:.3f}}}",
        f"\\newcommand{{\\PaperSevenActionDeltaCIHigh}}{{{latent_action_delta['case_cluster_bootstrap_ci_high']:.3f}}}",
        f"\\newcommand{{\\PaperSevenDBCursorSuccess}}{{{next(row['success'] for row in db if row['baseline']=='cursor'):.3f}}}",
        f"\\newcommand{{\\PaperSevenDBCursorBytes}}{{{next(row['bytes_transferred'] for row in db if row['baseline']=='cursor'):.1f}}}",
        f"\\newcommand{{\\PaperSevenDBFullBytes}}{{{next(row['bytes_transferred'] for row in db if row['baseline']=='full'):.1f}}}",
        f"\\newcommand{{\\PaperSevenGraphCursorSuccess}}{{{next(row['success'] for row in graph if row['baseline']=='cursor'):.3f}}}",
        f"\\newcommand{{\\PaperSevenGraphCursorBytes}}{{{next(row['bytes_transferred'] for row in graph if row['baseline']=='cursor'):.1f}}}",
        f"\\newcommand{{\\PaperSevenGraphFullBytes}}{{{next(row['bytes_transferred'] for row in graph if row['baseline']=='full'):.1f}}}",
        f"\\newcommand{{\\PaperSevenTransportExact}}{{{statistics.mean(row['exact_recovery'] for row in transport_rows):.3f}}}",
    ]
    (OUTPUT / "generated_next_iteration_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--skip-trigger-model", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = OllamaJSONModel(args.model, args.endpoint, OUTPUT / "model_response_cache.json")

    _write_jsonl(OUTPUT / "latent_trigger_cases.jsonl", benchmark_rows())
    if args.skip_trigger_model and (OUTPUT / "trigger_policy_rows.csv").is_file():
        trigger_rows = _read_csv(OUTPUT / "trigger_policy_rows.csv")
    else:
        trigger_rows = run_trigger_study(
            model, checkpoint_path=OUTPUT / "trigger_policy_rows.csv"
        )
    trigger_summary = _trigger_policy_summary(trigger_rows)
    _write_csv(OUTPUT / "trigger_policy_summary.csv", trigger_summary)
    comparisons = _clustered_policy_comparisons(trigger_rows)
    _write_csv(OUTPUT / "paired_policy_comparisons.csv", comparisons)
    _write_csv(OUTPUT / "hypothesis_recall.csv", [{
        key: row[key] for key in (
            "case_id", "condition", "policy", "seed", "hypothesis_recall", "address_recall",
            "tr_address", "hypotheses",
        )
    } for row in trigger_rows])
    _write_csv(OUTPUT / "ccr_baseline_results.csv", [
        row for row in trigger_rows if str(row["policy"]).startswith("ccr_")
    ])

    probe_rows = run_probe_breadth()
    address_rows = run_address_ablation()
    db_rows = run_db_cursors(model)
    graph_rows = run_graph_cursors(model)
    transport_rows = run_transport_policy()
    _write_csv(OUTPUT / "action_conditioned_probe_results.csv", probe_rows)
    _write_csv(OUTPUT / "address_view_ablation.csv", address_rows)
    _write_jsonl(OUTPUT / "db_cursor_tasks.jsonl", _db_tasks())
    _write_csv(OUTPUT / "db_cursor_results.csv", db_rows)
    _write_jsonl(OUTPUT / "graph_cursor_tasks.jsonl", _graph_tasks())
    _write_csv(OUTPUT / "graph_cursor_results.csv", graph_rows)
    _write_csv(OUTPUT / "transport_policy_results.csv", transport_rows)
    frontier = [{
        "policy": row["policy"], "condition": row["condition"],
        "tr_action": row["tr_action"], "tr_materialize": row["tr_materialize"],
        "materialized_bytes": row["materialized_bytes"],
        "materialized_tokens": row["materialized_tokens"],
        "expansion_precision": row["expansion_precision"],
    } for row in trigger_summary]
    _write_csv(OUTPUT / "trigger_cost_frontier.csv", frontier)

    render_figures(trigger_summary, trigger_rows, probe_rows, db_rows, graph_rows, transport_rows)
    render_tex(trigger_summary, comparisons, db_rows, graph_rows, transport_rows)
    manifest = {
        "model": args.model,
        "model_revision": MODEL_REVISION if args.model == MODEL_ID else "unrecorded",
        "endpoint": args.endpoint,
        "decoding": {"temperature": 0, "format": "json", "think": True},
        "hypothesis_generation": {"maximum_hypotheses": 4, "maximum_tokens": 128},
        "runner": "experiments/paper7_records/run_latent_trigger_cursor_iteration.py",
        "prompt_protocols": [
            "retrieval_decision",
            "bounded_hypothesis_generation",
            "typed_required_action_selection",
            "db_cursor_operation_selection",
            "graph_cursor_operation_selection",
        ],
        "seeds": SEEDS,
        "case_fingerprint": trigger_case_fingerprint(latent_trigger_cases()),
        "trigger_rows": len(trigger_rows),
        "db_rows": len(db_rows),
        "graph_rows": len(graph_rows),
        "transport_rows": len(transport_rows),
        "remote_transport_is_simulated": True,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            converted = {}
            for key, value in row.items():
                try:
                    converted[key] = float(value) if "." in value else int(value)
                except (TypeError, ValueError):
                    converted[key] = value
            rows.append(converted)
    return rows


if __name__ == "__main__":
    main()
