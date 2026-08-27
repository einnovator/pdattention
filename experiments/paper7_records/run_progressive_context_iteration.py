"""Run the corrected Paper 7 progressive PRA context-control benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.progressive_context_cases import (
    ContextCaseClass,
    ProgressiveContextCase,
    progressive_context_cases,
)
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
    ContextAction,
    ContextDecision,
    ContextTransport,
    PRAViewKind,
    ProgressiveContextRuntime,
    RecordCapabilities,
)


OUTPUT = ROOT / "docs/papers/shared/results/paper7_records/progressive_context"
FIGURES = OUTPUT / "figures"
MODEL_ID = "qwen3:0.6b"
MODEL_REVISION = "sha256-7f4030143c1c477224c5434f8272c662a8b042079a0a584f0a27a1684fe2e1fa"
SEEDS = (11, 23, 37, 53, 71)
PROTOCOL_REVISION = "progressive-context-v1"


class Baseline(str, Enum):
    FULL = "FULL"
    COMPACT_ONLY = "COMPACT_ONLY"
    PRA_AUTO = "PRA_AUTO"
    MODEL_ESCALATION = "MODEL_ESCALATION"
    PRA_PLUS_MODEL = "PRA_PLUS_MODEL"
    CCR_TOOL = "CCR_TOOL"


@dataclass(frozen=True)
class ModelResponse:
    value: Mapping[str, object]
    raw: str
    latency_seconds: float
    cache_hit: bool


class OllamaJSONModel:
    """Frozen Ollama JSON adapter with a content-addressed local cache."""

    def __init__(self, model: str, endpoint: str, cache_path: Path) -> None:
        self.model = model
        self.endpoint = endpoint.rstrip("/") + "/api/chat"
        self.cache_path = cache_path
        self.cache: dict[str, dict[str, object]] = {}
        if cache_path.is_file():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8-sig"))

    def chat(self, prompt: str, *, seed: int, max_tokens: int = 160) -> ModelResponse:
        request = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": "Return one valid JSON object only. /no_think",
                },
                {"role": "user", "content": prompt + "\n/no_think"},
            ],
            "options": {"temperature": 0, "seed": seed, "num_predict": max_tokens},
            "keep_alive": "30m",
        }
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        cached = self.cache.get(key)
        if cached is not None:
            return ModelResponse(
                cached["value"], str(cached["raw"]),
                float(cached["latency_seconds"]), True,
            )
        started = time.perf_counter()
        request_data = urllib.request.Request(
            self.endpoint,
            data=json.dumps(request).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request_data, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
        latency = time.perf_counter() - started
        raw = str(body.get("message", {}).get("content", "{}"))
        value = _json_object(raw)
        self.cache[key] = {"value": value, "raw": raw, "latency_seconds": latency}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, indent=2, sort_keys=True), encoding="utf-8"
        )
        return ModelResponse(value, raw, latency, False)


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


def _encoded_bytes(value: object) -> int:
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return len(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))


def _tokens(value: object) -> int:
    return math.ceil(_encoded_bytes(value) / 4)


def _context_payload(progressive: ProgressiveContextRuntime) -> list[object]:
    return [document.payload for document in progressive.registry.documents.values()]


def _contains_evidence(value: object, marker: str) -> bool:
    return marker.casefold() in json.dumps(value, sort_keys=True, default=str).casefold()


def _runtime(case: ProgressiveContextCase, seed: int):
    policies = {record_type: TypeContextPolicy(unit_limit=3) for record_type in RecordType}
    runtime = AdaptiveContextRuntime(
        RecordScope("paper7-progressive", f"{case.case_id}-{seed}"),
        ContextPolicy(
            topology=DeploymentTopology.REMOTE_MODEL,
            storage=StoragePolicy.ON_DEMAND,
            local_store=OUTPUT / ".stores" / f"{case.case_id}-{seed}-{time.time_ns()}",
            record_policies=policies,
            cursor_policy=CursorPolicy(page_size=4, max_page_size=16),
            persistent_store=False,
        ),
    )
    progressive = ProgressiveContextRuntime(runtime, chunk_tokens=32)
    record = runtime.ingest(
        case.payload,
        record_type=case.record_type,
        provenance={"source": "typed_result_fixture", "security_scope": "paper7-eval"},
    )
    capabilities = case.capabilities
    if case.case_class == ContextCaseClass.C3_CURSOR:
        cursor = runtime.open_cursor(record.record_id, collection=case.cursor_collection)
        first_page = runtime.fetch_cursor(cursor.cursor_id)
        operations = (
            (CursorOperation.NEXT,)
            if case.cursor_query is None
            else (CursorOperation.SEARCH,)
        )
        capabilities = replace(
            capabilities,
            cursor_available=True,
            cursor_id=cursor.cursor_id,
            has_more=first_page.has_more,
            allowed_cursor_operations=operations,
        )
        progressive.register_compact_record(record.record_id, capabilities=capabilities)
        progressive.registry.register_document(
            record.record_id,
            view=PRAViewKind.CURSOR_PAGE,
            payload=asdict(first_page),
            parent_uri=progressive.registry.views_by_record[record.record_id][0],
        )
    else:
        progressive.register_compact_record(record.record_id, capabilities=capabilities)
    return progressive, record


def _decision_prompt(
    case: ProgressiveContextCase,
    progressive: ProgressiveContextRuntime,
) -> str:
    record_id = next(iter(progressive.runtime.records))
    visible_codes = _visible_answer_codes(_context_payload(progressive))
    capabilities = progressive.registry.capabilities[record_id]
    allowed = [ContextAction.CONTINUE, ContextAction.MATERIALIZE_FULL]
    if case.tool_name:
        allowed.append(ContextAction.CALL_TOOL)
    if capabilities.partial_selectors:
        allowed.append(ContextAction.MATERIALIZE_MORE)
    if capabilities.searchable:
        allowed.append(ContextAction.SEARCH_RECORD)
    if capabilities.cursor_available:
        if CursorOperation.NEXT in capabilities.allowed_cursor_operations:
            allowed.append(ContextAction.CURSOR_NEXT)
        if any(
            operation != CursorOperation.NEXT
            for operation in capabilities.allowed_cursor_operations
        ):
            allowed.append(ContextAction.CURSOR_QUERY)
    visible_text = json.dumps(_decision_visible_context(progressive), sort_keys=True, default=str)
    external_marker = "requires_external_lookup" in visible_text
    return (
        "You control context disclosure for a typed PRA record. Decide whether the visible "
        "context is sufficient. You MUST use only an advertised allowed action. CONTINUE means the answer "
        "is already visible; if the requested ANSWER_CODE is visible, choose CONTINUE even when "
        "the record advertises has_more. MATERIALIZE_FULL is for a bounded exact backing record with no "
        "narrower useful interface. MATERIALIZE_MORE uses the advertised typed selector. "
        "SEARCH_RECORD searches one known large record. CURSOR_NEXT advances a sequential "
        "cursor; CURSOR_QUERY searches or filters it. CALL_TOOL is only for information absent "
        "from retained backing state. Never invent content.\n"
        "Apply this decision procedure exactly: (1) if Schema-parsed ANSWER_CODE values is "
        "non-empty, choose CONTINUE; (2) otherwise, NEVER choose CONTINUE; (3) if status says "
        "requires_external_lookup, choose CALL_TOOL; else if a cursor is available, choose its "
        "advertised CURSOR action; else if search is available, choose SEARCH_RECORD; else if "
        "partial selectors are available, choose MATERIALIZE_MORE; otherwise choose "
        "MATERIALIZE_FULL.\n"
        f"Task: {case.query}\n"
        f"Schema-parsed ANSWER_CODE values currently visible: {json.dumps(visible_codes)}\n"
        f"Allowed context actions: {json.dumps([action.value for action in allowed])}\n"
        f"Capabilities: {json.dumps(capabilities.prompt_descriptor(), sort_keys=True)}\n"
        f"External-information marker visible: {json.dumps(external_marker)}\n"
        "Return only the selected action class; the authorized runtime binds record, cursor, "
        "query, and selector arguments. Output exactly "
        '{"context_action":"ONE_ALLOWED_ACTION"}.'
    )


def _model_decision(
    model: OllamaJSONModel,
    case: ProgressiveContextCase,
    progressive: ProgressiveContextRuntime,
    seed: int,
) -> tuple[ContextDecision | None, ModelResponse, str | None]:
    response = model.chat(_decision_prompt(case, progressive), seed=seed, max_tokens=64)
    try:
        action = ContextAction(
            str(response.value.get("context_action", response.value.get("action", ""))).upper()
        )
        decision = _bind_context_action(action, case, progressive)
        return decision, response, None
    except (TypeError, ValueError) as exc:
        return None, response, f"{type(exc).__name__}: {exc}"


def _bind_context_action(
    action: ContextAction,
    case: ProgressiveContextCase,
    progressive: ProgressiveContextRuntime,
) -> ContextDecision:
    """Bind a model action class to the sole authorized benchmark target."""

    record_id = next(iter(progressive.runtime.records))
    if action == ContextAction.CONTINUE:
        return ContextDecision(action)
    if action == ContextAction.MATERIALIZE_FULL:
        return ContextDecision(action, record_id=record_id)
    if action == ContextAction.MATERIALIZE_MORE:
        return ContextDecision(action, record_id=record_id, selector=case.selector)
    if action == ContextAction.SEARCH_RECORD:
        return ContextDecision(
            action, record_id=record_id, query=case.search_query or case.query
        )
    if action == ContextAction.CALL_TOOL:
        return ContextDecision(action, tool_name=case.tool_name)
    cursor_id = progressive.runtime.cursors.cursor_ids[0]
    if action == ContextAction.CURSOR_NEXT:
        return ContextDecision(action, cursor_id=cursor_id)
    return ContextDecision(action, cursor_id=cursor_id, selector=case.cursor_query)


def _decision_visible_context(progressive: ProgressiveContextRuntime) -> list[object]:
    """Expose compact/derived payloads without repeated internal content hashes."""

    values = []
    for document in progressive.registry.documents.values():
        if document.view == PRAViewKind.SUMMARY and isinstance(document.payload, Mapping):
            values.append({
                "record_type": document.payload.get("record_type"),
                "view": "summary",
                "compact": document.payload.get("compact"),
            })
        else:
            values.append({"view": document.view.value, "payload": document.payload})
    return values


def _task_answer(
    model: OllamaJSONModel,
    case: ProgressiveContextCase,
    context: object,
    seed: int,
) -> tuple[str, ModelResponse]:
    visible_codes = _visible_answer_codes(context)
    prompt = (
        "Answer the task using only visible typed context. If an ANSWER_CODE is visible, return "
        "its actual value exactly. If it is absent, return UNKNOWN. Never return instruction "
        "placeholder text.\n"
        f"Task: {case.query}\nVisible context: "
        f"{json.dumps(context, sort_keys=True, default=str)}\n"
        f"Schema-parsed ANSWER_CODE values currently visible: {json.dumps(visible_codes)}\n"
        'Return {"answer":"<actual visible code>"} or {"answer":"UNKNOWN"}.'
    )
    response = model.chat(prompt, seed=seed, max_tokens=48)
    return str(response.value.get("answer", "UNKNOWN")), response


def _visible_answer_codes(value: object) -> list[str]:
    """Extract typed answer_code fields already present in the visible view."""

    codes = re.findall(
        r'["\']?answer_code["\']?\s*[:=]\s*["\']?([^"\'\s,}\\]+)',
        json.dumps(value, sort_keys=True, default=str),
        flags=re.I,
    )
    return list(dict.fromkeys(codes))


def _execute_with_tool_result(
    progressive: ProgressiveContextRuntime,
    case: ProgressiveContextCase,
    decision: ContextDecision,
):
    result = progressive.execute(decision, transport=ContextTransport.TOOL)
    if decision.action == ContextAction.CALL_TOOL and decision.tool_name == case.tool_name:
        progressive.ingest(
            case.tool_payload,
            record_type=RecordType.TOOL_RESPONSE,
            capabilities=RecordCapabilities(full_available=True, full_bounded=True),
            provenance={"tool_name": case.tool_name, "returned_for": case.case_id},
        )
    return result


def _pra_auto(progressive: ProgressiveContextRuntime, case: ProgressiveContextCase):
    """Run automatic PRA discovery before any explicit model escalation."""

    selection = progressive.automatic_select(case.query, top_k=2)
    record_id = next(iter(progressive.runtime.records))
    if selection.chunks and (
        case.capabilities.searchable or case.capabilities.partial_selectors
    ):
        query = case.search_query or f"needle-{case.case_id.split('-', 1)[1]}"
        result = progressive.search_record(record_id, query)
        return selection, result
    return selection, None


def _run_baseline(
    model: OllamaJSONModel,
    case: ProgressiveContextCase,
    seed: int,
    baseline: Baseline,
) -> dict[str, object]:
    progressive, record = _runtime(case, seed)
    compact_tokens = _tokens(_context_payload(progressive))
    predicted_action = ""
    decision_error = ""
    decision_latency = 0.0
    decision_target_action = ""
    explicit_expansions = 0
    pra_selected = 0
    mechanism_correct = False
    recognized_insufficient = False

    if baseline == Baseline.FULL:
        if case.case_class == ContextCaseClass.C5_TOOL:
            _execute_with_tool_result(
                progressive, case,
                ContextDecision(ContextAction.CALL_TOOL, tool_name=case.tool_name),
            )
        else:
            progressive.materialize_full(record.record_id)
        context = _context_payload(progressive)
    elif baseline == Baseline.COMPACT_ONLY:
        context = _context_payload(progressive)
        predicted_action = ContextAction.CONTINUE.value
    elif baseline == Baseline.PRA_AUTO:
        selection, result = _pra_auto(progressive, case)
        pra_selected = len(selection.chunks)
        context = _context_payload(progressive)
        predicted_action = "PRA_AUTO"
        mechanism_correct = bool(result) if case.case_class in {
            ContextCaseClass.C2_MORE, ContextCaseClass.C4_SEARCH
        } else case.case_class == ContextCaseClass.C0_CONTINUE
    elif baseline in {Baseline.MODEL_ESCALATION, Baseline.PRA_PLUS_MODEL}:
        auto_result = None
        if baseline == Baseline.PRA_PLUS_MODEL:
            selection, auto_result = _pra_auto(progressive, case)
            pra_selected = len(selection.chunks)
        evidence_before = _contains_evidence(_context_payload(progressive), case.evidence_marker)
        decision, response, decision_error = _model_decision(model, case, progressive, seed)
        decision_latency = response.latency_seconds
        if decision is not None:
            predicted_action = decision.action.value
            recognized_insufficient = decision.action != ContextAction.CONTINUE
            expected_now = (
                ContextAction.CONTINUE if evidence_before else case.expected_action
            )
            decision_target_action = expected_now.value
            mechanism_correct = decision.action == expected_now
            if decision.action != ContextAction.CONTINUE:
                _execute_with_tool_result(progressive, case, decision)
                explicit_expansions = 1
        context = _context_payload(progressive)
    else:
        # CCR exposes one generic retrieve operation, not mechanism-specific choices.
        prompt = (
            "Is this compact typed result sufficient to answer the task? Retrieve the retained "
            "original only when needed.\n"
            f"Task: {case.query}\nCompact: {json.dumps(_context_payload(progressive), default=str)}\n"
            'Return {"decision":"continue"} or {"decision":"retrieve"}.'
        )
        response = model.chat(prompt, seed=seed, max_tokens=48)
        decision_latency = response.latency_seconds
        retrieve = str(response.value.get("decision", "continue")).casefold() == "retrieve"
        predicted_action = "RETRIEVE" if retrieve else ContextAction.CONTINUE.value
        recognized_insufficient = retrieve
        mechanism_correct = (not retrieve) == (case.case_class == ContextCaseClass.C0_CONTINUE)
        if retrieve:
            progressive.materialize_full(record.record_id)
            explicit_expansions = 1
        context = _context_payload(progressive)

    evidence_recall = int(_contains_evidence(context, case.evidence_marker))
    answer, answer_response = _task_answer(model, case, context, seed)
    accounting = progressive.runtime.accounting()
    expansion_required = case.case_class != ContextCaseClass.C0_CONTINUE
    under_expansion = int(
        baseline in {Baseline.MODEL_ESCALATION, Baseline.PRA_PLUS_MODEL, Baseline.CCR_TOOL}
        and expansion_required and not evidence_recall
    )
    over_expansion = int(
        baseline in {Baseline.MODEL_ESCALATION, Baseline.PRA_PLUS_MODEL, Baseline.CCR_TOOL}
        and not expansion_required and explicit_expansions > 0
    )
    row = {
        "protocol_revision": PROTOCOL_REVISION,
        "model": model.model,
        "case_id": case.case_id,
        "case_class": case.case_class.value,
        "seed": seed,
        "baseline": baseline.value,
        "expected_initial_action": case.expected_action.value,
        "decision_target_action": decision_target_action,
        "predicted_action": predicted_action,
        "mechanism_correct": int(mechanism_correct),
        "recognized_insufficient": int(recognized_insufficient),
        "under_expansion": under_expansion,
        "over_expansion": over_expansion,
        "evidence_recall": evidence_recall,
        "task_success": int(answer == case.expected_answer),
        "answer": answer,
        "expected_answer": case.expected_answer,
        "compact_tokens": compact_tokens,
        "visible_tokens": _tokens(context),
        "materialized_tokens": math.ceil(accounting.materialized_bytes / 4),
        "materialized_bytes": accounting.materialized_bytes,
        "network_bytes": accounting.network_bytes,
        "active_kv_bytes": accounting.active_kv_bytes,
        "expansions": accounting.expansions + accounting.cursor_fetches,
        "explicit_expansions": explicit_expansions,
        "pra_selected_chunks": pra_selected,
        "round_trips": accounting.round_trips,
        "model_passes": 1 + int(baseline in {
            Baseline.MODEL_ESCALATION, Baseline.PRA_PLUS_MODEL, Baseline.CCR_TOOL
        }),
        "decision_latency_seconds": decision_latency,
        "answer_latency_seconds": answer_response.latency_seconds,
        "decision_error": decision_error or "",
        "decision_raw": response.raw if baseline in {
            Baseline.MODEL_ESCALATION, Baseline.PRA_PLUS_MODEL, Baseline.CCR_TOOL
        } else "",
        "full_ceiling_eligible": 1,
    }
    progressive.runtime.store.close()
    return row


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            converted = {}
            for key, value in row.items():
                try:
                    converted[key] = float(value) if "." in value else int(value)
                except (TypeError, ValueError):
                    converted[key] = value
            rows.append(converted)
    return rows


def _mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows) if rows else 0.0


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if not total:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return center - margin, center + margin


def _summaries(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["baseline"])].append(row)
    summaries = []
    for baseline in Baseline:
        values = grouped[baseline.value]
        successes = sum(int(row["task_success"]) for row in values)
        low, high = _wilson(successes, len(values))
        summaries.append({
            "baseline": baseline.value,
            "n": len(values),
            "task_success": successes / len(values),
            "task_success_ci_low": low,
            "task_success_ci_high": high,
            "evidence_recall": _mean(values, "evidence_recall"),
            "mechanism_correct": _mean(values, "mechanism_correct"),
            "under_expansion": _mean(values, "under_expansion"),
            "over_expansion": _mean(values, "over_expansion"),
            "visible_tokens": _mean(values, "visible_tokens"),
            "materialized_tokens": _mean(values, "materialized_tokens"),
            "network_bytes": _mean(values, "network_bytes"),
            "model_passes": _mean(values, "model_passes"),
            "latency_seconds": _mean(values, "decision_latency_seconds")
            + _mean(values, "answer_latency_seconds"),
        })
    return summaries


def _class_summaries(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(str(row["baseline"]), str(row["case_class"]))].append(row)
    result = []
    for (baseline, case_class), values in sorted(grouped.items()):
        result.append({
            "baseline": baseline,
            "case_class": case_class,
            "n": len(values),
            "task_success": _mean(values, "task_success"),
            "evidence_recall": _mean(values, "evidence_recall"),
            "mechanism_correct": _mean(values, "mechanism_correct"),
            "under_expansion": _mean(values, "under_expansion"),
            "over_expansion": _mean(values, "over_expansion"),
            "materialized_tokens": _mean(values, "materialized_tokens"),
        })
    return result


def _confusion(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    result = []
    for baseline in (Baseline.MODEL_ESCALATION, Baseline.PRA_PLUS_MODEL):
        values = [row for row in rows if row["baseline"] == baseline.value]
        counts = Counter(
            (
                str(row.get("decision_target_action") or row["expected_initial_action"]),
                str(row["predicted_action"]),
            )
            for row in values
        )
        for (expected, predicted), count in sorted(counts.items()):
            result.append({
                "baseline": baseline.value,
                "expected_action": expected,
                "predicted_action": predicted or "INVALID",
                "count": count,
            })
    return result


def _paired_bootstrap(
    rows: Sequence[Mapping[str, object]],
    left: Baseline,
    right: Baseline,
    *,
    iterations: int = 10_000,
) -> dict[str, object]:
    by_key = {(row["case_id"], row["seed"], row["baseline"]): row for row in rows}
    cases = sorted({str(row["case_id"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    deltas = [
        statistics.fmean(
            float(by_key[(case, seed, left.value)]["task_success"])
            - float(by_key[(case, seed, right.value)]["task_success"])
            for seed in seeds
        )
        for case in cases
    ]
    rng = random.Random(7107)
    sampled = sorted(
        statistics.fmean(rng.choice(deltas) for _ in deltas)
        for _ in range(iterations)
    )
    return {
        "left": left.value,
        "right": right.value,
        "metric": "task_success",
        "paired_case_delta": statistics.fmean(deltas),
        "bootstrap_ci_low": sampled[int(0.025 * iterations)],
        "bootstrap_ci_high": sampled[int(0.975 * iterations) - 1],
        "bootstrap_iterations": iterations,
        "cluster_unit": "case identity averaged over five decoding seeds",
    }


def _plots(
    rows: Sequence[Mapping[str, object]],
    summaries: Sequence[Mapping[str, object]],
    class_rows: Sequence[Mapping[str, object]],
    confusion: Sequence[Mapping[str, object]],
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {
        "FULL": "#1b4332", "COMPACT_ONLY": "#9d0208", "PRA_AUTO": "#2d6a4f",
        "MODEL_ESCALATION": "#457b9d", "PRA_PLUS_MODEL": "#ff9f1c", "CCR_TOOL": "#6c757d",
    }

    # 1. Context-control confusion matrix for the proposed controller.
    actions = [action.value for action in ContextAction]
    matrix = np.zeros((len(actions), len(actions)))
    selected = [row for row in confusion if row["baseline"] == Baseline.PRA_PLUS_MODEL.value]
    for row in selected:
        if row["expected_action"] in actions and row["predicted_action"] in actions:
            matrix[actions.index(row["expected_action"]), actions.index(row["predicted_action"])] += int(row["count"])
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    short = ["CONT", "FULL", "MORE", "SEARCH", "C-NEXT", "C-QUERY", "TOOL"]
    ax.set_xticks(range(len(actions)), short, rotation=35, ha="right")
    ax.set_yticks(range(len(actions)), short)
    ax.set_xlabel("Predicted context action")
    ax.set_ylabel("Required initial context action")
    for row in range(len(actions)):
        for column in range(len(actions)):
            ax.text(column, row, f"{normalized[row, column]:.2f}", ha="center", va="center",
                    color="white" if normalized[row, column] > 0.55 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label="Row-normalized rate")
    fig.tight_layout()
    _save(fig, "context_control_confusion")

    # 2. Task success against materialization cost.
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for row in summaries:
        ax.scatter(float(row["materialized_tokens"]), float(row["task_success"]),
                   s=90, color=colors[str(row["baseline"])], label=row["baseline"])
        ax.annotate(str(row["baseline"]), (float(row["materialized_tokens"]), float(row["task_success"])),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean materialized tokens")
    ax.set_ylabel("Final task success")
    ax.set_ylim(-0.03, 1.04)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "task_success_materialized_tokens")

    # 3. Under-expansion and over-expansion.
    controls = [Baseline.MODEL_ESCALATION.value, Baseline.PRA_PLUS_MODEL.value, Baseline.CCR_TOOL.value]
    values = {row["baseline"]: row for row in summaries}
    x = np.arange(len(controls))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(x - width / 2, [values[name]["under_expansion"] for name in controls], width, label="Under-expansion", color="#d62828")
    ax.bar(x + width / 2, [values[name]["over_expansion"] for name in controls], width, label="Over-expansion", color="#fcbf49")
    ax.set_xticks(x, ["Model", "PRA + model", "CCR tool"])
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, "under_over_expansion")

    # 4. Cursor task success and transfer.
    cursor = [row for row in class_rows if row["case_class"] == ContextCaseClass.C3_CURSOR.value]
    fig, ax1 = plt.subplots(figsize=(7.4, 4.8))
    names = [str(row["baseline"]) for row in cursor]
    transfer = [float(row["materialized_tokens"]) for row in cursor]
    success = [float(row["task_success"]) for row in cursor]
    x = np.arange(len(names))
    ax1.bar(x, transfer, color=[colors[name] for name in names], alpha=0.75)
    ax1.set_ylabel("Mean materialized tokens")
    ax1.set_xticks(x, names, rotation=30, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, success, color="#111111", marker="o", linewidth=2)
    ax2.set_ylabel("Task success")
    ax2.set_ylim(0, 1.05)
    fig.tight_layout()
    _save(fig, "cursor_success_transfer")

    # 5. Gain decomposition: automatic selection then explicit escalation.
    by_name = {row["baseline"]: row for row in summaries}
    compact = float(by_name[Baseline.COMPACT_ONLY.value]["task_success"])
    auto = float(by_name[Baseline.PRA_AUTO.value]["task_success"])
    combined = float(by_name[Baseline.PRA_PLUS_MODEL.value]["task_success"])
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    gains = [compact, auto - compact, combined - auto]
    bottoms = [0, compact, auto]
    ax.bar([0, 1, 2], gains, bottom=bottoms, color=["#9d0208", "#2d6a4f", "#ff9f1c"])
    ax.set_xticks([0, 1, 2], ["Compact", "+ PRA auto", "+ model escalation"])
    ax.set_ylabel("Cumulative task success")
    ax.set_ylim(0, 1.05)
    for index, (gain, bottom) in enumerate(zip(gains, bottoms)):
        ax.text(index, bottom + gain / 2, f"+{gain:.2f}", ha="center", va="center", color="white", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, "pra_model_gain_decomposition")


def _save(fig, name: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=190, bbox_inches="tight")
    plt.close(fig)


def _tex(summaries: Sequence[Mapping[str, object]], comparison: Mapping[str, object]) -> None:
    by_name = {row["baseline"]: row for row in summaries}
    def pct(value):
        return f"{100 * float(value):.1f}\\%"
    lines = [
        "% Generated by run_progressive_context_iteration.py; do not edit.",
        f"\\newcommand{{\\PaperSevenProgressiveN}}{{{sum(int(row['n']) for row in summaries):,}}}",
        f"\\newcommand{{\\PaperSevenFullTask}}{{{pct(by_name['FULL']['task_success'])}}}",
        f"\\newcommand{{\\PaperSevenCompactTask}}{{{pct(by_name['COMPACT_ONLY']['task_success'])}}}",
        f"\\newcommand{{\\PaperSevenPRAAutoTask}}{{{pct(by_name['PRA_AUTO']['task_success'])}}}",
        f"\\newcommand{{\\PaperSevenModelTask}}{{{pct(by_name['MODEL_ESCALATION']['task_success'])}}}",
        f"\\newcommand{{\\PaperSevenCombinedTask}}{{{pct(by_name['PRA_PLUS_MODEL']['task_success'])}}}",
        f"\\newcommand{{\\PaperSevenCCRTask}}{{{pct(by_name['CCR_TOOL']['task_success'])}}}",
        f"\\newcommand{{\\PaperSevenCombinedUnder}}{{{pct(by_name['PRA_PLUS_MODEL']['under_expansion'])}}}",
        f"\\newcommand{{\\PaperSevenCombinedOver}}{{{pct(by_name['PRA_PLUS_MODEL']['over_expansion'])}}}",
        f"\\newcommand{{\\PaperSevenCombinedTokens}}{{{float(by_name['PRA_PLUS_MODEL']['materialized_tokens']):.1f}}}",
        f"\\newcommand{{\\PaperSevenFullTokens}}{{{float(by_name['FULL']['materialized_tokens']):.1f}}}",
        f"\\newcommand{{\\PaperSevenCombinedDelta}}{{{float(comparison['paired_case_delta']):.3f}}}",
        f"\\newcommand{{\\PaperSevenCombinedDeltaLow}}{{{float(comparison['bootstrap_ci_low']):.3f}}}",
        f"\\newcommand{{\\PaperSevenCombinedDeltaHigh}}{{{float(comparison['bootstrap_ci_high']):.3f}}}",
    ]
    (OUTPUT / "generated_progressive_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _postprocess(rows: Sequence[Mapping[str, object]]) -> None:
    summaries = _summaries(rows)
    class_rows = _class_summaries(rows)
    confusion = _confusion(rows)
    comparison = _paired_bootstrap(rows, Baseline.PRA_PLUS_MODEL, Baseline.COMPACT_ONLY)
    _write_csv(OUTPUT / "progressive_baseline_summary.csv", summaries)
    _write_csv(OUTPUT / "progressive_class_summary.csv", class_rows)
    _write_csv(OUTPUT / "context_control_confusion.csv", confusion)
    (OUTPUT / "paired_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8"
    )
    _plots(rows, summaries, class_rows, confusion)
    _tex(summaries, comparison)
    full = next(row for row in summaries if row["baseline"] == Baseline.FULL.value)
    manifest = {
        "protocol": "Paper 7 corrected progressive PRA context-control benchmark",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seeds": list(SEEDS),
        "cases": len(progressive_context_cases()),
        "classes": [value.value for value in ContextCaseClass],
        "baselines": [value.value for value in Baseline],
        "rows": len(rows),
        "full_ceiling_task_success": full["task_success"],
        "full_ceiling_valid": float(full["task_success"]) >= 0.95,
        "statistics": "Wilson row intervals; 10,000-draw case-cluster bootstrap over five-seed means",
        "latent_trigger_diagnostic": "retained separately; not part of the primary benchmark",
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--postprocess-only", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--seed-limit", type=int)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows_path = OUTPUT / "progressive_context_rows.csv"
    if args.postprocess_only:
        _postprocess(_read_csv(rows_path))
        return
    model = OllamaJSONModel(
        args.model, args.endpoint, OUTPUT / "model_response_cache.json"
    )
    cases = list(progressive_context_cases())[: args.case_limit]
    seeds = list(SEEDS)[: args.seed_limit]
    rows = [] if args.fresh or not rows_path.is_file() else _read_csv(rows_path)
    rows = [
        row for row in rows
        if row.get("protocol_revision") == PROTOCOL_REVISION
        and row.get("model") == args.model
    ]
    completed = {
        (str(row["case_id"]), int(row["seed"]), str(row["baseline"]))
        for row in rows
    }
    for case in cases:
        for seed in seeds:
            for baseline in Baseline:
                key = (case.case_id, seed, baseline.value)
                if key in completed:
                    continue
                row = _run_baseline(model, case, seed, baseline)
                rows.append(row)
                completed.add(key)
                _write_csv(rows_path, rows)
                print(
                    f"{case.case_id} seed={seed} {baseline.value} "
                    f"task={row['task_success']} mechanism={row['mechanism_correct']}",
                    flush=True,
                )
    _postprocess(rows)


if __name__ == "__main__":
    main()
