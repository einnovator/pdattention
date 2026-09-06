"""Fair, agent-visible context treatments for the SWE-bench PRA frontier."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from pra_hf.large_record_index import LargeRecordIndex, LargeRecordSearchPolicy


_TOKEN = re.compile(r"\S+")


class ContextTreatment(str, Enum):
    """Request transformations compared after baseline reproduction."""

    PASSTHROUGH = "gateway-passthrough"
    TRUNCATION = "truncation"
    PRA_SELECTED_CONTEXT = "gateway-pra"
    DIRECT_NATIVE_PRA = "direct-native-pra"
    GATEWAY_NATIVE_PRA = "gateway-native-pra"


@dataclass(frozen=True)
class TreatmentTrace:
    """One request's disjoint logical, selected, and visible context accounting."""

    request_index: int
    session_id: str
    mode: str
    budget_fraction: float
    logical_input_tokens_estimate: int
    mandatory_tokens_estimate: int
    selected_tokens_estimate: int
    physical_input_tokens_estimate: int
    tokens_avoided_estimate: int
    token_saving_fraction_estimate: float
    candidate_segments: int
    selected_segments: int
    selected_resource_digest: str | None
    route_time_s: float
    token_estimator: str = "whitespace_v1"


def transform_chat_payload(
    payload: Mapping[str, Any],
    *,
    mode: ContextTreatment | str,
    budget_fraction: float,
    request_index: int = 0,
    segment_tokens: int = 256,
    frozen_selection: Sequence[tuple[str, str]] | None = None,
) -> tuple[dict[str, Any], TreatmentTrace]:
    """Apply a matched budget, optionally replaying an exact recorded selection."""

    mode = ContextTreatment(mode)
    if not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must be in (0, 1]")
    if segment_tokens <= 0:
        raise ValueError("segment_tokens must be positive")
    transformed = dict(payload)
    messages = [dict(row) for row in payload.get("messages", ())]
    if not messages:
        raise ValueError("chat payload requires messages")
    session_id = session_id_for_messages(messages)
    logical_tokens = sum(_count_tokens(row.get("content")) for row in messages)
    if mode is ContextTreatment.PASSTHROUGH:
        if frozen_selection is not None:
            raise ValueError("passthrough mode cannot replay a context selection")
        return transformed, _trace(
            request_index, session_id, mode, budget_fraction, logical_tokens, logical_tokens,
            0, logical_tokens, 0, 0, None, 0.0,
        )

    mandatory_indices = _mandatory_indices(messages)
    mandatory_tokens = sum(_count_tokens(messages[index].get("content")) for index in mandatory_indices)
    target_tokens = max(mandatory_tokens, math.ceil(logical_tokens * budget_fraction))
    available_tokens = max(0, target_tokens - mandatory_tokens)
    candidate_indices = [index for index in range(len(messages)) if index not in mandatory_indices]
    started = time.perf_counter()
    if mode is ContextTreatment.TRUNCATION:
        if frozen_selection is not None:
            raise ValueError("truncation mode cannot replay a context selection")
        selected = _truncate_recent(messages, candidate_indices, available_tokens)
        selected_tokens = sum(_count_tokens(row[1].get("content")) for row in selected)
        keep = [(index, messages[index]) for index in mandatory_indices] + selected
        transformed["messages"] = [row for _, row in sorted(keep, key=lambda item: item[0])]
        candidate_segments = len(candidate_indices)
        selected_segments = len(selected)
        selected_digest = _selection_digest(
            (f"m{index}", str(row.get("content", ""))) for index, row in selected
        )
    else:
        query = str(messages[max(mandatory_indices)].get("content", ""))
        segments = _segments(messages, candidate_indices, segment_tokens)
        selected_texts = (
            list(frozen_selection)
            if frozen_selection is not None
            else _select_segments(segments, query, available_tokens)
        )
        selected_tokens = sum(_count_tokens(text) for _, text in selected_texts)
        if selected_tokens > available_tokens:
            raise ValueError(
                "frozen selection exceeds the matched request context budget: "
                f"{selected_tokens} > {available_tokens}"
            )
        transformed["messages"] = [messages[index] for index in sorted(mandatory_indices)]
        resources = [
            {
                "resource_id": segment_id,
                "uri": f"pra://agent-trajectory/{segment_id}",
                "record_type": "agent_trajectory_segment",
                "text": text,
                "version": "v1",
                "source_fingerprint": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "authorization_scope": "swebench-agent-visible",
                "metadata": {"selection_policy": "typed_bm25_embedding_rrf"},
            }
            for segment_id, text in selected_texts
        ]
        envelope = dict(transformed.get("pra") or {})
        native_requested = mode in {
            ContextTreatment.DIRECT_NATIVE_PRA,
            ContextTreatment.GATEWAY_NATIVE_PRA,
        }
        envelope.update({
            "tenant_id": "paper4-5-swebench",
            "session_id": session_id,
            "resources": resources,
            "budget": {
                "max_resources": max(1, len(resources)),
                "max_selected_tokens": max(1, available_tokens),
            },
            "allow_text_fallback": not native_requested,
            "required_capabilities": ["logical_refs", "native_kv"] if native_requested else [],
            "pra_policy": {"profile": "swebench-balanced-v1"},
            "metadata": {
                "requested_mode": "native-memory" if native_requested else "selected-context",
                "connection": (
                    "direct" if mode is ContextTreatment.DIRECT_NATIVE_PRA else "gateway"
                ),
                "benchmark_fairness": "agent-visible-messages-only",
            },
        })
        transformed["pra"] = envelope
        candidate_segments = len(segments)
        selected_segments = len(selected_texts)
        selected_digest = _selection_digest(selected_texts)
    physical_tokens = mandatory_tokens + selected_tokens
    return transformed, _trace(
        request_index, session_id, mode, budget_fraction, logical_tokens, mandatory_tokens,
        selected_tokens, physical_tokens, candidate_segments, selected_segments,
        selected_digest, time.perf_counter() - started,
    )


def session_id_for_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    """Identify an agent task from its first non-system message without labels."""

    first_task_message = next(
        (row for row in messages if row.get("role") != "system"), messages[-1]
    )
    material = json.dumps(first_task_message, sort_keys=True, default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _mandatory_indices(messages: Sequence[Mapping[str, Any]]) -> set[int]:
    system = {index for index, row in enumerate(messages) if row.get("role") == "system"}
    latest_user = next(
        (index for index in range(len(messages) - 1, -1, -1) if messages[index].get("role") == "user"),
        len(messages) - 1,
    )
    return {*system, latest_user}


def _truncate_recent(
    messages: Sequence[Mapping[str, Any]], candidate_indices: Sequence[int], budget: int,
) -> list[tuple[int, dict[str, Any]]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    remaining = budget
    for index in reversed(candidate_indices):
        if remaining <= 0:
            break
        row = dict(messages[index])
        words = _words(row.get("content"))
        if len(words) > remaining:
            row["content"] = " ".join(words[-remaining:])
            words = words[-remaining:]
        selected.append((index, row))
        remaining -= len(words)
    return selected


def _segments(
    messages: Sequence[Mapping[str, Any]], candidate_indices: Sequence[int], segment_tokens: int,
) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    for index in candidate_indices:
        role = str(messages[index].get("role", "unknown"))
        words = _words(messages[index].get("content"))
        for offset in range(0, len(words), segment_tokens):
            text = " ".join(words[offset:offset + segment_tokens])
            if text:
                segments.append((f"m{index}-{offset // segment_tokens}-{role}", text))
    return segments


def _select_segments(
    segments: Sequence[tuple[str, str]], query: str, budget: int,
) -> list[tuple[str, str]]:
    if not segments or budget <= 0:
        return []
    index = LargeRecordIndex([text for _, text in segments])
    result = index.search(
        query, policy=LargeRecordSearchPolicy.HYBRID,
        top_k=len(segments), candidate_limit=len(segments),
    )
    selected: list[tuple[str, str]] = []
    selected_ids: set[str] = set()
    remaining = budget
    for hit in result.hits:
        words = _words(hit.text)
        if not words or remaining <= 0:
            continue
        text = hit.text if len(words) <= remaining else " ".join(words[:remaining])
        unit_index = int(hit.unit_id.split(":", 1)[1])
        segment_id = segments[unit_index][0]
        selected.append((segment_id, text))
        selected_ids.add(segment_id)
        remaining -= min(len(words), remaining)
    # Retrieval can return no positive score for a novel command. Fill the
    # residual budget from recent visible history, matching agent compaction's
    # conservative recency fallback without consulting benchmark labels.
    for segment_id, segment_text in reversed(segments):
        if remaining <= 0:
            break
        if segment_id in selected_ids:
            continue
        words = _words(segment_text)
        text = segment_text if len(words) <= remaining else " ".join(words[-remaining:])
        if text:
            selected.append((segment_id, text))
            remaining -= min(len(words), remaining)
    return selected


def _trace(
    request_index: int, session_id: str, mode: ContextTreatment, budget_fraction: float,
    logical: int, mandatory: int, selected: int, physical: int,
    candidates: int, selected_segments: int, selected_resource_digest: str | None,
    route_time_s: float,
) -> TreatmentTrace:
    avoided = max(0, logical - physical)
    return TreatmentTrace(
        request_index, session_id, mode.value, budget_fraction, logical, mandatory, selected,
        physical, avoided, avoided / logical if logical else 0.0,
        candidates, selected_segments, selected_resource_digest, route_time_s,
    )


def _selection_digest(rows: Sequence[tuple[str, str]]) -> str:
    """Fingerprint ordered selected records without retaining their content in metadata."""

    material = [
        {
            "resource_id": resource_id,
            "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        for resource_id, text in rows
    ]
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _words(value: Any) -> list[str]:
    if isinstance(value, str):
        return _TOKEN.findall(value)
    if value is None:
        return []
    return _TOKEN.findall(json.dumps(value, sort_keys=True, default=str))


def _count_tokens(value: Any) -> int:
    return len(_words(value))


class TreatmentProxy:
    """Small OpenAI-compatible proxy that persists a trace for every request."""

    def __init__(
        self, target_base_url: str, *, mode: ContextTreatment | str,
        budget_fraction: float, trace_path: Path,
        selection_record_path: Path | None = None,
        selection_replay_path: Path | None = None,
    ) -> None:
        if selection_record_path is not None and selection_replay_path is not None:
            raise ValueError("selection recording and replay are mutually exclusive")
        self.target_base_url = target_base_url.rstrip("/")
        self.mode = ContextTreatment(mode)
        self.budget_fraction = budget_fraction
        self.trace_path = trace_path
        self.selection_record_path = selection_record_path
        self.selection_replay_path = selection_replay_path
        self._frozen_selections = _load_selection_fixture(selection_replay_path)
        self._lock = threading.Lock()
        self._request_index = 0
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self, host: str = "127.0.0.1", port: int = 0) -> str:
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                proxy._forward(self)

            def do_POST(self) -> None:  # noqa: N802
                proxy._forward(self)

            def log_message(self, format: str, *args: Any) -> None:
                return None

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return f"http://{host}:{self._server.server_port}/v1"

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _forward(self, handler: BaseHTTPRequestHandler) -> None:
        body = handler.rfile.read(int(handler.headers.get("Content-Length", "0")))
        trace = None
        if handler.command == "POST" and urlparse(handler.path).path == "/v1/chat/completions":
            payload = json.loads(body.decode("utf-8"))
            with self._lock:
                self._request_index += 1
                request_index = self._request_index
            input_digest = _selection_input_digest(payload.get("messages", ()))
            frozen = self._frozen_selections.get(input_digest)
            if self.selection_replay_path is not None and frozen is None:
                raise RuntimeError(
                    "frozen selection replay has no exact request match for "
                    f"{input_digest}; the paired trajectories have diverged"
                )
            payload, trace = transform_chat_payload(
                payload, mode=self.mode, budget_fraction=self.budget_fraction,
                request_index=request_index, frozen_selection=frozen,
            )
            if self.selection_record_path is not None:
                resources = (payload.get("pra") or {}).get("resources") or ()
                self._record_selection(input_digest, trace, resources)
            body = json.dumps(payload).encode("utf-8")
        target = self.target_base_url.removesuffix("/v1") + handler.path
        headers = {
            key: value for key, value in handler.headers.items()
            if key.lower() not in {"host", "content-length", "connection"}
        }
        request = urllib.request.Request(target, data=body if handler.command == "POST" else None,
                                         headers=headers, method=handler.command)
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                response_body = response.read()
                handler.send_response(response.status)
                handler.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
        except urllib.error.HTTPError as error:
            response_body = error.read()
            handler.send_response(error.code)
            handler.send_header("Content-Type", error.headers.get("Content-Type", "application/json"))
        handler.send_header("Content-Length", str(len(response_body)))
        handler.end_headers()
        handler.wfile.write(response_body)
        if trace is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                with self.trace_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(asdict(trace), sort_keys=True) + "\n")

    def _record_selection(
        self, input_digest: str, trace: TreatmentTrace, resources: Sequence[Mapping[str, Any]],
    ) -> None:
        """Persist replayable selected content while telemetry retains hashes only."""

        row = {
            "request_input_sha256": input_digest,
            "session_id": trace.session_id,
            "selected_resource_digest": trace.selected_resource_digest,
            "resources": [
                {"resource_id": str(item["resource_id"]), "text": str(item["text"])}
                for item in resources
            ],
        }
        assert self.selection_record_path is not None
        self.selection_record_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.selection_record_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")


def _selection_input_digest(messages: Sequence[Mapping[str, Any]]) -> str:
    """Bind replay to the complete agent-visible request, including trajectory order."""

    encoded = json.dumps(messages, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _load_selection_fixture(path: Path | None) -> dict[str, list[tuple[str, str]]]:
    """Load a direct-run fixture and reject ambiguous duplicate request identities."""

    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"selection replay fixture does not exist: {path}")
    selections: dict[str, list[tuple[str, str]]] = {}
    digests: dict[str, str | None] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        request_digest = str(row["request_input_sha256"])
        resources = [
            (str(item["resource_id"]), str(item["text"]))
            for item in row.get("resources", ())
        ]
        selection_digest = _selection_digest(resources)
        expected_digest = row.get("selected_resource_digest")
        if expected_digest != selection_digest:
            raise ValueError(
                f"selection fixture line {line_number} failed its content digest"
            )
        if request_digest in selections and digests[request_digest] != selection_digest:
            raise ValueError(
                f"selection fixture has conflicting rows for request {request_digest}"
            )
        selections[request_digest] = resources
        digests[request_digest] = selection_digest
    return selections
