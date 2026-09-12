"""OpenAI-compatible text proxy for autonomous Paper 8.5 treatments.

The proxy owns no model state and knows nothing about K/V caches.  mini-swe-agent
keeps the canonical full trajectory.  For each ordinary chat request this
module derives a logical record plan, sends only the materialized text selected
for that decision, and records auditable content-token and exclusion metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import threading
from typing import Any, Callable, Mapping
from urllib.parse import urlparse
import urllib.error
import urllib.request

from .materialization import (
    MaterializationMode,
    ToolObservationMaterializer,
    materialize_plan,
)
from .model import AgentMemoryBudget, AgentMemoryPlan
from .negative_selection import (
    BashOperation,
    NEGATIVE_POLICY_RULES,
    NegativeHeuristicSelector,
    NegativeSelectionConfig,
    classify_bash_operation,
    reacquired_excluded_resources,
)
from .recordizer import extract_resource_ids, recordize_minisweagent_messages
from .selectors import FullHistorySelector, TokenCounter, whitespace_tokens
from .serialization import serialize_materialized_messages


_COMMAND = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.DOTALL)


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _query(messages: list[Mapping[str, Any]]) -> str:
    task = next(
        (str(row.get("content", "")) for row in messages if row.get("role") == "user"),
        "",
    )
    active = str(messages[-1].get("content", "")) if messages else ""
    return f"{task}\n{active}"


def _assistant_command(response_body: bytes) -> str | None:
    content = _assistant_content(response_body)
    if content is None:
        return None
    matches = _COMMAND.findall(content)
    return matches[0].strip() if len(matches) == 1 else None


def _assistant_content(response_body: bytes) -> str | None:
    try:
        payload = json.loads(response_body.decode("utf-8"))
        choices = payload.get("choices") or ()
        message = choices[0].get("message") if choices else {}
        content = message.get("content", "") if isinstance(message, Mapping) else ""
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, IndexError):
        return None
    return content if isinstance(content, str) else None


def join_instrumentation_sidecars(
    messages: list[dict[str, Any]],
    instrumentation_root: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach stripped observation metadata to a selector-only message copy.

    mini-swe-agent 2.4.6 removes message ``extra`` fields before calling the
    OpenAI endpoint.  The instrumented Docker environment writes an ordered
    receipt for every completed Bash action.  We join only when one receipt
    session has an exact, complete command-digest sequence.  Any missing,
    malformed, or ambiguous state returns the untouched copy, causing guarded
    negative rules to abstain.
    """

    copied = [dict(row) for row in messages]
    assistant_rows: list[tuple[int, str]] = []
    for index, message in enumerate(copied):
        if message.get("role") != "assistant":
            continue
        content = message.get("content", "")
        matches = _COMMAND.findall(content) if isinstance(content, str) else []
        if len(matches) != 1:
            return copied, {
                "status": "assistant_command_unparseable",
                "commands": len(assistant_rows),
                "receipts": 0,
                "joined": 0,
            }
        assistant_rows.append((index, matches[0].strip()))
    if not assistant_rows:
        return copied, {"status": "empty_exact", "commands": 0, "receipts": 0, "joined": 0}
    if instrumentation_root is None:
        return copied, {
            "status": "instrumentation_root_unset",
            "commands": len(assistant_rows),
            "receipts": 0,
            "joined": 0,
        }
    root = Path(instrumentation_root)
    paths = sorted(root.rglob("execution_*.json")) if root.is_dir() else []
    parents = {path.parent.resolve() for path in paths}
    if len(parents) > 1:
        return copied, {
            "status": "ambiguous_receipt_sessions",
            "commands": len(assistant_rows),
            "receipts": len(paths),
            "joined": 0,
        }
    receipts: list[dict[str, Any]] = []
    try:
        for path in paths:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(receipt, Mapping):
                raise ValueError("receipt is not an object")
            receipts.append(dict(receipt))
    except (OSError, json.JSONDecodeError, ValueError):
        return copied, {
            "status": "invalid_receipt",
            "commands": len(assistant_rows),
            "receipts": len(paths),
            "joined": 0,
        }
    receipts.sort(key=lambda row: int(row.get("step", -1)))
    steps = [row.get("step") for row in receipts]
    if (
        len(receipts) != len(assistant_rows)
        or steps != list(range(len(receipts)))
        or any(
            row.get("command_sha256") != hashlib.sha256(command.encode()).hexdigest()
            for (_, command), row in zip(assistant_rows, receipts)
        )
    ):
        return copied, {
            "status": "missing_or_mismatched_receipts",
            "commands": len(assistant_rows),
            "receipts": len(receipts),
            "joined": 0,
        }

    pending: list[tuple[int, dict[str, Any]]] = []
    for (assistant_index, _), receipt in zip(assistant_rows, receipts):
        metadata = receipt.get("observation_metadata")
        if not isinstance(metadata, Mapping):
            return copied, {
                "status": "invalid_observation_metadata",
                "commands": len(assistant_rows),
                "receipts": len(receipts),
                "joined": 0,
            }
        observation_index = assistant_index + 1
        if (
            observation_index >= len(copied)
            or copied[observation_index].get("role") not in {"user", "tool"}
        ):
            return copied, {
                "status": "missing_paired_observation",
                "commands": len(assistant_rows),
                "receipts": len(receipts),
                "joined": 0,
            }
        pending.append((observation_index, dict(metadata)))
    for observation_index, metadata in pending:
        existing = copied[observation_index].get("extra")
        if existing is not None and not isinstance(existing, Mapping):
            return [dict(row) for row in messages], {
                "status": "conflicting_inline_metadata",
                "commands": len(assistant_rows),
                "receipts": len(receipts),
                "joined": 0,
            }
        if isinstance(existing, Mapping) and any(
            key in existing and existing[key] != value
            for key, value in metadata.items()
        ):
            return [dict(row) for row in messages], {
                "status": "conflicting_inline_metadata",
                "commands": len(assistant_rows),
                "receipts": len(receipts),
                "joined": 0,
            }
        copied[observation_index]["extra"] = {**dict(existing or {}), **metadata}
    return copied, {
        "status": "exact",
        "commands": len(assistant_rows),
        "receipts": len(receipts),
        "joined": len(pending),
    }


@dataclass(frozen=True)
class AutonomousSelectionConfig:
    """Frozen policy and generation contract for one autonomous task arm."""

    policy: str = "full"
    budget_fraction: float = 1.0
    protected_head_turns: int = 1
    protected_tail_turns: int = 1
    search_delay_turns: int = 0
    write_delay_turns: int = 1
    same_span_reads_to_keep: int = 1
    working_set_resources: int = 4
    h2b_allow_workspace_verification: bool = False
    materialization_mode: MaterializationMode = MaterializationMode.WHOLE_RECORD
    materialization_threshold_tokens: int = 512
    materialization_head_lines: int = 20
    materialization_tail_lines: int = 30
    materialization_match_context_lines: int = 4
    materialization_max_matched_lines: int = 32
    expected_model: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    max_calls: int = 40
    max_completion_tokens: int | None = None
    tokenizer_identity: str = "whitespace_v1_diagnostic"
    task_id: str = "unassigned"
    require_exact_sidecars: bool = True

    def __post_init__(self) -> None:
        if self.policy != "full" and self.policy not in NEGATIVE_POLICY_RULES:
            raise ValueError(
                f"policy must be full or one of {tuple(NEGATIVE_POLICY_RULES)}"
            )
        if not 0 < self.budget_fraction <= 1:
            raise ValueError("budget_fraction must be in (0, 1]")
        if self.protected_head_turns < 0:
            raise ValueError("protected_head_turns cannot be negative")
        if self.protected_tail_turns < 1:
            raise ValueError("at least one tail turn is required to preserve current state")
        if self.max_calls < 1:
            raise ValueError("max_calls must be positive")
        if self.max_completion_tokens is not None and self.max_completion_tokens < 1:
            raise ValueError("max_completion_tokens must be positive")
        if self.policy == "full" and self.materialization_mode != MaterializationMode.WHOLE_RECORD:
            raise ValueError("FULL is an exact ordinary-text control and cannot compact records")

    def selector(self):
        if self.policy == "full":
            return FullHistorySelector()
        return NegativeHeuristicSelector(NegativeSelectionConfig(
            rules=NEGATIVE_POLICY_RULES[self.policy],
            search_delay_turns=self.search_delay_turns,
            write_delay_turns=self.write_delay_turns,
            same_span_reads_to_keep=self.same_span_reads_to_keep,
            working_set_resources=self.working_set_resources,
            protected_head_turns=self.protected_head_turns,
            protected_tail_turns=self.protected_tail_turns,
            h2b_allow_workspace_verification=self.h2b_allow_workspace_verification,
        ))


@dataclass(frozen=True)
class AutonomousTransformation:
    payload: dict[str, Any]
    plan: AgentMemoryPlan
    materialized_tokens: int
    trace: dict[str, Any]


def transform_autonomous_payload(
    payload: Mapping[str, Any],
    config: AutonomousSelectionConfig,
    *,
    count_tokens: TokenCounter = whitespace_tokens,
    instrumentation_root: Path | None = None,
) -> AutonomousTransformation:
    """Apply one logical policy without mutating the caller's full history."""

    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("chat request must contain a messages list")
    if bool(payload.get("stream")):
        raise ValueError("streaming chat is not supported by this audit proxy")
    messages = [dict(row) for row in raw_messages if isinstance(row, Mapping)]
    if len(messages) != len(raw_messages):
        raise ValueError("every chat message must be a JSON object")
    selector_messages, sidecar_join = join_instrumentation_sidecars(
        messages, instrumentation_root
    )
    history = recordize_minisweagent_messages(selector_messages)
    full_tokens = sum(count_tokens(row.content) for row in history.records)
    budget_tokens = max(1, math.ceil(full_tokens * config.budget_fraction))
    sidecar_exact = sidecar_join["status"] in {"exact", "empty_exact"}
    selection_abstained = bool(
        config.policy != "full" and config.require_exact_sidecars and not sidecar_exact
    )
    selector = FullHistorySelector() if selection_abstained else config.selector()
    plan = selector.select(
        history=history,
        query=_query(messages),
        budget=AgentMemoryBudget(max_tokens=budget_tokens),
        count_tokens=count_tokens,
    )
    materializer = ToolObservationMaterializer(
        mode=config.materialization_mode,
        threshold_tokens=config.materialization_threshold_tokens,
        head_lines=config.materialization_head_lines,
        tail_lines=config.materialization_tail_lines,
        match_context_lines=config.materialization_match_context_lines,
        max_matched_lines=config.materialization_max_matched_lines,
    )
    materialized = materialize_plan(
        history,
        plan,
        materializer,
        query=_query(messages),
        count_tokens=count_tokens,
    )

    all_record_ids = tuple(row.record_id for row in history.records)
    exact_logical_noop = (
        tuple(plan.selected_record_ids) == all_record_ids
        and all(
            row.mode == MaterializationMode.WHOLE_RECORD
            and row.content == history.record_by_id[row.record_id].content
            for row in materialized.records
        )
    )
    # FULL is the behavioral control.  A negative policy that currently has
    # nothing to remove must be the same control too: retain every incoming
    # message dictionary instead of silently changing the request envelope by
    # round-tripping through the logical serializer.  Once a real exclusion or
    # detail reduction occurs, emit only ordinary role/content records so
    # selector metadata never leaks into the model prompt.
    if config.policy == "full" or exact_logical_noop:
        selected_messages = messages
    else:
        selected_messages = serialize_materialized_messages(history, materialized)
    selected_ids = set(plan.selected_record_ids)
    immutable_ids = {
        row.record_id for row in history.records
        if row.primary_role.value in {"system", "task"}
    }
    if not immutable_ids.issubset(selected_ids):
        raise AssertionError("selector removed an immutable system/task record")
    if history.records and history.records[-1].record_id not in selected_ids:
        raise AssertionError("selector removed the current trajectory record")

    transformed = dict(payload)
    transformed["messages"] = selected_messages
    observations = [
        row for row in history.records if row.primary_role.value in {
            "tool_observation", "source_view", "verification", "error_or_rejection"
        }
    ]
    version_rows = sum(bool(row.metadata.get("resource_version_fingerprints")) for row in observations)
    completeness_rows = sum(row.metadata.get("output_complete") is not None for row in observations)
    excluded_tokens = sum(row.excluded_tokens for row in plan.exclusions)
    trace = {
        "schema_version": 1,
        "study": "paper8_5_autonomous_agent_memory",
        "task_id": config.task_id,
        "policy": config.policy,
        "plan_policy": plan.policy,
        "plan_digest": plan.digest,
        "request_input_sha256": _digest(messages),
        "selected_messages_sha256": _digest(selected_messages),
        "request_message_roles": [str(row.get("role", "")) for row in messages],
        "request_message_content_sha256": [
            hashlib.sha256(str(row.get("content", "")).encode("utf-8")).hexdigest()
            for row in messages
        ],
        "selected_message_content_sha256": [
            hashlib.sha256(str(row.get("content", "")).encode("utf-8")).hexdigest()
            for row in selected_messages
        ],
        "exact_request_passthrough": bool(
            config.policy == "full" or exact_logical_noop
        ),
        "tokenizer": config.tokenizer_identity,
        "token_accounting_scope": "message_content_only_excludes_chat_template",
        "requested_budget_fraction": config.budget_fraction,
        "requested_budget_tokens": budget_tokens,
        "full_tokens": full_tokens,
        "selected_tokens": plan.selected_tokens,
        "materialized_tokens": materialized.materialized_tokens,
        "logical_retention_fraction": plan.realized_retention_fraction,
        "materialized_retention_fraction": materialized.materialized_retention_fraction,
        "budget_satisfied": plan.selected_tokens <= budget_tokens,
        "full_message_count": len(messages),
        "selected_message_count": len(selected_messages),
        "selected_record_ids": list(plan.selected_record_ids),
        "selected_causal_group_ids": list(plan.selected_causal_group_ids),
        "excluded_tokens": excluded_tokens,
        "excluded_causal_group_count": len(plan.exclusions),
        "exclusions": [asdict(row) for row in plan.exclusions],
        "observation_metadata_coverage": {
            "observation_records": len(observations),
            "complete_status_records": completeness_rows,
            "resource_version_records": version_rows,
        },
        "instrumentation_sidecar_join": sidecar_join,
        "selection_abstained_for_sidecar": selection_abstained,
    }
    return AutonomousTransformation(transformed, plan, materialized.materialized_tokens, trace)


class AutonomousSelectionProxy:
    """Intercept non-streaming chat calls and forward selected ordinary text."""

    def __init__(
        self,
        upstream_base_url: str,
        *,
        config: AutonomousSelectionConfig,
        trace_path: Path,
        count_tokens: Callable[[str], int] = whitespace_tokens,
        upstream_api_key: str | None = None,
        timeout_seconds: int = 3600,
        instrumentation_root: Path | None = None,
    ) -> None:
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.config = config
        self.trace_path = Path(trace_path)
        self.count_tokens = count_tokens
        self.upstream_api_key = upstream_api_key
        self.timeout_seconds = timeout_seconds
        self.instrumentation_root = (
            Path(instrumentation_root) if instrumentation_root is not None else None
        )
        self._lock = threading.Lock()
        self._request_count = 0
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self, host: str = "127.0.0.1", port: int = 0) -> str:
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self) -> None:
                try:
                    proxy._forward(self)
                except (ValueError, TypeError, json.JSONDecodeError) as error:
                    self._error(400, "invalid_selection_request", error)
                except PermissionError as error:
                    self._error(429, "max_model_calls_exceeded", error)
                except (urllib.error.URLError, TimeoutError) as error:
                    self._error(502, "selection_upstream_unavailable", error)
                except Exception as error:  # pragma: no cover - defensive HTTP boundary
                    self._error(500, "selection_proxy_internal_error", error)

            def _error(self, status: int, code: str, error: Exception) -> None:
                body = json.dumps({"error": code, "message": str(error)}).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if urlparse(self.path).path == "/health":
                    body = json.dumps(proxy.health()).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self._handle()

            def do_POST(self) -> None:  # noqa: N802
                self._handle()

            def log_message(self, format: str, *args: Any) -> None:
                return None

        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._server = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return f"http://{host}:{self._server.server_port}/v1"

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "protocol": "paper8.5-autonomous-text-selection-v1",
            "policy": self.config.policy,
            "engine_state_owned": False,
            "kv_metrics_available": False,
            "request_count": self._request_count,
            "max_calls": self.config.max_calls,
        }

    def _validate_generation(self, payload: Mapping[str, Any]) -> None:
        if self.config.expected_model is not None and payload.get("model") != self.config.expected_model:
            raise ValueError(
                f"model mismatch: expected {self.config.expected_model!r}, "
                f"observed {payload.get('model')!r}"
            )
        expected = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "seed": self.config.seed,
        }
        for key, value in expected.items():
            if key not in payload or payload[key] != value:
                raise ValueError(
                    f"generation mismatch for {key}: expected {value!r}, "
                    f"observed {payload.get(key)!r}"
                )
        if self.config.max_completion_tokens is not None:
            observed = payload.get("max_completion_tokens", payload.get("max_tokens"))
            if observed != self.config.max_completion_tokens:
                raise ValueError(
                    "completion limit mismatch: expected "
                    f"{self.config.max_completion_tokens}, observed {observed!r}"
                )

    def _target(self, incoming_path: str) -> str:
        root = self.upstream_base_url.removesuffix("/v1")
        return root + incoming_path

    def _forward(self, handler: BaseHTTPRequestHandler) -> None:
        body = handler.rfile.read(int(handler.headers.get("Content-Length", "0")))
        transformation: AutonomousTransformation | None = None
        request_index: int | None = None
        if handler.command == "POST" and urlparse(handler.path).path == "/v1/chat/completions":
            payload = json.loads(body.decode("utf-8"))
            self._validate_generation(payload)
            with self._lock:
                if self._request_count >= self.config.max_calls:
                    raise PermissionError(
                        f"maximum {self.config.max_calls} model calls reached"
                    )
                self._request_count += 1
                request_index = self._request_count
            transformation = transform_autonomous_payload(
                payload,
                self.config,
                count_tokens=self.count_tokens,
                instrumentation_root=self.instrumentation_root,
            )
            body = json.dumps(transformation.payload).encode("utf-8")

        headers = {
            key: value for key, value in handler.headers.items()
            if key.lower() not in {"host", "content-length", "connection"}
        }
        if self.upstream_api_key is not None:
            headers["Authorization"] = f"Bearer {self.upstream_api_key}"
        request = urllib.request.Request(
            self._target(handler.path),
            data=body if handler.command == "POST" else None,
            headers=headers,
            method=handler.command,
        )
        status = 200
        response_headers: Mapping[str, str]
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_body = response.read()
                status = response.status
                response_headers = response.headers
        except urllib.error.HTTPError as error:
            response_body = error.read()
            status = error.code
            response_headers = error.headers

        if transformation is not None:
            assistant_content = _assistant_content(response_body)
            command = _assistant_command(response_body)
            operation = classify_bash_operation(command)
            resources = extract_resource_ids(command, "") if command else ()
            reacquired = reacquired_excluded_resources(command, transformation.plan.exclusions)
            trace = {
                **transformation.trace,
                "request_index": request_index,
                "generation": {
                    "model": transformation.payload.get("model"),
                    "temperature": transformation.payload.get("temperature"),
                    "top_p": transformation.payload.get("top_p"),
                    "seed": transformation.payload.get("seed"),
                    "max_completion_tokens": transformation.payload.get(
                        "max_completion_tokens", transformation.payload.get("max_tokens")
                    ),
                },
                "upstream_status": status,
                "response_sha256": hashlib.sha256(response_body).hexdigest(),
                "response_sha256_scope": (
                    "raw_http_body_includes_volatile_response_metadata"
                ),
                "assistant_content_sha256": (
                    hashlib.sha256(assistant_content.encode("utf-8")).hexdigest()
                    if assistant_content is not None else None
                ),
                "assistant_command_sha256": (
                    hashlib.sha256(command.encode("utf-8")).hexdigest() if command else None
                ),
                "assistant_operation": operation.value if command else None,
                "assistant_resource_ids": list(resources),
                "assistant_is_search": operation is BashOperation.SEARCH_DISCOVERY,
                "assistant_is_read": operation in {BashOperation.READ, BashOperation.DIFF},
                "assistant_is_test": operation is BashOperation.VERIFY,
                "reacquired_excluded_resources": list(reacquired),
                "reacquisition_count": len(reacquired),
                "reacquisition_proxy_for_false_exclusion": bool(reacquired),
            }
            self._append_trace(trace)

        handler.send_response(status)
        for key in ("Content-Type", "Retry-After"):
            if value := response_headers.get(key):
                handler.send_header(key, value)
        handler.send_header("Content-Length", str(len(response_body)))
        handler.end_headers()
        handler.wfile.write(response_body)

    def _append_trace(self, trace: Mapping[str, Any]) -> None:
        row = json.dumps(dict(trace), sort_keys=True, default=str) + "\n"
        with self._lock:
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(row)
                handle.flush()
