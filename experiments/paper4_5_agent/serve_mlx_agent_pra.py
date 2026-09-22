"""Serve direct stateful mlx-lm agent-history K/V without a PRA gateway."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.metadata
import json
from pathlib import Path
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping

from pra_hf.deployment import PRAEngineResult, PRAWireRequest
from experiments.paper4_5_agent.openai_response import completion_finish_reason
from pra_hf.live_history import LiveKVSessionTerminatedError


def _completion(request: PRAWireRequest, result: PRAEngineResult) -> dict[str, Any]:
    raw = dict(result.raw)
    native = raw.get("pra") if isinstance(raw.get("pra"), Mapping) else {}
    response = {
        "id": request.request_id,
        "object": "chat.completion",
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.text},
            "finish_reason": completion_finish_reason(request, result),
        }],
        "pra": {
            **dict(native),
            "native_kv": bool(native.get("native_kv_used")),
            "native_attach_bytes": raw.get("native_attach_bytes"),
            "prefix_cache_hit": raw.get("prefix_cache_hit"),
            "prefix_cached_tokens": raw.get("prefix_cached_tokens"),
        },
        "pra_trace": list(result.trace),
    }
    if isinstance(raw.get("usage"), Mapping):
        response["usage"] = dict(raw["usage"])
    return response


def _runtime_identity(
    pra_source_revision: str,
    observed_source_checkpoint_revision: str,
) -> dict[str, Any]:
    """Describe the packages that execute MLX requests, not the caller venv."""

    packages: dict[str, str | None] = {}
    records: dict[str, str | None] = {}
    for package in ("mlx", "mlx-lm"):
        try:
            distribution = importlib.metadata.distribution(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
            records[package] = None
            continue
        packages[package] = distribution.version
        record = distribution.read_text("RECORD")
        records[package] = (
            hashlib.sha256(record.encode("utf-8")).hexdigest()
            if record else None
        )
    return {
        "packages": packages,
        "package_record_sha256": records,
        "pra_source_revision": pra_source_revision,
        "observed_source_checkpoint_revision": observed_source_checkpoint_revision,
    }


def _direct_handler(
    executor: object,
    model_id: str,
    runtime_identity: Mapping[str, Any] | None = None,
    request_log: Path | None = None,
    run_token: str | None = None,
    interaction_log: Path | None = None,
):
    log_lock = threading.Lock()

    def write_receipt(payload: Mapping[str, Any]) -> None:
        if request_log is None:
            return
        request_log.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, sort_keys=True, default=str)
        with log_lock, request_log.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")

    def write_interaction(payload: Mapping[str, Any]) -> None:
        if interaction_log is None:
            return
        interaction_log.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, sort_keys=True, default=str)
        with log_lock, interaction_log.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")

    def economical_message(row: Mapping[str, Any], index: int) -> dict[str, Any]:
        content = str(row.get("content") or "")
        limit = 8_000
        visible = content
        elided = 0
        if len(content) > limit:
            visible = content[:4_000] + "\n...[ELIDED]...\n" + content[-4_000:]
            elided = len(content) - 8_000
        return {
            "message_index": index,
            "role": row.get("role"),
            "content": visible,
            "content_chars": len(content),
            "elided_chars": elided,
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }

    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: Mapping[str, Any]) -> None:
            encoded = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            path = urllib.parse.urlsplit(self.path).path
            if path == "/health":
                capabilities = dict(executor.capabilities())
                self._json(200, {
                    "status": "ok",
                    "endpoint_type": "engine",
                    "gateway_mode": None,
                    "prefix_cache_enabled": False,
                    "max_model_len": executor.max_model_len,
                    "chat_template_profile": executor.chat_template_profile,
                    "chat_template_digest": executor.chat_template_digest,
                    "runtime_identity": dict(runtime_identity or {}),
                    "effective_capabilities": capabilities,
                    "engine": capabilities,
                })
            elif path == "/v1/models":
                self._json(200, {
                    "object": "list",
                    "data": [{
                        "id": model_id,
                        "object": "model",
                        "owned_by": "mlx-lm-direct-pra",
                    }],
                })
            else:
                self._json(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            if urllib.parse.urlsplit(self.path).path != "/v1/chat/completions":
                self._json(404, {"error": "not_found"})
                return
            if run_token is not None and not hmac.compare_digest(
                str(self.headers.get("X-PRA-Run-Lease") or ""), run_token
            ):
                self._json(409, {
                    "error": "stale_run_lease",
                    "message": "request does not own this frozen engine run",
                })
                return
            try:
                payload = json.loads(
                    self.rfile.read(int(self.headers.get("Content-Length", "0")))
                )
                if payload.get("stream"):
                    raise ValueError("streaming is disabled for the frozen agent gate")
                request = PRAWireRequest.from_openai(payload)
                started = time.perf_counter()
                messages = payload.get("messages") or []
                message_receipts = [
                    {
                        "role": row.get("role"),
                        "content_chars": len(str(row.get("content") or "")),
                        "content_sha256": hashlib.sha256(
                            str(row.get("content") or "").encode("utf-8")
                        ).hexdigest(),
                    }
                    for row in messages
                    if isinstance(row, Mapping)
                ]
                rendered_tokens = executor.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
                write_receipt({
                    "event": "request_start",
                    "request_id": request.request_id,
                    "session_id": request.session_id,
                    "body_sha256": hashlib.sha256(
                        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
                    ).hexdigest(),
                    "payload_keys": sorted(payload),
                    "message_count": len(messages),
                    "messages": message_receipts,
                    "rendered_prompt_tokens": len(rendered_tokens),
                    "max_tokens": payload.get("max_tokens"),
                    "temperature": payload.get("temperature"),
                    "top_p": payload.get("top_p"),
                    "seed": payload.get("seed"),
                })
                # The first request records the system/task pair. Later calls
                # record only the newly completed action/observation pair,
                # keeping a human-auditable trajectory without quadratic dumps.
                tail_start = 0 if len(messages) <= 2 else max(0, len(messages) - 2)
                write_interaction({
                    "event": "request_messages",
                    "request_id": request.request_id,
                    "message_count": len(messages),
                    "messages": [
                        economical_message(row, index)
                        for index, row in enumerate(messages[tail_start:], tail_start)
                        if isinstance(row, Mapping)
                    ],
                })
                result = executor.generate(request)
                if bool(request.metadata.get("ephemeral_session", False)):
                    executor.close_session(str(request.session_id))
                response = _completion(request, result)
                write_receipt({
                    "event": "request_end",
                    "request_id": request.request_id,
                    "session_id": request.session_id,
                    "elapsed_seconds": time.perf_counter() - started,
                    "usage": response.get("usage"),
                    "finish_reason": response["choices"][0]["finish_reason"],
                })
                self._json(200, response)
            except LiveKVSessionTerminatedError as error:
                self._json(409, {
                    "error": "session_terminated",
                    "message": str(error),
                })
            except (ValueError, TypeError, PermissionError) as error:
                self._json(400, {"error": type(error).__name__, "message": str(error)})
            except Exception as error:  # noqa: BLE001 - diagnostic boundary
                traceback.print_exc()
                self._json(500, {
                    "error": "engine_internal_error",
                    "message": str(error),
                })

        def do_DELETE(self) -> None:  # noqa: N802
            prefix = "/v1/pra/sessions/"
            path = urllib.parse.urlsplit(self.path).path
            if not path.startswith(prefix):
                self._json(404, {"error": "not_found"})
                return
            session_id = urllib.parse.unquote(path[len(prefix):])
            if not session_id or "/" in session_id:
                self._json(400, {"error": "invalid_session_id"})
                return
            executor.close_session(session_id)
            self._json(200, {"closed": True, "session_id": session_id})

        def log_message(self, format: str, *args: object) -> None:
            return None

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument(
        "--served-model",
        help="Stable OpenAI model name; defaults to the model load path.",
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--pra-source-revision",
        default="NOT_RECORDED",
        help="Immutable pdattention source revision advertised in /health.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18124)
    parser.add_argument(
        "--request-log",
        type=Path,
        help="Optional JSONL request-shape and timing receipt (no message text).",
    )
    parser.add_argument(
        "--run-token",
        help=(
            "Optional frozen run lease. POST requests must present the same "
            "value in X-PRA-Run-Lease, preventing stale agents from crossing "
            "a server restart or task boundary."
        ),
    )
    parser.add_argument(
        "--interaction-log",
        type=Path,
        help="Optional economical JSONL action/observation trajectory.",
    )
    parser.add_argument("--wire-tail-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-abs-logit-delta", type=float, default=0.005)
    parser.add_argument(
        "--require-full-retention-reference",
        action="store_true",
        help=(
            "Qualification-only: compare every PRA-100 output token and logit "
            "step with a live native prefix-cache reference."
        ),
    )
    parser.add_argument(
        "--agent-history-qualified",
        action="store_true",
        help=(
            "Advertise sparse agent-history qualification for this explicitly "
            "validated model/profile; same-subset checking remains mandatory."
        ),
    )
    parser.add_argument(
        "--fused-disjoint-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the fused interval-addressed Metal consumer. Disable only for "
            "the eager same-subset diagnostic/control path."
        ),
    )
    parser.add_argument(
        "--chat-template-profile",
        choices=("native", "qwen3-stable-no-thinking", "pure-chatml-stable"),
        default="qwen3-stable-no-thinking",
    )
    args = parser.parse_args()
    served_model = args.served_model or args.model

    from mlx_lm import load

    from pra_hf.agent_executor import (
        configure_append_stable_template,
        validate_append_stable_template,
    )
    from pra_mlx import MLXAgentHistoryExecutor
    from experiments.paper4_5_agent.runtime_identity import (
        require_source_checkpoint_revision,
    )

    # mlx-lm resolves a local snapshot path or Hub identifier itself.  The
    # required revision remains part of the executor's compatibility identity;
    # use a pinned local snapshot path when exact Hub revision loading matters.
    model, tokenizer = load(args.model)
    observed_revision = require_source_checkpoint_revision(
        args.model,
        args.revision,
        runtime_config=getattr(model, "config", None),
    )
    template_digest = configure_append_stable_template(
        tokenizer, args.chat_template_profile
    )
    template_kwargs = (
        {"enable_thinking": False}
        if args.chat_template_profile == "qwen3-stable-no-thinking"
        else None
    )
    validate_append_stable_template(
        tokenizer, chat_template_kwargs=template_kwargs
    )
    executor = MLXAgentHistoryExecutor(
        model,
        tokenizer,
        model_id=served_model,
        model_revision=args.revision,
        wire_tail_tokens=args.wire_tail_tokens,
        chat_template_profile=args.chat_template_profile,
        chat_template_digest=template_digest,
        max_abs_logit_delta=args.max_abs_logit_delta,
        require_full_retention_reference=args.require_full_retention_reference,
        agent_history_qualified=args.agent_history_qualified,
        fused_disjoint_attention=args.fused_disjoint_attention,
        max_model_len=args.max_model_len,
    )
    try:
        ThreadingHTTPServer(
            (args.host, args.port),
            _direct_handler(
                executor,
                served_model,
                _runtime_identity(args.pra_source_revision, observed_revision),
                args.request_log,
                args.run_token,
                args.interaction_log,
            ),
        ).serve_forever()
    finally:
        executor.close()


if __name__ == "__main__":
    main()
