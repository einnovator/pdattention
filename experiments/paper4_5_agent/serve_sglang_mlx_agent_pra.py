"""Serve the frozen Paper 4.5 direct-engine gate on SGLang's MLX runner.

This process deliberately contains no :class:`PRAGateway`.  The HTTP boundary
only parses the OpenAI-compatible wire request and calls the engine adapter,
so direct-native measurements include no gateway selection, compaction, or
session lifecycle.
"""

from __future__ import annotations

import argparse
import json
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping

from pra_hf.deployment import PRAEngineResult, PRAWireRequest


def _completion(request: PRAWireRequest, result: PRAEngineResult) -> dict[str, Any]:
    raw = dict(result.raw)
    response: dict[str, Any] = {
        "id": request.request_id,
        "object": "chat.completion",
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.text},
            "finish_reason": "stop",
        }],
    }
    native = raw.get("pra") if isinstance(raw.get("pra"), Mapping) else {}
    response["pra"] = {
        **dict(native),
        "native_kv": bool(native.get("native_kv", native.get("native_kv_used"))),
        "native_attach_bytes": raw.get("native_attach_bytes"),
        "prefix_cache_hit": raw.get("prefix_cache_hit"),
        "prefix_cached_tokens": raw.get("prefix_cached_tokens"),
    }
    response["pra_trace"] = list(result.trace)
    return response


def _direct_handler(adapter: object, model: str):
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
                capabilities = adapter.capabilities().to_dict()
                capabilities["prefix_cache_enabled"] = bool(
                    adapter.prefix_cache_enabled
                )
                capabilities["chat_template_profile"] = (
                    adapter.chat_template_profile
                )
                capabilities["chat_template_digest"] = (
                    adapter.chat_template_digest
                )
                capabilities["additional_stop_token_ids"] = list(
                    adapter.additional_stop_token_ids
                )
                capabilities["effective_stop_token_ids"] = list(
                    adapter.effective_stop_token_ids
                )
                capabilities["default_repetition_penalty"] = (
                    adapter.default_repetition_penalty
                )
                capabilities["default_repeat_last_n"] = (
                    adapter.default_repeat_last_n
                )
                self._json(200, {
                    "status": "ok",
                    "endpoint_type": "engine",
                    "prefix_cache_enabled": bool(adapter.prefix_cache_enabled),
                    "chat_template_profile": adapter.chat_template_profile,
                    "chat_template_digest": adapter.chat_template_digest,
                    "additional_stop_token_ids": list(
                        adapter.additional_stop_token_ids
                    ),
                    "effective_stop_token_ids": list(
                        adapter.effective_stop_token_ids
                    ),
                    "default_repetition_penalty": (
                        adapter.default_repetition_penalty
                    ),
                    "default_repeat_last_n": adapter.default_repeat_last_n,
                    "effective_capabilities": capabilities,
                    "engine": capabilities,
                })
            elif path == "/v1/models":
                self._json(200, {
                    "object": "list",
                    "data": [{
                        "id": model,
                        "object": "model",
                        "owned_by": "sglang-mlx-pra",
                    }],
                })
            else:
                self._json(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            if urllib.parse.urlsplit(self.path).path != "/v1/chat/completions":
                self._json(404, {"error": "not_found"})
                return
            try:
                payload = json.loads(
                    self.rfile.read(int(self.headers.get("Content-Length", "0")))
                )
                if payload.get("stream"):
                    raise ValueError("streaming is disabled for the frozen agent gate")
                request = PRAWireRequest.from_openai(payload)
                result = adapter.generate(request)
                if bool(request.metadata.get("ephemeral_session", False)):
                    adapter.close_session(str(request.session_id))
                self._json(200, _completion(request, result))
            except (ValueError, TypeError, PermissionError) as error:
                self._json(400, {
                    "error": type(error).__name__,
                    "message": str(error),
                })
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
            try:
                adapter.close_session(session_id)
                self._json(200, {"closed": True, "session_id": session_id})
            except Exception as error:  # noqa: BLE001 - diagnostic boundary
                traceback.print_exc()
                self._json(500, {
                    "error": "engine_internal_error",
                    "message": str(error),
                })

        def log_message(self, format: str, *args: object) -> None:
            return None

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mlx-community/Qwen3-14B-4bit")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18121)
    parser.add_argument("--wire-tail-tokens", type=int, default=32)
    parser.add_argument("--pool-size", type=int, default=32768)
    parser.add_argument("--mem-fraction-static", type=float, default=0.25)
    parser.add_argument(
        "--additional-stop-token",
        action="append",
        default=[],
        help=(
            "Additional generated special token that terminates a response; "
            "repeat for model-specific renderer parity."
        ),
    )
    parser.add_argument(
        "--chat-template-profile",
        choices=("native", "qwen3-stable-no-thinking", "pure-chatml-stable"),
        default="qwen3-stable-no-thinking",
    )
    parser.add_argument("--default-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--default-repeat-last-n", type=int, default=64)
    args = parser.parse_args()

    from pra_hf.engine_memory import LogicalPRABlockStore
    from pra_sglang import (
        SGLangEngineAdapter,
        SGLangMLXAgentHistoryExecutor,
        configure_append_stable_template,
    )
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
    from transformers import AutoTokenizer

    block_store = LogicalPRABlockStore()
    runner = MlxModelRunner(
        args.model,
        revision=args.revision,
        disable_radix_cache=True,
        pool_size=args.pool_size,
        mem_fraction_static=args.mem_fraction_static,
        enable_sampling=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision
    )
    chat_template_digest = configure_append_stable_template(
        tokenizer, args.chat_template_profile
    )
    additional_stop_token_ids = []
    for token in args.additional_stop_token:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or (
            getattr(tokenizer, "unk_token_id", None) is not None
            and token_id == tokenizer.unk_token_id
            and token != getattr(tokenizer, "unk_token", None)
        ):
            raise ValueError(f"Additional stop token is unknown: {token!r}")
        additional_stop_token_ids.append(int(token_id))
    executor = SGLangMLXAgentHistoryExecutor(
        runner,
        tokenizer,
        model_id=args.model,
        model_revision=args.revision,
        block_store=block_store,
        wire_tail_tokens=args.wire_tail_tokens,
        chat_template_profile=args.chat_template_profile,
        chat_template_digest=chat_template_digest,
        additional_stop_token_ids=additional_stop_token_ids,
        default_repetition_penalty=args.default_repetition_penalty,
        default_repeat_last_n=args.default_repeat_last_n,
    )
    adapter = SGLangEngineAdapter(
        "http://in-process",
        block_store=block_store,
        native_executor=executor,
    )
    try:
        ThreadingHTTPServer(
            (args.host, args.port), _direct_handler(adapter, args.model)
        ).serve_forever()
    finally:
        executor.close()


if __name__ == "__main__":
    main()
