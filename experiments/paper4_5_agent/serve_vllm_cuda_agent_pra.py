"""Direct OpenAI-compatible vLLM V1 scheduler-alias agent server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from pra_hf.deployment import PRAEngineResult, PRAWireRequest


def _completion(request: PRAWireRequest, result: PRAEngineResult) -> dict[str, Any]:
    raw = dict(result.raw)
    pra = raw.get("pra") if isinstance(raw.get("pra"), Mapping) else {}
    return {
        "id": request.request_id,
        "object": "chat.completion",
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.text},
            "finish_reason": "stop",
        }],
        "pra": {
            **dict(pra),
            "native_kv": bool(pra.get("native_kv_used")),
            "native_attach_bytes": raw.get("native_attach_bytes"),
            "prefix_cache_hit": raw.get("prefix_cache_hit"),
            "prefix_cached_tokens": raw.get("prefix_cached_tokens"),
        },
        "pra_trace": list(result.trace),
    }


def _handler(executor: object, model_id: str):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: Mapping[str, Any]) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urllib.parse.urlsplit(self.path).path
            if path == "/health":
                caps = dict(executor.capabilities())
                self._json(200, {
                    "status": "ok",
                    "endpoint_type": "engine",
                    "gateway_mode": None,
                    "prefix_cache_enabled": True,
                    "chat_template_profile": executor.chat_template_profile,
                    "chat_template_digest": executor.chat_template_digest,
                    "effective_capabilities": caps,
                    "engine": caps,
                })
            elif path == "/v1/models":
                self._json(200, {
                    "object": "list",
                    "data": [{"id": model_id, "object": "model", "owned_by": "vllm-pra"}],
                })
            else:
                self._json(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            if urllib.parse.urlsplit(self.path).path != "/v1/chat/completions":
                self._json(404, {"error": "not_found"})
                return
            try:
                value = json.loads(
                    self.rfile.read(int(self.headers.get("Content-Length", "0")))
                )
                if value.get("stream"):
                    raise ValueError("streaming is disabled for the frozen agent gate")
                request = PRAWireRequest.from_openai(value)
                result = executor.generate(request)
                self._json(200, _completion(request, result))
            except (ValueError, TypeError, PermissionError) as error:
                self._json(400, {"error": type(error).__name__, "message": str(error)})
            except Exception as error:  # noqa: BLE001
                traceback.print_exc()
                self._json(500, {"error": "engine_internal_error", "message": str(error)})

        def do_DELETE(self) -> None:  # noqa: N802
            prefix = "/v1/pra/sessions/"
            path = urllib.parse.urlsplit(self.path).path
            if not path.startswith(prefix):
                self._json(404, {"error": "not_found"})
                return
            session_id = urllib.parse.unquote(path[len(prefix) :])
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
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18125)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--storage", default=".pra/vllm-cuda-agent")
    args = parser.parse_args()

    from vllm import LLM

    from pra_hf.agent_executor import validate_append_stable_template
    from pra_vllm.agent_executor import (
        VLLMCudaAgentHistoryExecutor,
        VLLMInProcessSchedulerDriver,
    )

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        enforce_eager=True,
        disable_hybrid_kv_cache_manager=True,
        kv_transfer_config={
            "kv_connector": "PRASparseConnector",
            "kv_connector_module_path": "pra_vllm.cuda_sparse_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "storage_path": str(Path(args.storage).resolve()),
                "scheduler_page_aliases": True,
            },
        },
    )
    tokenizer = llm.get_tokenizer()
    validate_append_stable_template(tokenizer)
    digest = hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest()
    executor = VLLMCudaAgentHistoryExecutor(
        VLLMInProcessSchedulerDriver(llm),
        tokenizer,
        model_id=args.model,
        chat_template_digest=digest,
    )
    ThreadingHTTPServer((args.host, args.port), _handler(executor, args.model)).serve_forever()


if __name__ == "__main__":
    main()
