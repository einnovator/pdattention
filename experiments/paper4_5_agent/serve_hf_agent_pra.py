"""Serve the Paper 4.5 direct HF live-agent K/V gate without a gateway."""

from __future__ import annotations

import argparse
import hmac
import json
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

from pra_hf.deployment import PRAEngineResult, PRAWireRequest
from experiments.paper4_5_agent.openai_response import completion_finish_reason


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


def _direct_handler(
    executor: object,
    model_id: str,
    runtime_identity: Mapping[str, Any] | None = None,
    run_token: str | None = None,
):
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
                    "dense_attention_implementation": getattr(
                        executor, "dense_attention_implementation", None
                    ),
                    "load_in_4bit": bool(getattr(executor, "load_in_4bit", False)),
                    "cpu_offload_enabled": bool(
                        getattr(executor, "cpu_offload_enabled", False)
                    ),
                    "max_gpu_memory_mib": getattr(
                        executor, "max_gpu_memory_mib", None
                    ),
                    "max_cpu_memory_gib": getattr(
                        executor, "max_cpu_memory_gib", None
                    ),
                    "hf_device_map": dict(
                        getattr(executor, "hf_device_map", {})
                    ),
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
                        "owned_by": "huggingface-direct-pra",
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
                self._json(409, {"error": "stale_run_lease"})
                return
            request = None
            try:
                payload = json.loads(
                    self.rfile.read(int(self.headers.get("Content-Length", "0")))
                )
                if payload.get("stream"):
                    raise ValueError("streaming is disabled for the frozen agent gate")
                request = PRAWireRequest.from_openai(payload)
                result = executor.generate(request)
                self._json(200, _completion(request, result))
            except (ValueError, TypeError, PermissionError) as error:
                self._json(400, {"error": type(error).__name__, "message": str(error)})
            except Exception as error:  # noqa: BLE001 - diagnostic boundary
                traceback.print_exc()
                self._json(500, {
                    "error": "engine_internal_error",
                    "message": str(error),
                })
            finally:
                if request is not None and bool(
                    request.metadata.get("ephemeral_session", False)
                ):
                    executor.close_session(str(request.session_id))

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
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--served-model",
        help="Stable OpenAI model name; defaults to the model load path.",
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18123)
    parser.add_argument("--run-token")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--wire-tail-tokens", type=int, default=32)
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=256,
        help="Maximum tokens per prompt/cache-extension forward pass.",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa"),
        default="eager",
        help="Dense attention backend for plain and PRA-100 requests.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Opt in to BitsAndBytes NF4 loading; the default load path is unchanged.",
    )
    parser.add_argument(
        "--max-gpu-memory-mib",
        type=int,
        help=(
            "Cap device 0 placement during Accelerate dispatch. Requires "
            "--device-map auto and permits overflow modules to reside on CPU."
        ),
    )
    parser.add_argument(
        "--max-cpu-memory-gib",
        type=int,
        default=64,
        help="CPU placement budget used with --max-gpu-memory-mib (default: 64 GiB).",
    )
    parser.add_argument(
        "--offload-folder",
        type=Path,
        help="Optional Accelerate disk-offload directory for the bounded loader.",
    )
    parser.add_argument(
        "--chat-template-profile",
        choices=("auto", "native", "qwen3-stable-no-thinking", "pure-chatml-stable"),
        default="auto",
    )
    args = parser.parse_args()
    served_model = args.served_model or args.model
    if args.max_gpu_memory_mib is not None:
        if args.max_gpu_memory_mib <= 0:
            parser.error("--max-gpu-memory-mib must be positive")
        if args.max_cpu_memory_gib <= 0:
            parser.error("--max-cpu-memory-gib must be positive")
        if args.device_map != "auto":
            parser.error("--max-gpu-memory-mib requires --device-map auto")
    if args.offload_folder is not None:
        args.offload_folder.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pra_hf.agent_executor import (
        HFAgentHistoryExecutor,
        configure_append_stable_template,
        resolve_append_stable_template_profile,
        validate_append_stable_template,
    )
    from experiments.paper4_5_agent.runtime_identity import (
        require_source_checkpoint_revision,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    chat_template_profile = resolve_append_stable_template_profile(
        tokenizer, args.chat_template_profile
    )
    template_digest = configure_append_stable_template(
        tokenizer, chat_template_profile
    )
    validate_append_stable_template(
        tokenizer,
        chat_template_kwargs=(
            {"enable_thinking": False}
            if chat_template_profile == "qwen3-stable-no-thinking"
            else None
        ),
    )
    model_kwargs: dict[str, Any] = {
        "revision": args.revision,
        "device_map": args.device_map,
        "torch_dtype": "auto",
        "attn_implementation": args.attn_implementation,
    }
    if args.max_gpu_memory_mib is not None:
        model_kwargs["max_memory"] = {
            0: f"{args.max_gpu_memory_mib}MiB",
            "cpu": f"{args.max_cpu_memory_gib}GiB",
        }
        model_kwargs["offload_state_dict"] = True
    if args.offload_folder is not None:
        model_kwargs["offload_folder"] = str(args.offload_folder)
    if args.load_in_4bit:
        import torch
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=(
                args.max_gpu_memory_mib is not None
            ),
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    observed_revision = require_source_checkpoint_revision(
        args.model,
        args.revision,
        runtime_config=getattr(model, "config", None),
    )
    executor = HFAgentHistoryExecutor(
        model,
        tokenizer,
        model_id=served_model,
        model_revision=args.revision,
        wire_tail_tokens=args.wire_tail_tokens,
        prefill_step_size=args.prefill_step_size,
        max_model_len=args.max_model_len,
        chat_template_profile=chat_template_profile,
        chat_template_digest=template_digest,
    )
    executor.dense_attention_implementation = args.attn_implementation
    executor.load_in_4bit = args.load_in_4bit
    executor.cpu_offload_enabled = args.max_gpu_memory_mib is not None
    executor.max_gpu_memory_mib = args.max_gpu_memory_mib
    executor.max_cpu_memory_gib = (
        args.max_cpu_memory_gib if args.max_gpu_memory_mib is not None else None
    )
    executor.hf_device_map = {
        str(name): str(device)
        for name, device in dict(getattr(model, "hf_device_map", {}) or {}).items()
    }
    try:
        ThreadingHTTPServer(
            (args.host, args.port),
            _direct_handler(
                executor,
                served_model,
                {
                    "engine": "huggingface",
                    "observed_source_checkpoint_revision": observed_revision,
                },
                args.run_token,
            ),
        ).serve_forever()
    finally:
        executor.close()


if __name__ == "__main__":
    main()
